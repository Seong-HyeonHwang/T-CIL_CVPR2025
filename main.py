import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import copy
import os

from utils import cal_ece, cal_aece, tune_temp_batch_efficient, set_seed
from TCIL import AdversarialTrainer, find_optimal_epsilon
from model import resnet32

def parse_args():
    parser = argparse.ArgumentParser(description='T-CIL: Temperature Scaling using Adversarial Perturbation for Calibration in Class-Incremental Learning')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='../CIFAR-100/data/02/',
                       help='Root directory for dataset')
    
    # Task parameters
    parser.add_argument('--num_tasks', type=int, default=10,
                       help='Number of incremental learning tasks')
    parser.add_argument('--num_classes', type=int, default=100,
                       help='Total number of classes')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for temperature tuning')
    parser.add_argument('--val_size', type=int, default=100,
                       help='Validation set size')
    parser.add_argument('--method_type', type=str, default='ER',
                       help='Continual learning method type')
    
    # Experiment parameters
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                       help='Random seeds for experiments')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:X or cpu)')
    
    # Model saving/loading paths
    parser.add_argument('--model_path', type=str, default='./saved_model/',
                       help='Path to saved models')
    parser.add_argument('--buffer_path', type=str, default='./saved_buffer_indices/',
                       help='Path to saved buffer indices')
    
    return parser.parse_args()

def setup_dataset(args):
    """Setup dataset and transforms"""
    if args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                       std=[0.2675, 0.2565, 0.2761])
        num_classes = 100
    else:  # cifar10
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010])
        num_classes = 10
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if args.dataset == 'cifar100':
        train_data = datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=transform_test)
    else:
        train_data = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
    
    print(f'Number of training data: {len(train_data)}')
    print(f'Number of test data: {len(test_data)}')
    
    return train_data, test_data

def create_task_splits(train_data, test_data, num_tasks, num_classes):
    """Split dataset into tasks"""
    num_class_per_task = num_classes // num_tasks
    
    train_task = {x: [] for x in range(num_tasks)}
    test_task = {x: [] for x in range(num_tasks)}
    
    train_class_idx = {x: [] for x in range(num_classes)}
    test_class_idx = {x: [] for x in range(num_classes)}
    
    # Index training data by class
    for cnt, (data, label) in enumerate(train_data):
        train_class_idx[label].append(cnt)
    
    # Index test data by class
    for cnt, (data, label) in enumerate(test_data):
        test_class_idx[label].append(cnt)
    
    # Create task splits
    for i in range(num_tasks):
        curr_task_idx_train = []
        curr_task_idx_test = []
        for j in range(num_class_per_task):
            curr_task_idx_train += train_class_idx[i * num_class_per_task + j]
            curr_task_idx_test += test_class_idx[i * num_class_per_task + j]
        
        train_task[i] = [train_data[j] for j in curr_task_idx_train]
        test_task[i] = [test_data[j] for j in curr_task_idx_test]
    
    return train_task, test_task, num_class_per_task

def run_experiment(args, train_data, test_task, num_class_per_task, seed):
    """Run experiment for a single seed"""
    print(f"\n=== Running experiment with seed {seed} ===")
    
    ece_overall_hist = []
    aece_overall_hist = []
    set_seed(seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    for t in range(args.num_tasks):
        # Prepare test data up to current task
        test_task_total = copy.deepcopy(test_task[0])
        if t > 0:
            for i in range(1, t + 1):
                test_task_total += test_task[i]
        
        # Load model
        model = resnet32().to(device)
        model_path = f'{args.model_path}/{args.method_type}_{seed}_seed_{args.num_tasks}_tasks_{args.val_size}_val_{t}_task.pt'
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        # Load buffer and validation data
        buffer_indices_path = f'{args.buffer_path}/{args.method_type}_buffer_indices_{seed}_seed_{args.num_tasks}_tasks_{args.val_size}_val_{t}_task.npy'
        buffer_indices = np.load(buffer_indices_path)
        buffer = torch.utils.data.Subset(train_data, buffer_indices)
        
        valid_indices_path = f'{args.buffer_path}/{args.method_type}_valid_indices_{seed}_seed_{args.num_tasks}_tasks_{args.val_size}_val_{t}_task.npy'
        valid_new_task_indices = np.load(valid_indices_path)
        valid_new_task = torch.utils.data.Subset(train_data, valid_new_task_indices)
        
        # Tune temperature for new task
        temperature_new_task_opt = tune_temp_batch_efficient(
            model, valid_new_task, (t + 1) * num_class_per_task, 
            args.epochs, args.batch_size, device
        ).item()
        
        # Extract new task data from buffer
        new_task_idx = []
        for j, (data, label) in enumerate(buffer):
            if label // num_class_per_task == t:
                new_task_idx.append(j)
        new_task_data = torch.utils.data.Subset(buffer, new_task_idx)
        
        # Initialize adversarial trainer
        trainer = AdversarialTrainer(model, device, args.method_type)
        
        # Find optimal epsilon
        best_epsilon = find_optimal_epsilon(
            trainer=trainer,
            buffer_data=buffer,
            valid_data=new_task_data,
            target_temp=temperature_new_task_opt,
            num_class=(t + 1) * num_class_per_task,
            num_task=t + 1,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Generate adversarial data
        adv_data, labels = trainer.generate_adversarial_data(
            buffer, buffer, (t + 1) * num_class_per_task, 
            args.batch_size, best_epsilon, num_class_per_task
        )
        adv_dataset = torch.utils.data.TensorDataset(adv_data, labels)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Get final temperature
        temperature = trainer.get_temperature(
            adv_loader, (t + 1) * num_class_per_task, (t + 1), 
            args.epochs, args.batch_size
        ).item()
        
        # Calculate calibration metrics
        ece_overall, Bm, acc, conf, _, _ = cal_ece(
            model, test_task_total, (t + 1) * num_class_per_task, num_class_per_task,
            args.batch_size, n_bins=10, temperature=temperature, device=device
        )
        
        aece_overall, _, _, _ = cal_aece(
            model, test_task_total, (t + 1) * num_class_per_task, num_class_per_task,
            args.batch_size, n_bins=10, temperature=temperature, device=device
        )
        
        ece_overall_hist.append(ece_overall)
        aece_overall_hist.append(aece_overall)
        
        print(f"[Task {t}] ECE Overall: {ece_overall * 100:.2f}%, AECE Overall: {aece_overall * 100:.2f}%")
    
    seed_ece_avg = np.mean(ece_overall_hist) * 100
    seed_aece_avg = np.mean(aece_overall_hist) * 100
    
    print(f"[Seed {seed} Avg] ECE Overall: {seed_ece_avg:.2f}%, AECE Overall: {seed_aece_avg:.2f}%")
    
    return seed_ece_avg, seed_aece_avg

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup dataset
    train_data, test_data = setup_dataset(args)
    train_task, test_task, num_class_per_task = create_task_splits(
        train_data, test_data, args.num_tasks, args.num_classes
    )
    
    # Print configuration
    print("\n" + "="*60)
    print("T-CIL Experiment Configuration:")
    print("="*60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Classes per task: {num_class_per_task}")
    print(f"Method: {args.method_type}")
    print(f"Seeds: {args.seeds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")
    
    # Run experiments for all seeds
    all_seeds_ece_overall = []
    all_seeds_aece_overall = []
    
    for seed in args.seeds:
        seed_ece, seed_aece = run_experiment(args, train_data, test_task, num_class_per_task, seed)
        all_seeds_ece_overall.append(seed_ece)
        all_seeds_aece_overall.append(seed_aece)
    
    # Calculate final statistics
    final_ece_overall_mean = np.mean(all_seeds_ece_overall)
    final_ece_overall_std = np.std(all_seeds_ece_overall)
    final_aece_overall_mean = np.mean(all_seeds_aece_overall)
    final_aece_overall_std = np.std(all_seeds_aece_overall)
    
    # Print final results
    print("\n" + "="*60)
    print("=== FINAL RESULTS ===")
    print("="*60)
    print(f"ECE Overall  - Mean: {final_ece_overall_mean:.2f}%, Std: {final_ece_overall_std:.2f}%")
    print(f"AECE Overall - Mean: {final_aece_overall_mean:.2f}%, Std: {final_aece_overall_std:.2f}%")
    print("="*60)
    
    return {
        'ece_mean': final_ece_overall_mean,
        'ece_std': final_ece_overall_std,
        'aece_mean': final_aece_overall_mean,
        'aece_std': final_aece_overall_std
    }

if __name__ == "__main__":
    results = main()
