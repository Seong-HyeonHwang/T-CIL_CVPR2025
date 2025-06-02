import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
from datetime import datetime
import copy
from torch.utils.data import DataLoader, Subset, TensorDataset
from model import resnet32
from utils import (
    set_seed, cal_ece, cal_aece, tune_temp_batch_efficient
)
from TCIL import AdversarialTrainer, find_optimal_epsilon

class CIFAR100_TCIL:
    def __init__(self, config):
        """
        Initialize CIFAR100 T-CIL evaluation
        
        Args:
            config (dict): Configuration dictionary containing:
                - cuda_device (int): GPU device ID to use
                - data_root (str): Path to CIFAR-100 dataset
                - model_path (str): Path to save/load model checkpoints
                - method_type (str): Training method type (e.g., 'ER' for Experience Replay)
                - num_tasks (int): Number of sequential tasks
                - num_classes (int): Total number of classes
                - classes_per_task (int): Number of classes per task
                - batch_size (int): Batch size for training/evaluation
                - epochs (int): Number of training epochs
                - val_size (int): Size of validation set
                - seed_list (list): List of random seeds for multiple runs
                - seed (int): Current random seed
        """
        self.config = config
        self.device = self._setup_device()
        self.transform_train, self.transform_test = self._setup_transforms()
        self.train_data, self.test_data = self._load_datasets()
        self.test_tasks = self._setup_test_tasks()
        
    def _setup_device(self):
        """Setup and return CUDA device"""
        device = torch.device(f'cuda:{self.config["cuda_device"]}' if torch.cuda.is_available() else 'cpu')
        print(f'Current cuda device is {device}')
        torch.set_num_threads(2)
        return device

    def _setup_transforms(self):
        """Setup data preprocessing transforms"""
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
        
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        return transform_train, transform_test

    def _load_datasets(self):
        """Load and return CIFAR100 datasets"""
        train_data = datasets.CIFAR100(
            root=self.config['data_root'],
            train=True,
            download=True,
            transform=self.transform_train
        )
        
        test_data = datasets.CIFAR100(
            root=self.config['data_root'],
            train=False,
            download=True,
            transform=self.transform_test
        )
        
        print(f'Number of training data: {len(train_data)}')
        print(f'Number of test data: {len(test_data)}')
        return train_data, test_data

    def _setup_test_tasks(self):
        """Split test data into tasks"""
        num_classes = self.config['num_classes']
        num_tasks = self.config['num_tasks']
        classes_per_task = num_classes // num_tasks
        
        # Create class indices
        test_class_idx = {x: [] for x in range(num_classes)}
        for idx, (_, label) in enumerate(self.test_data):
            test_class_idx[label].append(idx)
            
        # Create task datasets
        test_tasks = {}
        for task_id in range(num_tasks):
            task_indices = []
            for class_id in range(classes_per_task):
                class_idx = task_id * classes_per_task + class_id
                task_indices.extend(test_class_idx[class_idx])
            test_tasks[task_id] = [self.test_data[j] for j in task_indices]
            
        return test_tasks

    def evaluate(self):
        """Run T-CIL evaluation"""
        all_seeds_ece_overall = []
        all_seeds_aece_overall = []
        
        for seed in self.config['seed_list']:
            print(f"\n=== Running evaluation with seed {seed} ===")
            ece_overall_hist, aece_overall_hist = self._evaluate_seed(seed)
            
            print(f"[Seed {seed} Avg] ece_overall: {np.mean(ece_overall_hist)*100:.2f}, "
                  f"aece_overall: {np.mean(aece_overall_hist)*100:.2f}")
            
            all_seeds_ece_overall.append(np.mean(ece_overall_hist)*100)
            all_seeds_aece_overall.append(np.mean(aece_overall_hist)*100)
        
        self._print_final_results(all_seeds_ece_overall, all_seeds_aece_overall)

    def _evaluate_seed(self, seed):
        """
        Evaluate all tasks for a specific random seed
        
        Args:
            seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (ece_overall_hist, aece_overall_hist)
                - ece_overall_hist (list): ECE values for each task
                - aece_overall_hist (list): AECE values for each task
        """
        set_seed(seed)
        ece_overall_hist = []
        aece_overall_hist = []
        
        for task_id in range(self.config['num_tasks']):
            # Load model and prepare data
            model = self._load_model(seed, task_id)
            test_data = self._prepare_test_data(task_id)
            buffer_data = self._load_buffer_data(seed, task_id)
            
            # Process task evaluation
            ece, aece = self._evaluate_task(model, test_data, buffer_data, task_id)
            
            ece_overall_hist.append(ece)
            aece_overall_hist.append(aece)
            print(f"[Task {task_id}] ece_overall: {ece*100:.2f}, aece_overall: {aece*100:.2f}")
            
        return ece_overall_hist, aece_overall_hist

    def _load_model(self, seed, task_id):
        """Load pretrained model for given seed and task"""
        model_path = os.path.join(
            self.config['model_path'],
            f'{self.config["method_type"]}_{seed}_seed_{self.config["num_tasks"]}_tasks_'
            f'{self.config["val_size"]}_val_{task_id}_task.pt'
        )
        model = resnet32().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        return model

    def _prepare_test_data(self, task_id):
        """
        Prepare test data for current task evaluation by combining all previous tasks
        
        Args:
            task_id (int): Current task identifier
            
        Returns:
            list: Combined test data from all tasks up to current task
        """
        test_data = copy.deepcopy(self.test_tasks[0])
        for i in range(1, task_id + 1):
            test_data.extend(self.test_tasks[i])
        return test_data

    def _load_buffer_data(self, seed, task_id):
        """
        Load experience replay buffer data for a specific task
        
        Args:
            seed (int): Random seed
            task_id (int): Task identifier
            
        Returns:
            Subset: Subset of training data used as replay buffer
        """
        buffer_indices = np.load(
            f'./saved_buffer_indices/{self.config["method_type"]}_buffer_indices_'
            f'{seed}_seed_{self.config["num_tasks"]}_tasks_'
            f'{self.config["val_size"]}_val_{task_id}_task.npy'
        )
        return Subset(self.train_data, buffer_indices)

    def _evaluate_task(self, model, test_data, buffer_data, task_id):
        """
        Evaluate a single task's performance
        
        Args:
            model (nn.Module): The neural network model
            test_data (Dataset): Test dataset for evaluation
            buffer_data (Dataset): Experience replay buffer data
            task_id (int): Current task identifier
            
        Returns:
            tuple: (ece, aece)
                - ece (float): Expected Calibration Error
                - aece (float): Adaptive Expected Calibration Error
        """
        trainer = AdversarialTrainer(model, self.device, self.config["method_type"])
        
        # Get optimal temperature
        valid_indices = np.load(
            f'./saved_buffer_indices/{self.config["method_type"]}_valid_indices_'
            f'{self.config["seed"]}_seed_{self.config["num_tasks"]}_tasks_'
            f'{self.config["val_size"]}_val_{task_id}_task.npy'
        )
        valid_data = Subset(self.train_data, valid_indices)
        
        temperature_new_task_opt = tune_temp_batch_efficient(
            model, valid_data, 
            (task_id + 1) * self.config['classes_per_task'],
            self.config['epochs'], self.config['batch_size'], 
            self.device
        ).item()

        new_task_idx = []
        for j, (data, label) in enumerate(buffer_data):
            if label // self.config['classes_per_task'] == task_id:
                new_task_idx.append(j)
        new_task_data = torch.utils.data.Subset(buffer_data, new_task_idx)

        best_epsilon = find_optimal_epsilon(
            trainer=trainer,
            buffer_data=buffer_data,
            valid_data=new_task_data,
            target_temp=temperature_new_task_opt,
            num_class=(task_id + 1) * self.config['classes_per_task'],
            num_task=task_id + 1,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size']
        )
        
        adv_data, labels = trainer.generate_adversarial_data(buffer_data, buffer_data, 
                                                               (task_id + 1) * self.config['classes_per_task'], 
                                                               self.config['batch_size'], best_epsilon, self.config['classes_per_task'])
        adv_dataset = torch.utils.data.TensorDataset(adv_data, labels)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=self.config['batch_size'], shuffle=True)
            
        temperature = trainer.get_temperature(adv_loader, (task_id + 1) * self.config['classes_per_task'], task_id + 1, self.config['epochs'], self.config['batch_size']).item()
        
        # Calculate metrics
        ece, _, _, _, _, _ = cal_ece(
            model, test_data, 
            (task_id + 1) * self.config['classes_per_task'],
            self.config['classes_per_task'],
            self.config['batch_size'], 
            temperature=temperature, 
            device=self.device
        )
        
        aece, _, _, _ = cal_aece(
            model, test_data, 
            (task_id + 1) * self.config['classes_per_task'],
            self.config['classes_per_task'],
            self.config['batch_size'], 
            temperature=temperature, 
            device=self.device
        )
        
        return ece, aece

    def _print_final_results(self, ece_results, aece_results):
        """
        Print final evaluation results with mean and standard deviation
        
        Args:
            ece_results (list): ECE results across all seeds
            aece_results (list): AECE results across all seeds
        """
        final_ece_mean = np.mean(ece_results)
        final_ece_std = np.std(ece_results)
        final_aece_mean = np.mean(aece_results)
        final_aece_std = np.std(aece_results)
        
        print("\n=== Final Results ===")
        print(f"ECE Overall - Mean: {final_ece_mean:.2f}%, Std: {final_ece_std:.2f}%")
        print(f"AECE Overall - Mean: {final_aece_mean:.2f}%, Std: {final_aece_std:.2f}%")

def main():
    config = {
        'cuda_device': 0,
        'data_root': '../CIFAR-100/data/02/',
        'model_path': './saved_model/',
        'method_type': 'ER',
        'num_tasks': 10,
        'num_classes': 100,
        'classes_per_task': 10,
        'batch_size': 128,
        'epochs': 100,
        'val_size': 100,
        'seed_list': [1],
        'seed': 1
    }
    
    evaluator = CIFAR100_TCIL(config)
    evaluator.evaluate()

if __name__ == "__main__":
    main() 