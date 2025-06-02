import torch
import numpy as np
import random
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed: Integer seed for random number generation
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_id: int = 0) -> torch.device:
    """Get PyTorch device (GPU if available, else CPU).
    
    Args:
        device_id: GPU device index to use
        
    Returns:
        torch.device: Selected computation device
    """
    return torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

def get_data_loaders() -> Tuple[datasets.CIFAR100, datasets.CIFAR100]:
    """Create CIFAR100 train and test datasets with appropriate transforms.
    
    Returns:
        tuple: (train_data, test_data)
            - train_data: CIFAR100 training dataset
            - test_data: CIFAR100 test dataset
    """
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_data = datasets.CIFAR100(root='../CIFAR-100/data/02/',
                                   train=True,
                                   download=False,
                                   transform=transform_train)
    test_data = datasets.CIFAR100(root='../CIFAR-100/data/02/',
                                  train=False,
                                  download=False,
                                  transform=transform_test)
    return train_data, test_data

def split_data(train_data: datasets.CIFAR100, test_data: datasets.CIFAR100, 
               num_step: int, num_class: int, num_class_per_task: int) -> Tuple[Dict, Dict, Dict, Dict]:
    """Split CIFAR100 data into tasks for continual learning.
    
    Args:
        train_data: CIFAR100 training dataset
        test_data: CIFAR100 test dataset
        num_step: Number of training steps/tasks
        num_class: Total number of classes
        num_class_per_task: Number of classes per task
        
    Returns:
        tuple: (train_task, test_task, train_class_idx, test_class_idx)
            - train_task: Dict mapping task ID to training data subset
            - test_task: Dict mapping task ID to test data subset
            - train_class_idx: Dict mapping class ID to training indices
            - test_class_idx: Dict mapping class ID to test indices
    """
    train_task = {x: [] for x in range(num_step)}
    test_task = {x: [] for x in range(num_step)}
    train_class_idx = {x: [] for x in range(num_class)}
    test_class_idx = {x: [] for x in range(num_class)}

    for cnt, (_, y) in enumerate(train_data):
        train_class_idx[y].append(cnt)

    for cnt, (_, y) in enumerate(test_data):
        test_class_idx[y].append(cnt)

    for i in range(num_step):
        curr_task_idx_train = []
        curr_task_idx_test = []
        for j in range(num_class_per_task):
            curr_task_idx_train += train_class_idx[i * num_class_per_task + j]
            curr_task_idx_test += test_class_idx[i * num_class_per_task + j]
        train_task[i] = torch.utils.data.Subset(train_data, curr_task_idx_train)
        test_task[i] = torch.utils.data.Subset(test_data, curr_task_idx_test)

    return train_task, test_task, train_class_idx, test_class_idx

def cal_ece(model: nn.Module, test_dataset: torch.utils.data.Dataset, 
            num_class: int, num_class_per_task: int, batch_size: int,
            n_bins: int = 10, temperature: float = 1, 
            device: Optional[torch.device] = None, 
            method_type: Optional[str] = None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Calculate Expected Calibration Error (ECE) for model predictions.
    
    Args:
        model: Neural network model
        test_dataset: Dataset for evaluation
        num_class: Total number of classes
        num_class_per_task: Number of classes per task
        batch_size: Batch size for evaluation
        n_bins: Number of confidence bins for ECE calculation
        temperature: Temperature scaling parameter
        device: Computation device (CPU/GPU)
        method_type: Training method type ('DER' or other)
        
    Returns:
        tuple: (ece, Bm, acc, conf, avg_acc, avg_conf)
            - ece: Expected Calibration Error
            - Bm: Sample counts per bin
            - acc: Accuracy per bin
            - conf: Confidence per bin
            - avg_acc: Average accuracy
            - avg_conf: Average confidence
    """
    model.eval()
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size = batch_size, shuffle = False)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if method_type == 'DER':
                output = model(data)['logits']
            else:
                output = model(data)[:, :num_class]
            pred_max = torch.max(output, dim = -1)
            pred_idx = pred_max[1]
            for i in range(len(data)):
                scaled_conf = torch.max(F.softmax(output[i]/temperature, dim = -1), dim = -1)[0]
                
                temp_idx = int(torch.floor(scaled_conf*n_bins))
                if temp_idx == n_bins:
                    temp_idx = n_bins-1
                Bm[temp_idx] += 1
                if pred_idx[i] == target[i]:
                    acc[temp_idx] += 1
                conf[temp_idx] += scaled_conf
                
    for m in range(n_bins):
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    avg_acc = 0
    avg_conf = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
        avg_acc += Bm[m] * acc[m] / sum(Bm)
        avg_conf += Bm[m] * conf[m] / sum(Bm)
    return ece / sum(Bm), Bm, acc, conf, avg_acc, avg_conf

def cal_aece(model: nn.Module, test_dataset: torch.utils.data.Dataset,
             num_class: int, num_class_per_task: int, batch_size: int,
             n_bins: int = 10, temperature: float = 1,
             device: Optional[torch.device] = None,
             method_type: Optional[str] = None) -> Tuple[float, List[Dict], float, float]:
    """Calculate Adaptive Expected Calibration Error (AECE) for model predictions.
    
    Args:
        model: Neural network model
        test_dataset: Dataset for evaluation
        num_class: Total number of classes
        num_class_per_task: Number of classes per task
        batch_size: Batch size for evaluation
        n_bins: Number of adaptive bins for AECE calculation
        temperature: Temperature scaling parameter
        device: Computation device (CPU/GPU)
        method_type: Training method type ('DER' or other)
        
    Returns:
        tuple: (aece, bins, avg_acc, avg_conf)
            - aece: Adaptive Expected Calibration Error
            - bins: List of dictionaries containing bin statistics
            - avg_acc: Average accuracy across all samples
            - avg_conf: Average confidence across all samples
    """
    model.eval()
    all_confidences = []
    all_accuracies = []
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if method_type == 'DER':
                output = model(data)['logits']
            else:
                output = model(data)[:, :num_class]
            pred_max = torch.max(output, dim=-1)
            pred_idx = pred_max[1]
            
            for i in range(len(data)):
                confidence = torch.max(F.softmax(output[i]/temperature, dim=-1), dim = -1)[0].item()
                accuracy = float(pred_idx[i] == target[i])
                
                all_confidences.append(confidence)
                all_accuracies.append(accuracy)
    
    # Sort samples by confidence
    sorted_indices = np.argsort(all_confidences)
    sorted_confidences = np.array(all_confidences)[sorted_indices]
    sorted_accuracies = np.array(all_accuracies)[sorted_indices]
    
    # Adaptive binning
    total_samples = len(sorted_confidences)
    samples_per_bin = total_samples // n_bins
    bins = []
    
    for i in range(n_bins):
        start_idx = i * samples_per_bin
        end_idx = (i + 1) * samples_per_bin if i < n_bins - 1 else total_samples
        bins.append({
            'confidences': sorted_confidences[start_idx:end_idx],
            'accuracies': sorted_accuracies[start_idx:end_idx],
        })
    
    # Calculate AECE
    aece = 0
    total_samples = sum(len(bin['confidences']) for bin in bins)
    
    for bin in bins:
        bin_size = len(bin['confidences'])
        if bin_size > 0:
            bin_acc = np.mean(bin['accuracies'])
            bin_conf = np.mean(bin['confidences'])
            aece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
    
    # Calculate average accuracy and confidence
    avg_acc = np.mean(all_accuracies)
    avg_conf = np.mean(all_confidences)
    
    return aece, bins, avg_acc, avg_conf


def tune_temp_batch_efficient(model: nn.Module, valid_data: torch.utils.data.Dataset,
                            num_class: int, epochs: int, batch_size: int,
                            device: torch.device, method_type: Optional[str] = None) -> torch.Tensor:
    """Efficiently tune temperature scaling parameter for better model calibration.
    
    Args:
        model: Neural network model
        valid_data: Validation dataset
        num_class: Total number of classes
        epochs: Number of optimization epochs
        batch_size: Batch size for optimization
        device: Computation device (CPU/GPU)
        method_type: Training method type ('DER' or other)
        
    Returns:
        torch.Tensor: Optimized temperature scaling parameter
    """
    model.eval()
    
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data, label in valid_loader:
            data = data.to(device)
            if method_type == 'DER':
                output = model(data)['logits']
            else:
                output = model(data)[:, :num_class]
            all_outputs.append(output.cpu())
            all_labels.append(label.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    temperature = torch.ones(1, requires_grad=True, device='cpu')
    
    optimizer = optim.SGD([temperature], lr=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    valid_dataset = torch.utils.data.TensorDataset(all_outputs, all_labels)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for output, label in valid_loader:
            loss = criterion(output / temperature, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return temperature