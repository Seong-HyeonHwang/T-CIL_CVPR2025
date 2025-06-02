import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from typing import Dict, Tuple
from torch.utils.data import DataLoader


class EmbeddingDistanceCalculator:
    """Calculator for computing embedding distances and finding nearest class centroids."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, method_type: str):
        """
        Args:
            model: Neural network model
            device: Device to run computations on (CPU/GPU)
            method_type: Training method type ('DER' or other)
        """
        self.model = model
        self.device = device
        self.method_type = method_type
        self.model.eval()

    def calculate_mean_vectors(self, buffer_loader: torch.utils.data.DataLoader) -> Dict[int, torch.Tensor]:
        """Calculate mean embedding vectors for each class in the buffer.
        
        Args:
            buffer_loader: DataLoader containing buffer samples
            
        Returns:
            Dictionary mapping class labels to their mean embedding vectors
        """
        mean_vectors = {}
        counts = {}
        
        with torch.no_grad():
            for data, labels in buffer_loader:
                data = data.to(self.device)
                if self.method_type == 'DER':
                    embeddings = self.model(data)['features']
                else:
                    embeddings = self.model.get_features(data)
                
                for embedding, label in zip(embeddings, labels):
                    label_item = label.item()
                    if label_item not in mean_vectors:
                        mean_vectors[label_item] = embedding.cpu()
                        counts[label_item] = 1
                    else:
                        mean_vectors[label_item] += embedding.cpu()
                        counts[label_item] += 1

        return {k: mean_vectors[k] / counts[k] for k in mean_vectors}

    def get_new_labels(self, inputs: torch.Tensor, labels: torch.Tensor, 
                      num_classes: int, mean_vectors: Dict[int, torch.Tensor], 
                      num_class_per_task: int) -> torch.Tensor:
        """Find new labels based on distance to class mean embeddings.
        
        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width)
            labels: Original class labels
            num_classes: Total number of classes
            mean_vectors: Dictionary of mean embeddings per class
            num_class_per_task: Number of classes in each task
            
        Returns:
            Tensor of new labels based on embedding distances
        """
        new_labels = torch.zeros_like(labels)
        mean_tensor = torch.stack([mean_vectors[i] for i in range(num_classes)]).to(self.device)
        current_task_class_threshold = num_classes - num_class_per_task
        with torch.no_grad():
            if self.method_type == 'DER':
                embeddings = self.model(inputs)['features']
            else:
                embeddings = self.model.get_features(inputs)
            
            for i, (embedding, label) in enumerate(zip(embeddings, labels)):
                distances = torch.norm(mean_tensor - embedding.unsqueeze(0), dim=1)
                label_val = label.item()
                
                if label_val >= current_task_class_threshold:
                    distances[label_val] = float('-inf')
                    new_labels[i] = distances.argmax()
                else:
                    distances[label_val] = float('inf')
                    new_labels[i] = distances.argmin()
                    
        return new_labels

class AdversarialTrainer:
    """Trainer for generating adversarial examples using embedding distances."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, method_type: str):
        """
        Args:
            model: Neural network model
            device: Device to run computations on (CPU/GPU)
            method_type: Training method type ('DER' or other)
        """
        self.model = model
        self.device = device
        self.method_type = method_type
        self.calculator = EmbeddingDistanceCalculator(model, device, method_type)

    def generate_adversarial_data(self, buffer_data: torch.utils.data.Dataset, 
                                valid_data: torch.utils.data.Dataset,
                                num_class: int, batch_size: int, 
                                epsilon: float, num_class_per_task: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate adversarial examples using FGSM attack.
        
        Args:
            buffer_data: Dataset containing buffer samples
            valid_data: Dataset containing validation samples
            num_class: Total number of classes
            batch_size: Batch size for processing
            epsilon: Perturbation magnitude for FGSM
            num_class_per_task: Number of classes per task
            
        Returns:
            Tuple of (adversarial_data, original_labels)
        """
        buffer_loader = torch.utils.data.DataLoader(buffer_data, batch_size=batch_size, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        
        mean_vectors = self.calculator.calculate_mean_vectors(buffer_loader)
        data_list, labels_list = [], []
        
        for data, labels in valid_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            data_adv = data.clone().detach().requires_grad_(True)

            least_labels = self.calculator.get_new_labels(data_adv, labels, num_class, mean_vectors, num_class_per_task)
            if self.method_type == 'DER':
                output = self.model(data_adv)['logits']
            else:
                output = self.model(data_adv)[:, :num_class]
            loss = -F.cross_entropy(output, least_labels)
            loss.backward()

            with torch.no_grad():
                data_adv = data_adv + epsilon * data_adv.grad.sign()

            data_list.append(data_adv.detach().cpu())
            labels_list.append(labels.cpu())
            
        return torch.cat(data_list), torch.cat(labels_list)

    def get_temperature(self, data_loader: torch.utils.data.DataLoader, 
                         num_class: int, num_task: int, epochs: int, 
                         batch_size: int) -> torch.Tensor:
        """Optimize temperature scaling parameter for better calibration.
        
        Args:
            data_loader: DataLoader containing samples
            num_class: Total number of classes
            num_task: Number of tasks
            epochs: Number of optimization epochs
            batch_size: Batch size for optimization
            
        Returns:
            Optimized temperature scaling parameter
        """
        self.model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for data, label in data_loader:
                data = data.to(self.device)
                if self.method_type == 'DER':
                    output = self.model(data)['logits']
                else:
                    output = self.model(data)[:, :num_class]
                all_outputs.append(output.cpu())
                all_labels.append(label.cpu())

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        temperature = torch.ones(1, requires_grad=True, device='cpu')
        optimizer = optim.SGD([temperature], lr=0.1)
        scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)

        valid_dataset = torch.utils.data.TensorDataset(all_outputs, all_labels)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for output, label in valid_loader:
                loss = F.cross_entropy(output / temperature, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        return temperature

def find_optimal_epsilon(trainer: AdversarialTrainer, buffer_data: torch.utils.data.Dataset,
                        valid_data: torch.utils.data.Dataset, target_temp: float,
                        num_class: int, num_task: int, epochs: int, 
                        batch_size: int, tolerance: float = 1e-3) -> float:
    """Find optimal epsilon for FGSM attack using binary search.
    
    Args:
        trainer: AdversarialTrainer instance
        buffer_data: Dataset containing buffer samples
        valid_data: Dataset containing validation samples
        target_temp: Target temperature value
        num_class: Total number of classes
        num_task: Number of tasks
        epochs: Number of temperature optimization epochs
        batch_size: Batch size for processing
        tolerance: Binary search tolerance
        
    Returns:
        Optimal epsilon value for FGSM attack
    """
    epsilon_low, epsilon_high = 0.0, 1.0
    num_class_per_task = num_class//num_task
    while epsilon_high - epsilon_low > tolerance:
        epsilon = (epsilon_low + epsilon_high) / 2.0
        adv_data, labels = trainer.generate_adversarial_data(buffer_data, valid_data, 
                                                           num_class, batch_size, epsilon, num_class_per_task)
        dataset = torch.utils.data.TensorDataset(adv_data, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        temperature = trainer.get_temperature(loader, num_class, num_task, epochs, batch_size).item()
        
        if temperature < target_temp:
            epsilon_low = epsilon
        else:
            epsilon_high = epsilon
            
    return (epsilon_low + epsilon_high) / 2.0