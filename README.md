# T-CIL: Temperature Scaling using Adversarial Perturbation for Calibration in Class-Incremental Learning
This repository contains the official implementation of our paper accepted at CVPR 2025.

## Abstract
T-CIL addresses the critical challenge of model calibration in class-incremental learning scenarios. Our approach combines temperature scaling with adversarial perturbations to maintain well-calibrated predictions as new classes are incrementally learned using only a new-task validation set.

## Quick start
```
### Command Line Usage
```bash
# Run with default parameters (CIFAR-100, 10 tasks, 5 seeds)
python main.py

# Quick test with single seed
python main.py --seeds 1

# CIFAR-10 experiment
python main.py --dataset cifar10 --num_tasks 5 --num_classes 10
```
