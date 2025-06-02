# T-CIL: Temperature Scaling using Adversarial Perturbation for Calibration in Class-Incremental Learning
This repository contains the official implementation of our paper accepted at CVPR 2025.
[![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/pdf/2503.22163)
## Abstract
T-CIL addresses the critical challenge of model calibration in class-incremental learning scenarios using only a new-task validation set. Our approach perturbs exemplars from memory and optimizes the temperature on the perturbed samples. T-CIL outperforms post-hoc calibration baselines and is compatible with existing class-incremental learning techniques.

## Quick start
Since T-Cil is a post-hoc calibration approach, the pretrained model is necessary.
### Command Line Usage
```bash
# Run with default parameters (CIFAR-100, 10 tasks, 5 seeds)
python main.py

# Quick test with single seed
python main.py --seeds 1

# CIFAR-10 experiment
python main.py --dataset cifar10 --num_tasks 5 --num_classes 10
```
