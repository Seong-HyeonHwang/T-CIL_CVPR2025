# T-CIL: Temperature Scaling using Adversarial Perturbation for Calibration in Class-Incremental Learning
This repository contains the official implementation of our paper accepted at CVPR 2025.
[![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/pdf/2503.22163)
## Abstract
T-CIL addresses the critical challenge of model calibration in class-incremental learning scenarios using only a new-task validation set. Our approach perturbs exemplars from memory and optimizes the temperature on the perturbed samples. T-CIL outperforms post-hoc calibration baselines and is compatible with existing class-incremental learning techniques.

## Quick start
**Note**: T-CIL is a post-hoc calibration method that requires pretrained models. Make sure you have saved models from your incremental learning experiments.
### Command Line Usage
```bash
# Run with default parameters (CIFAR-100, 10 tasks, 5 seeds)
python main.py

# Quick test with single seed
python main.py --seeds 1

# CIFAR-10 experiment
python main.py --dataset cifar10 --num_tasks 5 --num_classes 10
```

## Citation
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{hwang2025t,
  title={T-CIL: Temperature Scaling using Adversarial Perturbation for Calibration in Class-Incremental Learning},
  author={Hwang, Seong-Hyeon and Kim, Minsu and Whang, Steven Euijong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={15339--15348},
  year={2025}
}
```
