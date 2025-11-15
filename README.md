This repository contains a modular benchmarking framework for evaluating uncertainty quantification (UQ) techniques across different deep neural network architectures and multiple datasets. The goal is to analyze how well models estimate uncertainty under distributional shift and how effectively they detect out-of-distribution (OOD) samples.

Project Summary

This project benchmarks commonly used UQ techniques on pretrained deep learning models and evaluates their performance across various ID/OOD dataset combinations. The focus is on creating a unified, reproducible evaluation pipeline for uncertainty estimation in computer vision.

UQ Techniques Implemented

Vanilla Softmax Uncertainty

MC Dropout

Test-Time Augmentation (TTA)

(Additional methods such as Temperature Scaling, Dirichlet Meta-Models, Evidential DL, and GMMs will be added later.)

 Models Evaluated

ResNet-18 , VGG16-BN , ResNet34, EfficientNet-V2, ConvNeXt

Datasets Used
In-Distribution (ID)

Tiny-ImageNet (validation split)

Out-of-Distribution (OOD)

CIFAR-10, CIFAR-100, SVHN

These provide a mix of mild (CIFAR) and strong (SVHN) distribution shifts.

Metrics Computed

For every model × UQ method × dataset combination:

Predictive Entropy, Max Probability (Max-P), Expected Calibration Error (ECE)

AUROC for OOD detection using

Entropy

Max Probability

All results are automatically saved as CSV files.
