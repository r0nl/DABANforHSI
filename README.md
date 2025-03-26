# Hyperspectral Image Classification Based on Domain Adversarial Broad Adaptation Network

## Overview

This project is the implementation of Hyperspectral Image Classification based on DABAN Architecture as mentioned in the research paper provided.

## System Architecture

The system consists of two main components:

### 1. Domain Adversarial Adaptation Network (DAAN)

- **Feature Extractor (Gf)**: Extracts deep features using 1D convolutional layers
- **Bottleneck Adaptation Module (Gb)**: Reduces dimensionality and adapts domain distributions
- **Domain Discriminator (Gd)**: Performs adversarial learning to align source and target domains
- **Classifier (Gc)**: Classifies extracted features based on source domain labels

### 2. Conditional Adaptation Broad Network (CABN)

- **Mapped Feature Layer**: Maps domain-invariant features to a common space
- **Feature Expansion Layer**: Expands features through random weights to enhance representation
- **Conditional Distribution Adaptation**: Aligns class-level distributions using MMD-based regularization

## Data Processing Pipeline

1. **Data Loading**: Hyperspectral images and labels are loaded from files
2. **Preprocessing**: Data normalization and reshaping for model input
3. **Feature Extraction**: Extracting domain-invariant features using DAAN
4. **Feature Adaptation**: Adapting domain distributions using adversarial learning
5. **Feature Expansion**: Expanding features through the CABN for improved representation
6. **Classification**: Final classification of target domain samples

## Implementation Details

### Domain Adaptation Mechanisms

The system implements three complementary domain adaptation mechanisms:

1. **Adversarial Adaptation**: Trains a domain discriminator to confuse source and target domains
2. **Marginal Distribution Alignment**: MMD loss to reduce the difference between domain distributions
3. **Second-Order Statistic Alignment**: CORAL loss to adapt the covariance structure between domains

### Key Components Implementation

#### MMD Loss Function
def compute_mmd_loss(source_features, target_features):
  mean_source = torch.mean(source_features, dim=0)
  mean_target = torch.mean(target_features, dim=0)
  loss_mmd = torch.sum((mean_source - mean_target) ** 2)
  return loss_mmd

#### CORAL Loss Function
def compute_coral_loss(source_features, target_features):
  source_cov = torch.matmul(source_features.T, source_features) / source_features.size(0)
  target_cov = torch.matmul(target_features.T, target_features) / target_features.size(0)
  loss_coral = torch.mean((source_cov - target_cov) ** 2)
  return loss_coral


## Dataset

https://drive.google.com/drive/folders/1sYcq9mjG4yEaA1LMNWZHY3vSPWRQcFQG?usp=drive_link

## Results and Performance

The model achieves state-of-the-art performance on standard hyperspectral image classification benchmarks:

- High classification accuracy even with limited labeled samples
- Effective knowledge transfer between source and target domains
- Robust performance across different hyperspectral sensors

## References

- H. Wang, Y. Cheng, C. L. Philip Chen, and X. Wang, "Hyperspectral Image Classification Based on Domain Adversarial Broad Adaptation Network," IEEE Transactions on Geoscience and Remote Sensing, 2022.
- Y. Linde, A. Buzo, and R. Gray, "An Algorithm for Vector Quantizer Design," IEEE Transactions on Communications, 1980.
