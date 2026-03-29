# Hybrid Deep Learning Ensemble for Phishing URL Detection

A multi-class URL classification system that combines CNN, BiLSTM, and Transformer architectures to detect phishing and malicious URLs with 99.5%+ accuracy.


## Overview

This project builds a hybrid deep learning ensemble that classifies URLs into 5 categories (benign or various phishing/malicious types). Three parallel neural network branches each extract different feature representations, which are then fused by a meta-learner for the final prediction.

## Model Architecture

- **CNN Branch** — Multi-scale 1D convolutions for local pattern extraction
- **BiLSTM Branch** — Bidirectional LSTM for sequential feature modeling
- **Transformer Branch** — Multi-head self-attention mechanism
- **Meta-Learner** — Dense fusion layers that combine all three branches into a softmax output

## Pipeline

1. Exploratory Data Analysis (class distribution, missing values, correlations)
2. Preprocessing (outlier clipping, median imputation, RobustScaler normalization)
3. Feature Selection (Variance Threshold + Mutual Information, top 60 features)
4. Model Training (Adam optimizer, early stopping, class-balanced weights)
5. Ablation Study (CNN-only, BiLSTM-only, Transformer-only, pairwise combos vs. full ensemble)
6. Evaluation (accuracy, precision, recall, F1, confusion matrix, ROC curves)

## Requirements

```
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install with:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```





## Dataset

- **File:** `All.csv`
- **Target column:** `URL_Type_obf_Type`
- **Classes:** 5 URL categories