# Multiclass Perceptron Classifier

## Overview

This project implements a **Multiclass Perceptron** from scratch and evaluates its performance using the **Digits dataset** from Scikit-Learn. The perceptron is a fundamental machine learning algorithm used for classification tasks, particularly in **linear classification** problems. This implementation supports **tracking the weight history** across different training iterations, allowing for better understanding and visualization of the learning process.

## Features
- **Custom Multiclass Perceptron Implementation**: Built using NumPy with configurable hyperparameters.
- **Weight Tracking**: Stores weight vectors after each update for analysis.
- **Dataset**: Uses the **Digits dataset** from `sklearn.datasets`, containing 8x8 pixel grayscale images of handwritten digits (0-9).
- **Performance Evaluation**:
  - **Accuracy Score**
  - **Mean Squared Error (MSE)**
  - **Zero-One Loss**
  - **Log Loss**
- **Hyperparameter Tuning**:
  - Different **learning rates (eta)** are tested to analyze performance impact.
  - The algorithm runs until either the **maximum iteration count** is reached or the **error threshold** is met.

## Dataset
The **Digits dataset** contains **1797 grayscale images** of digits 0 through 9, represented as **8×8 pixel arrays**. Each image is flattened into a **64-dimensional feature vector**.

## Implementation Details
- **Step 1**: Load the **Digits dataset** and preprocess it.
- **Step 2**: Implement a **Multiclass Perceptron** with adjustable parameters:
  - `eta`: Learning rate
  - `tau_max`: Maximum iterations
  - `epsilon`: Error threshold for early stopping
- **Step 3**: Train and test the perceptron using different learning rates.
- **Step 4**: Evaluate the model's accuracy and error rates.
- **Step 5**: Store weight history for visualization and analysis.

## Results and Analysis
- The perceptron’s performance varies depending on the chosen learning rate.
- The error decreases with training iterations, but higher learning rates may lead to **instability**.
- The weight history can be visualized to understand how the model learns.

## Requirements
To run this project, install the following dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib
