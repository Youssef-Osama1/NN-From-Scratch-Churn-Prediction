# Telco Customer Churn Prediction with Neural Network from Scratch

## Overview

This project implements a deep neural network from scratch using only NumPy to predict customer churn based on the "WA_Fn-UseC_-Telco-Customer-Churn" dataset. The implementation closely follows the concepts taught in Andrew Ng's Deep Learning Specialization on Coursera, focusing on building the core components of a neural network, including forward propagation, backward propagation, cost calculation, and parameter updates using the Adam optimizer.

The goal is to demonstrate an understanding of the underlying mechanics of neural networks by building one without relying on high-level deep learning frameworks like TensorFlow or PyTorch for the network structure itself.

## Dataset

The project uses the publicly available Telco Customer Churn dataset (typically found as `WA_Fn-UseC_-Telco-Customer-Churn.csv`). This dataset contains information about telecom customers and whether they churned (left the service) or not.

Features include:
*   Customer demographics (gender, SeniorCitizen, Partner, Dependents)
*   Account information (tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges)
*   Services subscribed to (PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
*   The target variable: `Churn` (Yes/No)

## Implementation Details

The Python script `telco_churn_nn_from_scratch.py` performs the following steps:

1.  **Import Libraries:** Imports necessary libraries like Pandas, NumPy, Matplotlib, and Scikit-learn.
2.  **Data Loading and Exploration:** Loads the dataset using Pandas and performs initial exploration (`head()`, `info()`).
3.  **Data Preprocessing:** 
    *   Handles missing values in the `TotalCharges` column.
    *   Drops the `customerID` column.
    *   Encodes binary categorical features ('Yes'/'No') into 1/0.
    *   Encodes the `gender` feature into 1/0.
    *   Applies One-Hot Encoding to the remaining multi-class categorical features using `pd.get_dummies()`.
4.  **Data Splitting and Scaling:**
    *   Separates features (X) and the target variable (y).
    *   Splits the data into training (60%), validation (20%), and test (20%) sets using `train_test_split` with stratification to maintain class distribution.
    *   Scales the features using `StandardScaler`.
    *   Transposes the data matrices to match the expected input format of the neural network functions (features x samples).
5.  **Neural Network Implementation (from Scratch):**
    *   Defines helper functions for parameter initialization (He initialization), forward propagation (linear step, ReLU and Sigmoid activations, dropout), cost computation (cross-entropy with L2 regularization), backward propagation (handling activations, regularization, and dropout), and parameter updates using the Adam optimizer.
    *   Combines these helpers into the main `L_layer_model` function to train the network.
    *   Includes a `predict` function for making predictions on new data.
6.  **Model Training:**
    *   Defines the neural network architecture (e.g., `[input_features, 16, 1]`).
    *   Sets hyperparameters (learning rate, number of iterations, L2 regularization lambda, dropout keep probability).
    *   Trains the model using the training data and monitors cost on the validation set.
7.  **Cost Visualization:** Plots the training and validation cost over iterations using Matplotlib to visualize the learning process.
8.  **Model Evaluation:**
    *   Makes predictions on the training, validation, and test sets.
    *   Calculates and prints the accuracy for each set.
    *   Includes optional code to generate and display a confusion matrix and classification report for the test set using Scikit-learn, providing deeper insights into performance, especially given the potential class imbalance in the churn dataset.

## Future Work

*   **Handle Class Imbalance:** Implement techniques like SMOTE (Synthetic Minority Over-sampling Technique) or class weighting during training to potentially improve prediction performance for the minority (Churn) class.
*   **Hyperparameter Tuning:** Systematically tune hyperparameters (learning rate, layer sizes, regularization strength, dropout rate, Adam betas) using techniques like grid search or random search on the validation set.
*   **Experiment with Architectures:** Try different numbers of layers and units per layer.
*   **Compare with Frameworks:** Implement the same model using TensorFlow/Keras or PyTorch for comparison.
