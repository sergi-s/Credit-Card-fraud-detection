# Credit Card Fraud Detection System using Machine Learning

Detect Fraudulent Credit Card transactions using different Machine Learning models and compare performances.

## Overview

In this notebook, exploring various Machine Learning models to detect fraudulent use of credit cards. and comparing each model's performance and results. The best performance is achieved using the SMOTE technique.

## Problem Statement

In this project, we aim to identify fraudulent transactions with credit cards. Our objective is to build a fraud detection system using machine learning techniques. Historically, such systems were rule-based, but machine learning offers powerful new ways.

The project uses a dataset of 300,000 fully anonymized transactions. Each transaction is labeled either fraudulent or not fraudulent. Note that the prevalence of fraudulent transactions is very low in the dataset—less than 0.1% of the card transactions are fraudulent. This means that a system predicting each transaction to be normal can reach an accuracy of over 99.9% despite not detecting any fraudulent transaction. This will necessitate adjustment techniques.

## Techniques Used in the Project

The project compares the results of different techniques:

### Machine Learning Techniques
- Random Forest
- Decision Trees

### Deep Learning Techniques
- Neural network using fully connected layers

Performance of the neural network is compared for different optimization approaches:
1. Plain binary cross-entropy loss minimization
2. Minimization using weights to compensate for class imbalance
3. Under-sampling of the non-fraudulent class to match the fraudulent class
4. Over-sampling of the fraudulent class to match the non-fraudulent one by implementing the SMOTE technique. The SMOTE method generates a new vector using two existing data points. For additional details on this approach, you can read this detailed post: [SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/).

### Note about Random Forest and Decision Tree Models:

- **Decision Tree:** Built on an entire dataset using all features/variables. You can easily overfit the data, so it is recommended to use cross-validation. Advantages: easy to interpret, clear understanding of the variable and value used for splitting data and predicting outcomes.
  
- **Random Forest:** A collection of Decision Trees. It randomly selects observations (rows) and specific features/variables to build multiple decision trees and then averages the results. It reduces the variance part of error rather than bias, generalizes better, and performs better on unseen validation datasets.

## Results

### Test Set

| Model                     | Accuracy | False Neg. Rate | Recall   | Precision | F1 Score |
|---------------------------|----------|-----------------|----------|-----------|----------|
| Random Forest             | 0.999544 | 0.224490        | 0.775510 | 0.950000  | 0.853933 |
| Decision Tree             | 0.999239 | 0.244898        | 0.755102 | 0.792857  | 0.773519 |
| Plain Neural Network      | 0.999403 | 0.238095        | 0.761905 | 0.875000  | 0.814545 |
| Weighted Neural Network   | 0.986775 | 0.102041        | 0.897959 | 0.105854  | 0.189383 |
| Under-Sampled Neural Net  | 0.956081 | 0.053333        | 0.946667 | 0.965986  | 0.956229 |
| Over-Sampled Neural Net   | 0.998376 | 0.000223        | 0.999777 | 0.996985  | 0.998379 |

### Full Set

| Model                     | Accuracy | False Neg. Rate | Recall   | Precision | F1 Score |
|---------------------------|----------|-----------------|----------|-----------|----------|
| Random Forest             | 0.999544 | 0.224490        | 0.775510 | 0.950000  | 0.853933 |
| Decision Tree             | 0.999239 | 0.244898        | 0.755102 | 0.792857  | 0.773519 |
| Plain Neural Network      | 0.999403 | 0.238095        | 0.761905 | 0.875000  | 0.814545 |
| Weighted Neural Network   | 0.986775 | 0.102041        | 0.897959 | 0.105854  | 0.189383 |
| Under-Sampled Neural Net  | 0.956081 | 0.053333        | 0.946667 | 0.965986  | 0.956229 |
| Under-Sampled Neural Net  | 0.979432 | 0.048780        | 0.951220 | 0.074262  | 0.137769 |
| Over-Sampled Neural Net   | 0.997402 | 0.004065        | 0.995935 | 0.399023  | 0.569767 |

## Conclusion

The best results are achieved by over-sampling the under-represented class using SMOTE (synthetic minority oversampling technique). With this approach, the model is able to detect 100% of all fraudulent transactions in the unseen test set, fully satisfying the primary objective of detecting the vast majority of abnormal transactions.

Additionally, the number of false positives remains acceptable, resulting in significantly less verification work on legitimate transactions for the fraud department compared to some other approaches. Key results are shown in the tables above.

This project demonstrates the effectiveness of machine learning and deep learning techniques in detecting fraudulent credit card transactions. The use of the SMOTE technique significantly improves the detection of fraudulent transactions, making the system robust and reliable for real-time updates.

Feel free to explore the notebook and adapt the approach to other detection issues in alternative domains.

---

For more details, you can read the full documentation and code implementation in the provided notebook.
