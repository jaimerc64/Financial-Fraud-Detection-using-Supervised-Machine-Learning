# Financial Fraud Detection using Supervised Machine Learning

## Overview

This project develops and compares two supervised machine learning models for detecting fraudulent credit card transactions: **Logistic Regression** and **K-Nearest Neighbors (KNN)**.

The objective is to identify fraudulent transactions within a highly imbalanced dataset and evaluate which model provides the best performance in a real-world fraud detection scenario.

## Problem Statement

Financial institutions process millions of transactions daily, while fraudulent operations represent only a very small fraction of the total volume. This severe class imbalance makes fraud detection a challenging classification problem.

The goal of this project is to build predictive models capable of distinguishing legitimate transactions from fraudulent ones while minimizing false negatives.

## Dataset

* Dataset: Credit Card Fraud Detection Dataset
* Source: Kaggle
* Type: Binary Classification
* Classes:

  * 0 → Legitimate Transaction
  * 1 → Fraudulent Transaction

## Methodology

The project follows the SEMMA methodology:

1. **Sample**

   * Dataset selection and class distribution analysis.

2. **Explore**

   * Exploratory Data Analysis (EDA).
   * Variable distribution analysis.
   * Fraud vs. non-fraud comparison.

3. **Modify**

   * Feature scaling using StandardScaler.
   * Removal of irrelevant variables.
   * Handling class imbalance through data balancing techniques.
   * Train-test split.

4. **Model**

   * Logistic Regression.
   * K-Nearest Neighbors (KNN).

5. **Assess**

   * Model evaluation and comparison.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn

## Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

Special attention was given to **Recall**, as failing to detect fraudulent transactions can have significant financial consequences.

## Results

Both models successfully classified fraudulent and legitimate transactions.

The comparative analysis highlights the strengths and limitations of each approach regarding:

* Predictive performance
* Interpretability
* Computational efficiency
* Fraud detection capability


## Author

Jaime Rodríguez Contreras

Mathematical Engineering Student 
