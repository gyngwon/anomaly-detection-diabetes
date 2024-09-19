# Diabetes Prediction and Anomaly Detection

## Introduction

This project focuses on predicting diabetes outcomes and detecting anomalies in patient health metrics using various machine learning techniques. The primary objective is to provide insights into diabetes prediction and identify irregular patterns in the dataset.

## Dataset Overview

The dataset used for this analysis is the Diabetes dataset, which includes features related to patients' health metrics such as Glucose, BMI, and Diabetes Pedigree Function. The target variable indicates whether a patient is diabetic (1) or not (0).

### Data Exploration

1. **Initial Inspection**:
   - The dataset is loaded and basic statistics are computed to understand its structure.
   - Missing values are checked, revealing no significant gaps.

2. **Target Variable Distribution**:
   - The target variable shows an imbalanced distribution between diabetic and non-diabetic patients, which may affect model performance.

## Data Preprocessing

1. **Feature Selection**:
   - The target variable (`Outcome`) is separated from the feature set.

2. **Oversampling**:
   <img width="497" alt="resempled-1" src="https://github.com/user-attachments/assets/c91c6ec5-e96e-4d4b-a7fd-83e20d8c3307">
   - The `RandomOverSampler` technique is employed to balance the class distribution by oversampling the minority class to achieve a target ratio of 95%.

4. **Adding Noise**:
   - Gaussian noise is added to certain features (`BMI` and `DiabetesPedigreeFunction`) to simulate variations in the data.

5. **Standardization**:
   - Features are standardized using `StandardScaler` to normalize the data.

## Model Development

### Logistic Regression

1. **Model Training**:
   - A Logistic Regression model is trained on the resampled dataset.
   - Feature importance is derived from model coefficients.

2. **Visualization**:
   - A bar plot displays the importance of each feature in predicting diabetes outcomes.

### Random Forest Classifier

1. **Model Training**:
   - A Random Forest Classifier is utilized for its robustness and ability to handle imbalanced data.
   - Hyperparameter tuning is conducted using `RandomizedSearchCV` to optimize model performance.

### Anomaly Detection

1. **Isolation Forest**:
   - An Isolation Forest model is trained on selected features (`Glucose` and `DiabetesPedigreeFunction`) to identify anomalies.
   - Performance metrics such as accuracy, precision, recall, and F1 score are calculated.

2. **One-Class SVM**:
   - A One-Class SVM is also employed for anomaly detection, with hyperparameter tuning using `RandomizedSearchCV`.

3. **Elliptic Envelope**:
   - An Elliptic Envelope model is trained to detect anomalies, focusing solely on the `Glucose` feature.

### Autoencoder

1. **Model Design**:
   - An autoencoder is designed to reconstruct the input features, trained solely on normal data to identify anomalies based on reconstruction error.

2. **Training**:
   - The autoencoder is trained over 500 epochs, with validation on both normal and abnormal data.

## Results

### Performance Metrics

- The models are evaluated based on their confusion matrices and performance metrics:
  - **Accuracy**: Measures the proportion of true results among the total cases.
  - **Precision**: Indicates the proportion of positive identifications that were actually correct.
  - **Recall**: Measures the proportion of actual positives correctly identified.
  - **F1 Score**: The harmonic mean of precision and recall.

### Confusion Matrices

Confusion matrices for each model are presented to visualize the true positive, true negative, false positive, and false negative rates.

## Conclusion

The analysis successfully demonstrates the application of various machine learning techniques for diabetes prediction and anomaly detection. The logistic regression and random forest models show promising results in predicting diabetes outcomes, while the anomaly detection models effectively identify irregular patterns in the dataset. Future work could involve more complex models and feature engineering techniques to enhance predictive performance.

