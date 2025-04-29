#  Blood Group Prediction from Fingerprint Images

This project explores the feasibility of predicting human blood groups (A+, A-, B+, B-, AB+, AB-, O+, O-) using fingerprint images through classical machine learning and deep learning methods.

##  Project Overview

- **Objective:** Predict blood group types from fingerprint images.
- **Dataset:** 6000 fingerprint images labeled across 8 blood group classes.
- **Approaches Used:**
  - Classical ML: SVM, Random Forest
  - Deep Learning: Baseline CNN, MobileNet
- **Feature Extraction (ML):** Histogram of Oriented Gradients (HOG)
- **Performance:**
  - **SVM Accuracy:** ~91.6%
  - **Random Forest Accuracy:** ~87.8%
  - **Baseline CNN Accuracy:** ~98%
  - **MobileNet Accuracy:** ~85.7%

##  Models

### Classical ML Models
- **SVM (RBF Kernel):**
  - Feature extraction: HOG
  - StandardScaler normalization
- **Random Forest:**
  - 100 estimators
  - Feature importance visualized with SHAP and permutation

###  Deep Learning Models
- **Baseline CNN:**
  - 2 Conv layers → Flatten → Dense → Softmax
- **MobileNet:**
  - Fine-tuned pre-trained MobileNet
- **ResNet50:**
  - Intended but not fully trained due to hardware limits

> Note: Class imbalance was addressed using **random oversampling** in training data.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training/Validation Loss Curves

##  Setup Instructions

### Environment
```bash
python 3.9+
TensorFlow 2.x
scikit-learn
matplotlib
shap
