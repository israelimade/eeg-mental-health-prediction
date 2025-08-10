# EEG-Based Mental Health Prediction (Demo)

This repository demonstrates the workflow from my MSc dissertation:  
**“Use of Machine Learning to Predict Mental Health Disorders using EEG Dataset”** (University of Wolverhampton, 2024).

> ⚠️ **Privacy note:** No real data is included here. The example uses **synthetic** EEG-like features to illustrate the pipeline end-to-end.

## Project goal
Build a machine learning pipeline to classify major psychiatric disorder categories from EEG-derived features. In the dissertation, models included Logistic Regression, SVM, Random Forest, and a Soft Voting Ensemble with hyperparameter tuning and 5-fold cross-validation.

## What’s in this repo
- `src/generate_synthetic_data.py` — creates a small synthetic dataset with:
  - demographic columns (`sex`, `age`, `education`, `IQ`)
  - 60 EEG-like numeric features (`EEG_0 … EEG_59`)
  - a target label `main_disorder` with 7 classes (e.g., mood, anxiety, schizophrenia, etc.)
- `src/model_training.py` — trains Logistic Regression, SVM, and Random Forest using GridSearchCV, evaluates on a hold-out test set, and prints Accuracy/Precision/Recall/F1.

## Quick start

### 1) Install dependencies
```bash
pip install -r requirements.txt
python src/model_training.py
