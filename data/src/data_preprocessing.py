# data_preprocessing.py
# Prepares EEG dataset for model training

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    # Load EEG dataset
    df = pd.read_csv(filepath)
    X = df.drop('label', axis=1)
    y = df['label']

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
