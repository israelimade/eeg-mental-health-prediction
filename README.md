# eeg-mental-health-prediction

Create folder **`src/`** with three minimal files:

**src/data_preprocessing.py**
```python
"""
Preprocess EEG: load csv, clean, feature engineer.
Uses ONLY synthetic/public data.
"""
import pandas as pd

def load_data(path="data/example_eeg.csv"):
    return pd.read_csv(path)

def basic_features(df):
    # placeholder: mean/std per channel
    feat = df.copy()
    return feat

if __name__ == "__main__":
    df = load_data()
    X = basic_features(df)
    X.to_csv("data/processed.csv", index=False)
