# model_training.py
# Trains an EEG-based mental health prediction model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess

# Load preprocessed data
X_train, X_test, y_train, y_test = load_and_preprocess('data/sample_eeg.csv')

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {acc:.2f}")
