# ═══════════════════════════════════════════════════════
# Titanic Survival Classification Model
# Author: Sy Savane Usman
# Date: June 2026
# Description: Logistic regression model to predict
#              Titanic passenger survival probability
# Tools: Python, Pandas, scikit-learn
# Dataset: Kaggle Titanic Competition
# ═══════════════════════════════════════════════════════

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv("/kaggle/input/competitions/titanic/train.csv")

# ── Data Cleaning ────────────────────────────────────────
# Drop columns that are irrelevant or too complex to encode
df = df.drop(columns=["Name", "PassengerId", "Ticket", "Cabin"])

# Impute missing values
df["Age"] = df["Age"].fillna(round(df["Age"].mean()))         # Mean imputation for continuous variable
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # Mode imputation for categorical variable

# Encode categorical columns using one-hot encoding
# drop_first=True avoids multicollinearity (reference categories: Female, Cherbourg)
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Convert boolean columns to integers (0/1)
df = df.astype({col: int for col in df.select_dtypes(bool).columns})

# ── Model Training ───────────────────────────────────────
X = df.drop(columns=["Survived"])   # Feature matrix
y = df["Survived"]                  # Target vector

# 80/20 train-test split with fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ── Model Evaluation ─────────────────────────────────────
# Default threshold (0.5)
y_pred = model.predict(X_test)

print("=" * 55)
print("EVALUATION — Default Threshold (0.5)")
print("=" * 55)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Tuned threshold (0.4) — prioritises recall to catch more survivors
y_proba = model.predict_proba(X_test)[:, 1]
y_pred_tuned = [1 if p >= 0.4 else 0 for p in y_proba]

print("=" * 55)
print("EVALUATION — Tuned Threshold (0.4)")
print("=" * 55)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))
