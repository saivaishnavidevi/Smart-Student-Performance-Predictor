"""
train_model.py
--------------
Loads the synthetic student dataset, trains a RandomForestRegressor,
evaluates it on a held-out test set, and saves the model artifact plus
a metadata file recording the expected feature order.

DISCLAIMER: The dataset is entirely synthetic. Metrics here do not
establish real-world predictive performance.
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Paths ───────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(ROOT_DIR, "data", "students.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "model_artifacts")
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "student_performance_model.joblib")
META_PATH     = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

# ── Feature / target definitions ────────────────────────────────
FEATURES = ["study_hours", "attendance_pct", "prev_exam_score", "assignment_score"]
TARGET   = "final_score"

# ── 1. Load dataset ─────────────────────────────────────────────
print(f"Loading dataset from: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"ERROR: Dataset not found at '{DATA_PATH}'. "
          "Run data/generate_dataset.py first.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")

# ── 2. Validate columns ─────────────────────────────────────────
required_cols = FEATURES + [TARGET]
missing_cols  = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    sys.exit(1)

# ── 3. Validate no missing values ───────────────────────────────
null_counts = df[required_cols].isnull().sum()
if null_counts.any():
    print(f"ERROR: Missing values detected:\n{null_counts[null_counts > 0]}")
    sys.exit(1)

# ── 4. Validate all values are numeric ──────────────────────────
for col in required_cols:
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"ERROR: Column '{col}' contains non-numeric values.")
        sys.exit(1)

print("  Validation passed: no missing or non-numeric values.")

# ── 5. Prepare features and target ──────────────────────────────
X = df[FEATURES].values   # shape (300, 4), preserves FEATURES order
y = df[TARGET].values     # shape (300,)

# ── 6. Train / test split (80 / 20, no stratification) ──────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nSplit:  {len(X_train)} training samples / {len(X_test)} test samples")

# ── 7. Train RandomForestRegressor ──────────────────────────────
print("\nTraining RandomForestRegressor (n_estimators=100, random_state=42) …")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("  Training complete.")

# ── 8. Evaluate on held-out test set ────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
# mean_squared_error returns MSE; take sqrt manually (works across all sklearn versions)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n── Evaluation metrics (test set) ──────────────────────────")
print(f"  MAE  (Mean Absolute Error)       : {mae:.4f}")
print(f"  RMSE (Root Mean Squared Error)   : {rmse:.4f}")
print(f"  R²   (Coefficient of Determination): {r2:.4f}")

# ── 9. Save model artifact ──────────────────────────────────────
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to : {MODEL_PATH}")

# ── 10. Save metadata (feature names + order, target name) ──────
metadata = {
    "features": FEATURES,          # exact order expected at inference time
    "target":   TARGET,
    "n_estimators": 100,
    "random_state": 42,
    "disclaimer": (
        "Trained on a fully synthetic dataset. "
        "Results do not reflect real-world student performance."
    ),
}
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {META_PATH}")
