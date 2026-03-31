"""
Train and save Random Forest models for Turbofan Engine Predictive Maintenance.
Run this once to generate the pkl files used by the Streamlit app.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("final_dataset.csv")

# Strip whitespace from column names (sensor_4_min has trailing spaces)
df.columns = df.columns.str.strip()

print(f"Dataset shape: {df.shape}")
print(f"Failure type distribution:\n{df['failure_type'].value_counts()}")

# ── Define feature columns (144 engineered features) ─────────────────────────
COLS_TO_DROP = ["engine_id", "failed", "failure_cycle", "failure_type"]
feature_cols = [c for c in df.columns if c not in COLS_TO_DROP]
print(f"\nNumber of features: {len(feature_cols)}")

X = df[feature_cols]
y_type = df["failure_type"]          # classification target

# ── Compute RUL as max_cycle - failure_cycle (so lower = closer to failure) ──
# This gives a proper RUL: engines with high failure_cycle have more life left
max_cycle = df["failure_cycle"].max()
y_rul = max_cycle - df["failure_cycle"]   # RUL range: 0 to (max-min)
print(f"RUL range after transform: {y_rul.min()} to {y_rul.max()}, mean={y_rul.mean():.0f}")

# ── Imputer ───────────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# ── Label encoder ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y_type)
print(f"Classes: {le.classes_}")

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_rul_train, y_rul_test, y_cls_train, y_cls_test = train_test_split(
    X_imputed, y_rul, y_encoded, test_size=0.2, random_state=42
)

# ── Random Forest Regressor ───────────────────────────────────────────────────
print("\nTraining RF Regressor...")
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_rul_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error
y_pred_rul = rf_reg.predict(X_test)
mae = mean_absolute_error(y_rul_test, y_pred_rul)
rmse = np.sqrt(mean_squared_error(y_rul_test, y_pred_rul))
print(f"  MAE: {mae:.2f} cycles | RMSE: {rmse:.2f} cycles")

# ── Random Forest Classifier ──────────────────────────────────────────────────
print("\nTraining RF Classifier...")
rf_cls = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_cls.fit(X_train, y_cls_train)

from sklearn.metrics import accuracy_score
y_pred_cls = rf_cls.predict(X_test)
acc = accuracy_score(y_cls_test, y_pred_cls)
print(f"  Accuracy: {acc*100:.2f}%")

# ── Save artifacts ────────────────────────────────────────────────────────────
with open("rf_regressor.pkl", "wb") as f:
    pickle.dump(rf_reg, f)

with open("rf_classifier.pkl", "wb") as f:
    pickle.dump(rf_cls, f)

with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save feature column names so the app can align inputs correctly
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("\nAll artifacts saved: rf_regressor.pkl, rf_classifier.pkl, imputer.pkl, label_encoder.pkl, feature_cols.pkl")
