# ✈️ Turbofan Engine Predictive Maintenance

Predicts **Remaining Useful Life (RUL)** and **failure type** of turbofan engines using Random Forest models trained on the NASA C-MAPSS dataset. Includes an interactive Streamlit dashboard.

---

## Requirements

- Python 3.9 or higher
- Windows 10 / 11

---

## Setup

**1. Clone or download the repository**

```cmd
git clone <your-repo-url>
cd nash-pro
```

**2. Create a virtual environment**

```cmd
python -m venv .venv
```

**3. Activate the virtual environment**

```cmd
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your terminal prompt.

**4. Install dependencies**

```cmd
pip install streamlit==1.50.0 pandas==2.3.3 numpy==2.0.2 scikit-learn==1.6.1 plotly==6.6.0 python-pptx
```

---

## Files

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard |
| `train_models.py` | Trains and saves the ML models |
| `final_dataset.csv` | Pre-engineered training dataset |
| `sample_input.csv` | Ready-to-use demo CSV (7 engines) |
| `rf_regressor.pkl` | Trained RUL regression model |
| `rf_classifier.pkl` | Trained failure type classifier |
| `imputer.pkl` | Fitted imputer |
| `label_encoder.pkl` | Fitted label encoder |
| `feature_cols.pkl` | Feature column names |

> The `.pkl` model files are already included. You only need to run `train_models.py` if you want to retrain from scratch.

---

## Run the App

```cmd
.venv\Scripts\activate
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## Retrain Models (optional)

If you want to retrain the models from scratch:

```cmd
.venv\Scripts\activate
python train_models.py
```

This will overwrite the existing `.pkl` files.

---

## Using the Dashboard

1. Open the app in your browser at `http://localhost:8501`
2. Go to the **Upload & Predict** tab
3. Upload `sample_input.csv` (included) or your own CSV
4. View RUL predictions, failure types, and fleet charts

**Accepted CSV formats:**

- Raw sensor data (multi-cycle): `engine_id, cycle, op_setting_1..3, sensor_1..21`
- Pre-engineered features (one row per engine): `engine_id, op_setting_1_avg_30, ..., sensor_21_var` (144 feature columns)

---

## Deactivate the Environment

```cmd
deactivate
```
