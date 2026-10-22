"""
=============================================================
  TA-03 | Customer Churn Prediction
  04 - Simpan Model Terlatih dengan Joblib
=============================================================
Script ini melakukan:
  - Melatih ulang model final (pipeline SMOTE + LogReg)
    menggunakan seluruh data latih
  - Menyimpan model ke models/model_churn.pkl
  - Menyimpan daftar nama fitur ke models/feature_names.pkl
=============================================================
"""

import sys
import os
import warnings
import pandas as pd
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Setup path ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
#  1. MUAT DATA
# ══════════════════════════════════════════════════════════
print("=" * 55)
print("  MENYIMPAN MODEL FINAL")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

print(f"\n▸ Total data: {len(df):,} baris")
print(f"▸ Jumlah fitur: {X.shape[1]}")
print(f"▸ Fitur: {list(X.columns)}")

# ══════════════════════════════════════════════════════════
#  2. SPLIT DATA (untuk konsistensi dengan eval)
# ══════════════════════════════════════════════════════════
numeric_features     = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ══════════════════════════════════════════════════════════
#  3. BUAT DAN LATIH PIPELINE
# ══════════════════════════════════════════════════════════
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Model menggunakan parameter optimal C=10 hasil tuning GridSearchCV (05_validasi_model.py)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("smote",        SMOTE(random_state=42)),
    ("classifier",   LogisticRegression(C=10, max_iter=2000, random_state=42))
])

print("\n▸ Melatih model final (C=10, SMOTE) ...")
model.fit(X_train, y_train)
print("✔  Pelatihan selesai!")

# Evaluasi pada data uji untuk memastikan konsistensi
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\n" + "=" * 55)
print("  EVALUASI MODEL PADA DATA UJI (C=10)")
print("=" * 55)
print(f"  Akurasi  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"  Recall   : {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"  F1-Score : {f1_score(y_test, y_pred) * 100:.2f}%")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob) * 100:.2f}%")

# ══════════════════════════════════════════════════════════
#  4. SIMPAN MODEL DAN METADATA
# ══════════════════════════════════════════════════════════
model_path    = os.path.join(MODELS_DIR, "model_churn.pkl")
features_path = os.path.join(MODELS_DIR, "feature_names.pkl")

joblib.dump(model, model_path)
joblib.dump(list(X.columns), features_path)

print(f"\n✔  Model disimpan : {model_path}")
print(f"✔  Fitur disimpan : {features_path}")

# Verifikasi model bisa dimuat kembali
loaded = joblib.load(model_path)
pred   = loaded.predict(X_test[:3])
print(f"\n▸ Verifikasi prediksi (3 sampel): {pred.tolist()}")
print("\n" + "=" * 55)
print("  MODEL BERHASIL DISIMPAN!")
print("=" * 55)
