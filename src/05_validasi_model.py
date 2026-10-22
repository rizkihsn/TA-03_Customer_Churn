"""
=============================================================
  TA-03 | Customer Churn Prediction
  05 - Validasi Silang & Hyperparameter Tuning
=============================================================
Script ini melakukan:
  - GridSearchCV dengan StratifiedKFold (5-fold)
  - Mencari nilai C terbaik untuk LogisticRegression
  - Evaluasi model terbaik pada data uji
  - Visualisasi skor cross-validation per nilai C
  - Menyimpan model terbaik hasil tuning
=============================================================
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, recall_score, f1_score
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Setup path ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
IMG_DIR     = os.path.join(BASE_DIR, "..", "static", "img", "eda")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
#  1. MUAT DATA
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  VALIDASI SILANG & HYPERPARAMETER TUNING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_features     = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ══════════════════════════════════════════════════════════
#  2. PIPELINE
# ══════════════════════════════════════════════════════════
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote",        SMOTE(random_state=42)),
    ("classifier",   LogisticRegression(max_iter=2000, random_state=42))
])

# ══════════════════════════════════════════════════════════
#  3. GRID SEARCH CROSS-VALIDATION
# ══════════════════════════════════════════════════════════
param_grid = {"classifier__C": [0.01, 0.1, 1, 10, 100]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    return_train_score=True,
    n_jobs=-1,
    verbose=1
)

print("\n▸ Menjalankan GridSearchCV (5-fold)...")
search.fit(X_train, y_train)

# ══════════════════════════════════════════════════════════
#  4. TAMPILKAN HASIL CROSS-VALIDATION
# ══════════════════════════════════════════════════════════
results = pd.DataFrame(search.cv_results_)

print("\n" + "=" * 60)
print("  HASIL VALIDASI SILANG (AUC)")
print("=" * 60)
print(f"\n  {'C':>8} | {'Train AUC':>10} | {'Val AUC':>10} | {'Overfitting?':>12}")
print("  " + "-" * 50)
for _, row in results.iterrows():
    gap  = row["mean_train_score"] - row["mean_test_score"]
    flag = "⚠ Ya" if gap > 0.05 else "✔ Tidak"
    print(f"  {row['param_classifier__C']:>8} | "
          f"{row['mean_train_score']:>10.4f} | "
          f"{row['mean_test_score']:>10.4f} | "
          f"{flag:>12}")

best_C   = search.best_params_["classifier__C"]
best_auc = search.best_score_
print(f"\n  ★ Nilai C terbaik           : {best_C}")
print(f"  ★ Rata-rata Validation AUC  : {best_auc:.4f}")

# ══════════════════════════════════════════════════════════
#  5. EVALUASI MODEL TERBAIK PADA DATA UJI
# ══════════════════════════════════════════════════════════
best_model = search.best_estimator_
pred       = best_model.predict(X_test)
prob       = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("  EVALUASI MODEL TERBAIK PADA DATA UJI")
print("=" * 60)
print(f"\n  Akurasi  : {accuracy_score(y_test, pred):.4f}")
print(f"  Recall   : {recall_score(y_test, pred):.4f}  ← Prioritas utama!")
print(f"  F1-Score : {f1_score(y_test, pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, prob):.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, pred,
                             target_names=["Tidak Churn", "Churn"]))

# ══════════════════════════════════════════════════════════
#  PLOT 9 — Skor CV per Nilai C
# ══════════════════════════════════════════════════════════
c_values  = [str(c) for c in results["param_classifier__C"]]
train_auc = results["mean_train_score"].values
val_auc   = results["mean_test_score"].values

fig, ax = plt.subplots(figsize=(9, 5))
x_pos = range(len(c_values))
ax.plot(x_pos, train_auc, "o-", color="#3b82f6", lw=2.5, label="Train AUC", markersize=8)
ax.plot(x_pos, val_auc,   "s-", color="#f59e0b", lw=2.5, label="Validation AUC", markersize=8)

# Tandai nilai terbaik
best_idx = val_auc.argmax()
ax.axvline(best_idx, color="#10b981", lw=1.5, linestyle="--", alpha=0.7)
ax.scatter([best_idx], [val_auc[best_idx]], color="#10b981", zorder=5, s=100,
           label=f"Terbaik: C={best_C} (AUC={val_auc[best_idx]:.4f})")

ax.set_xticks(x_pos)
ax.set_xticklabels([f"C={c}" for c in c_values])
ax.set_xlabel("Nilai Parameter C (Regularisasi)", fontsize=11)
ax.set_ylabel("ROC-AUC Score", fontsize=11)
ax.set_title("Cross-Validation Score per Nilai C\n(5-Fold StratifiedKFold)",
             fontsize=13, fontweight="bold")
ax.legend()
ax.set_ylim(0.75, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "09_cv_scores.png"), dpi=110, bbox_inches="tight")
plt.close()
print("\n✔  Plot 9 disimpan: 09_cv_scores.png")

# ══════════════════════════════════════════════════════════
#  6. SIMPAN MODEL TERBAIK
# ══════════════════════════════════════════════════════════
best_path = os.path.join(MODELS_DIR, "model_churn.pkl")
joblib.dump(best_model, best_path)
print(f"✔  Model terbaik disimpan: {best_path}")

print("\n" + "=" * 60)
print("  VALIDASI SELESAI")
print("=" * 60)
