"""
=============================================================
  TA-03 | Customer Churn Prediction
  03 - Pelatihan Model Regresi Logistik + Evaluasi
=============================================================
Script ini melakukan:
  - Melatih model Regresi Logistik dengan pipeline SMOTE
  - Evaluasi: Confusion Matrix, Precision, Recall, F1, AUC
  - Visualisasi Confusion Matrix (heatmap)
  - Visualisasi ROC Curve
  - Interpretasi bisnis hasil model
=============================================================
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Setup path ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
IMG_DIR   = os.path.join(BASE_DIR, "..", "static", "img", "eda")
os.makedirs(IMG_DIR, exist_ok=True)

COLOR_OK  = "#22c55e"
COLOR_BAD = "#ef4444"

# ══════════════════════════════════════════════════════════
#  1. MUAT DAN BERSIHKAN DATA
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  PELATIHAN MODEL REGRESI LOGISTIK")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# ══════════════════════════════════════════════════════════
#  2. SPLIT DATA
# ══════════════════════════════════════════════════════════
numeric_features     = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ══════════════════════════════════════════════════════════
#  3. PIPELINE: Preprocessing → SMOTE → Regresi Logistik
# ══════════════════════════════════════════════════════════
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# imblearn Pipeline mendukung SMOTE di tengah pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("smote",        SMOTE(random_state=42)),
    ("classifier",   LogisticRegression(C=10, max_iter=2000, random_state=42))
])

print("\n▸ Melatih model (LogisticRegression + SMOTE) ...")
model.fit(X_train, y_train)
print("✔  Pelatihan selesai!")

# ══════════════════════════════════════════════════════════
#  4. PREDIKSI
# ══════════════════════════════════════════════════════════
train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)
test_prob  = model.predict_proba(X_test)[:, 1]

# ══════════════════════════════════════════════════════════
#  5. EVALUASI MODEL
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  EVALUASI MODEL")
print("=" * 60)

train_acc = accuracy_score(y_train, train_pred)
test_acc  = accuracy_score(y_test,  test_pred)
prec      = precision_score(y_test, test_pred)
rec       = recall_score(y_test, test_pred)
f1        = f1_score(y_test, test_pred)
auc       = roc_auc_score(y_test, test_prob)

print(f"\n  Akurasi Data Latih : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Akurasi Data Uji   : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"\n  Precision (Churn)  : {prec:.4f}")
print(f"  Recall    (Churn)  : {rec:.4f}   ← Metrik paling penting!")
print(f"  F1-Score  (Churn)  : {f1:.4f}")
print(f"  ROC-AUC            : {auc:.4f}")

print("\n  Classification Report:")
print(classification_report(y_test, test_pred,
                             target_names=["Tidak Churn", "Churn"]))

cm = confusion_matrix(y_test, test_pred)
tn, fp, fn, tp = cm.ravel()
print("\n  Confusion Matrix:")
print(f"  ┌──────────────────────────────────────────┐")
print(f"  │  True Negative  (TN): {tn:>5}              │")
print(f"  │  False Positive (FP): {fp:>5}  ← Risiko rendah │")
print(f"  │  False Negative (FN): {fn:>5}  ← PALING MERUGIKAN │")
print(f"  │  True Positive  (TP): {tp:>5}              │")
print(f"  └──────────────────────────────────────────┘")

# ══════════════════════════════════════════════════════════
#  INTERPRETASI BISNIS
# ══════════════════════════════════════════════════════════
print("""
  ══ INTERPRETASI BISNIS ══════════════════════════════════

  ▸ FALSE POSITIVE (FP): Model memprediksi pelanggan AKAN
    churn, padahal sebenarnya TIDAK. Risiko: perusahaan
    mengeluarkan biaya retensi yang tidak perlu.
    → Kerugian relatif kecil.

  ▸ FALSE NEGATIVE (FN): Model memprediksi pelanggan TIDAK
    akan churn, padahal sebenarnya AKAN churn. Risiko:
    pelanggan pergi tanpa ada tindakan pencegahan.
    → PALING MERUGIKAN — kehilangan pelanggan = kehilangan
      pendapatan jangka panjang.

  ▸ Oleh karena itu, RECALL untuk kelas Churn adalah metrik
    paling kritis. Nilai Recall tinggi berarti model berhasil
    menangkap sebanyak mungkin pelanggan yang benar-benar
    akan churn.

  ▸ AUC = {:.4f} → Model secara keseluruhan memiliki
    kemampuan diskriminasi yang {}.
  ═════════════════════════════════════════════════════════
""".format(auc, "baik (>0.8)" if auc > 0.8 else "sedang"))

# ══════════════════════════════════════════════════════════
#  PLOT 7 — Confusion Matrix Heatmap
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

annot_labels = np.array([
    [f"True Negative\n{tn}", f"False Positive\n{fp}"],
    [f"False Negative\n{fn}", f"True Positive\n{tp}"]
])

sns.heatmap(
    cm, annot=annot_labels, fmt="", cmap="Blues",
    linewidths=2, linecolor="white",
    xticklabels=["Prediksi: Tidak Churn", "Prediksi: Churn"],
    yticklabels=["Aktual: Tidak Churn", "Aktual: Churn"],
    ax=ax, annot_kws={"size": 11, "weight": "bold"}
)
ax.set_title("Confusion Matrix — Logistic Regression + SMOTE",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Kelas Aktual", fontsize=11)
ax.set_xlabel("Kelas Prediksi", fontsize=11)

# Tambahkan anotasi metrik di bawah
fig.text(0.5, -0.04,
         f"Accuracy: {test_acc:.4f}  |  Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f}",
         ha="center", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", edgecolor="#3b82f6"))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "07_confusion_matrix.png"), dpi=110, bbox_inches="tight")
plt.close()
print("\n✔  Plot 7 disimpan: 07_confusion_matrix.png")

# ══════════════════════════════════════════════════════════
#  PLOT 8 — ROC Curve
# ══════════════════════════════════════════════════════════
fpr_vals, tpr_vals, thresholds = roc_curve(y_test, test_prob)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_vals, tpr_vals, color="#3b82f6", lw=2.5,
        label=f"Logistic Regression (AUC = {auc:.4f})")
ax.fill_between(fpr_vals, tpr_vals, alpha=0.12, color="#3b82f6")
ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5,
        linestyle="--", label="Random Classifier (AUC = 0.5)")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
ax.set_title("ROC Curve — Logistic Regression + SMOTE",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.annotate(f"AUC = {auc:.4f}",
            xy=(0.6, 0.35), fontsize=14, fontweight="bold",
            color="#1d4ed8",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", edgecolor="#3b82f6"))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "08_roc_curve.png"), dpi=110, bbox_inches="tight")
plt.close()
print("✔  Plot 8 disimpan: 08_roc_curve.png")

print("\n" + "=" * 60)
print("  PELATIHAN & EVALUASI SELESAI")
print("=" * 60)
