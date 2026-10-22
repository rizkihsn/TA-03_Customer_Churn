"""
=============================================================
  TA-03 | Customer Churn Prediction
  02 - Pra-Pemrosesan Data (Preprocessing + SMOTE)
=============================================================
Script ini melakukan:
  - Membersihkan data (missing values, konversi tipe)
  - Encoding variabel kategorikal
  - Penanganan ketidakseimbangan kelas dengan SMOTE
  - Visualisasi distribusi kelas sebelum & sesudah SMOTE
=============================================================
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Setup path ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
IMG_DIR   = os.path.join(BASE_DIR, "..", "static", "img", "eda")
os.makedirs(IMG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
#  1. MEMUAT DAN MEMBERSIHKAN DATA
# ══════════════════════════════════════════════════════════
print("=" * 55)
print("  PRA-PEMROSESAN DATA")
print("=" * 55)

df = pd.read_csv(DATA_PATH)

# Hapus kolom ID (tidak informatif untuk prediksi)
df = df.drop(columns=["customerID"])

# Konversi TotalCharges ke numerik (ada nilai string kosong)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print(f"\n▸ Nilai kosong sebelum penanganan:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Hapus baris dengan nilai kosong (hanya 11 baris — aman dihapus)
df = df.dropna()
print(f"\n▸ Ukuran data setelah hapus missing values: {df.shape}")

# Periksa duplikat
dup = df.duplicated().sum()
print(f"▸ Jumlah data duplikat: {dup}")

# ══════════════════════════════════════════════════════════
#  2. PISAHKAN FITUR DAN TARGET
# ══════════════════════════════════════════════════════════
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

print(f"\n▸ Distribusi target sebelum SMOTE:")
print(y.value_counts())
print(f"   Rasio Churn: {y.mean():.2%}")

# ══════════════════════════════════════════════════════════
#  3. TENTUKAN KOLOM NUMERIK DAN KATEGORIKAL
# ══════════════════════════════════════════════════════════
numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [col for col in X.columns if col not in numeric_features]

print(f"\n▸ Fitur Numerik  ({len(numeric_features)}) : {numeric_features}")
print(f"▸ Fitur Kategorikal ({len(categorical_features)}) : {categorical_features}")

# ══════════════════════════════════════════════════════════
#  4. SPLIT DATA
# ══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n▸ Ukuran data latih : {X_train.shape}")
print(f"▸ Ukuran data uji   : {X_test.shape}")

# ══════════════════════════════════════════════════════════
#  5. PREPROCESSING (StandardScaler + OneHotEncoder)
# ══════════════════════════════════════════════════════════
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

print(f"\n▸ Ukuran setelah encoding: {X_train_enc.shape}")

# ══════════════════════════════════════════════════════════
#  6. SMOTE — PENYEIMBANGAN KELAS PADA DATA LATIH
# ══════════════════════════════════════════════════════════
print("\n▸ Menerapkan SMOTE pada data latih ...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)

print(f"▸ Distribusi target SEBELUM SMOTE:")
import pandas as pd
# Menggunakan series langsung
before = pd.Series(y_train).value_counts().sort_index()
after  = pd.Series(y_train_sm).value_counts().sort_index()
print(f"   0 (Tidak Churn) : {before[0]}")
print(f"   1 (Churn)       : {before[1]}")
print(f"\n▸ Distribusi target SESUDAH SMOTE:")
print(f"   0 (Tidak Churn) : {after[0]}")
print(f"   1 (Churn)       : {after[1]}")

# ══════════════════════════════════════════════════════════
#  PLOT 6 — Distribusi Kelas Sebelum vs Sesudah SMOTE
# ══════════════════════════════════════════════════════════
COLOR_OK  = "#22c55e"
COLOR_BAD = "#ef4444"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribusi Kelas: Sebelum vs Sesudah SMOTE", fontsize=14, fontweight="bold")

for ax, counts, title in zip(
    axes,
    [before, after],
    ["Sebelum SMOTE (Data Latih)", "Sesudah SMOTE (Data Latih)"]
):
    bars = ax.bar(["Tidak Churn (0)", "Churn (1)"],
                  [counts[0], counts[1]],
                  color=[COLOR_OK, COLOR_BAD],
                  edgecolor="white", width=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Jumlah Sampel")
    for bar, val in zip(bars, [counts[0], counts[1]]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 30,
                f"{val:,}", ha="center", fontweight="bold", fontsize=12)
    ax.set_ylim(0, max(counts[0], counts[1]) * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "06_smote_balance.png"), dpi=110, bbox_inches="tight")
plt.close()
print("\n✔  Plot 6 disimpan: 06_smote_balance.png")

print("\n" + "=" * 55)
print("  PREPROCESSING SELESAI")
print("=" * 55)
