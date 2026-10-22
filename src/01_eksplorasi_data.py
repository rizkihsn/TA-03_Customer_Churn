"""
=============================================================
  TA-03 | Customer Churn Prediction
  01 - Eksplorasi Data (EDA) + Visualisasi
=============================================================
Script ini melakukan:
  - Memuat dan memeriksa dataset Telco Customer Churn
  - Analisis distribusi fitur & target
  - Visualisasi hubungan antar fitur dengan target Churn
  - Menyimpan semua plot ke static/img/eda/
=============================================================
"""

import sys
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ── Setup path ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
IMG_DIR    = os.path.join(BASE_DIR, "..", "static", "img", "eda")
os.makedirs(IMG_DIR, exist_ok=True)

# ── Gaya visual ────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor"  : "#f8fafc",
    "axes.grid"       : True,
    "grid.alpha"      : 0.4,
    "font.family"     : "DejaVu Sans",
})

COLOR_OK  = "#22c55e"   # hijau  → tidak churn
COLOR_BAD = "#ef4444"   # merah  → churn

# ── Muat dataset ───────────────────────────────────────────
print("=" * 55)
print("  EKSPLORASI DATA - TELCO CUSTOMER CHURN")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print(f"\n▸ Ukuran dataset : {df.shape[0]:,} baris × {df.shape[1]} kolom")
print(f"▸ Jumlah nilai kosong :")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n▸ Distribusi Target Churn :")
churn_counts = df["Churn"].value_counts()
print(churn_counts)
churn_pct = df["Churn"].value_counts(normalize=True) * 100
print(f"   Tidak Churn : {churn_pct['No']:.1f}%")
print(f"   Churn       : {churn_pct['Yes']:.1f}%")

print("\n▸ Statistik Deskriptif :")
print(df.describe().round(2))

# ══════════════════════════════════════════════════════════
#  PLOT 1 — Distribusi Target (Bar + Pie)
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Distribusi Kelas Target (Churn)", fontsize=15, fontweight="bold", y=1.01)

labels  = ["Tidak Churn", "Churn"]
values  = [churn_counts["No"], churn_counts["Yes"]]
colors  = [COLOR_OK, COLOR_BAD]

# Bar chart
bars = axes[0].bar(labels, values, color=colors, edgecolor="white", width=0.5, linewidth=1.5)
axes[0].set_title("Jumlah Pelanggan per Kelas", fontweight="bold")
axes[0].set_ylabel("Jumlah Pelanggan")
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 50,
                 f"{val:,}", ha="center", fontweight="bold", fontsize=12)
axes[0].set_ylim(0, max(values) * 1.15)

# Pie chart
axes[1].pie(values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops={"fontsize": 12})
axes[1].set_title("Persentase Distribusi Kelas", fontweight="bold")

# Annotation ketidakseimbangan kelas
fig.text(0.5, -0.03,
         "⚠  Ketidakseimbangan Kelas (Class Imbalance): Kelas Churn hanya ~26% — perlu teknik penyeimbangan (SMOTE).",
         ha="center", fontsize=10, color="#b45309",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7", edgecolor="#f59e0b"))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "01_distribusi_target.png"), dpi=110, bbox_inches="tight")
plt.close()
print("\n✔  Plot 1 disimpan: 01_distribusi_target.png")

# ══════════════════════════════════════════════════════════
#  PLOT 2 — Distribusi Fitur Numerik per Kelas Churn
# ══════════════════════════════════════════════════════════
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
labels_num   = ["Lama Berlangganan (bulan)", "Tagihan Bulanan ($)", "Total Tagihan ($)"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Distribusi Fitur Numerik per Kelas Churn", fontsize=14, fontweight="bold")

for ax, col, label in zip(axes, numeric_cols, labels_num):
    for churn_val, color, lbl in [("No", COLOR_OK, "Tidak Churn"), ("Yes", COLOR_BAD, "Churn")]:
        subset = df[df["Churn"] == churn_val][col].dropna()
        ax.hist(subset, bins=30, alpha=0.65, color=color, label=lbl, edgecolor="none")
    ax.set_title(label, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Frekuensi")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "02_distribusi_numerik.png"), dpi=110, bbox_inches="tight")
plt.close()
print("✔  Plot 2 disimpan: 02_distribusi_numerik.png")

# ══════════════════════════════════════════════════════════
#  PLOT 3 — Fitur Kategorikal vs Churn
# ══════════════════════════════════════════════════════════
cat_cols  = ["Contract", "InternetService", "PaymentMethod", "gender"]
cat_label = ["Jenis Kontrak", "Layanan Internet", "Metode Pembayaran", "Jenis Kelamin"]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Distribusi Fitur Kategorikal vs Churn", fontsize=14, fontweight="bold")
axes = axes.flatten()

for ax, col, label in zip(axes, cat_cols, cat_label):
    cross = pd.crosstab(df[col], df["Churn"])
    cross.columns = ["Tidak Churn", "Churn"]
    cross.plot(kind="bar", ax=ax, color=[COLOR_OK, COLOR_BAD],
               edgecolor="white", alpha=0.85, width=0.65)
    ax.set_title(label, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right")
    ax.set_ylabel("Jumlah Pelanggan")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "03_distribusi_kategorikal.png"), dpi=110, bbox_inches="tight")
plt.close()
print("✔  Plot 3 disimpan: 03_distribusi_kategorikal.png")

# ══════════════════════════════════════════════════════════
#  PLOT 4 — Heatmap Korelasi Fitur Numerik
# ══════════════════════════════════════════════════════════
df_corr = df.copy()
df_corr["Churn"]       = df_corr["Churn"].map({"No": 0, "Yes": 1})
df_corr["SeniorCitizen"] = df_corr["SeniorCitizen"].astype(float)

numeric_for_corr = df_corr[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
corr_matrix = numeric_for_corr.corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = None
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, linewidths=0.5,
    square=True, ax=ax,
    annot_kws={"size": 12, "weight": "bold"}
)
ax.set_title("Heatmap Korelasi Fitur Numerik dengan Target Churn",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "04_heatmap_korelasi.png"), dpi=110, bbox_inches="tight")
plt.close()
print("✔  Plot 4 disimpan: 04_heatmap_korelasi.png")

# ══════════════════════════════════════════════════════════
#  PLOT 5 — Boxplot: MonthlyCharges & tenure vs Churn
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Boxplot Fitur Numerik vs Churn", fontsize=14, fontweight="bold")

for ax, col, label in zip(axes,
                           ["MonthlyCharges", "tenure"],
                           ["Tagihan Bulanan ($)", "Lama Berlangganan (bulan)"]):
    data_no  = df[df["Churn"] == "No"][col].dropna()
    data_yes = df[df["Churn"] == "Yes"][col].dropna()
    bp = ax.boxplot([data_no, data_yes],
                    tick_labels=["Tidak Churn", "Churn"],
                    patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(COLOR_OK)
    bp["boxes"][1].set_facecolor(COLOR_BAD)
    ax.set_title(label, fontweight="bold")
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "05_boxplot.png"), dpi=110, bbox_inches="tight")
plt.close()
print("✔  Plot 5 disimpan: 05_boxplot.png")

print("\n" + "=" * 55)
print("  EDA SELESAI — Semua plot disimpan di static/img/eda/")
print("=" * 55)
