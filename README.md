# ⚡ ChurnSense — Prediksi Customer Churn

> **TA-03 · Tugas Akhir Machine Learning · Regresi Logistik**

Aplikasi web prediksi risiko pelanggan berhenti berlangganan (*customer churn*) menggunakan **Logistic Regression** + **SMOTE**, dibangun dengan Flask dan di-*deploy* ke Vercel.

---

## 🎯 Deskripsi Proyek

Proyek ini membangun sistem prediksi churn pelanggan telekomunikasi. Seorang analis dapat memasukkan data pelanggan melalui form web dan mendapatkan prediksi instan beserta tingkat risiko:

- ✅ **Tidak Berisiko Churn** (probabilitas < 40%)
- ⚠️ **Berisiko Sedang** (probabilitas 40–70%)
- 🚨 **Berisiko Tinggi** (probabilitas > 70%)

---

## 📊 Dataset

| Properti | Detail |
|----------|--------|
| **Nama** | Telco Customer Churn |
| **Sumber** | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Ukuran** | 7,043 baris × 21 kolom |
| **Target** | `Churn` (Yes/No) |
| **Fitur** | Campuran numerik (tenure, MonthlyCharges, TotalCharges) dan kategorikal (Contract, InternetService, PaymentMethod, dll.) |

---

## 🤖 Model & Hasil Evaluasi

| Metrik | Nilai |
|--------|-------|
| **Algoritma** | Logistic Regression + SMOTE |
| **Akurasi** | 73.13% |
| **Recall (Churn)** | **77.54%** ← Metrik utama |
| **F1-Score** | 60.54% |
| **ROC-AUC** | **83.34%** |

> **Recall** diprioritaskan karena False Negative (pelanggan churn yang tidak terdeteksi) jauh lebih merugikan dibanding False Positive.

---

## 🗂️ Struktur Proyek

```
TA-03_Customer_Churn/
├── app.py                    # Flask web application
├── vercel.json               # Konfigurasi deployment Vercel
├── requirements.txt          # Dependensi Python
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── model_churn.pkl       # Model terlatih (Pipeline SMOTE + LogReg)
│   └── feature_names.pkl     # Daftar nama fitur
│
├── src/
│   ├── 01_eksplorasi_data.py     # EDA + 5 visualisasi
│   ├── 02_preprocessing.py       # Preprocessing + SMOTE demo
│   ├── 03_pelatihan_model.py     # Training + Confusion Matrix + ROC
│   ├── 04_simpan_model.py        # Simpan model final
│   └── 05_validasi_model.py      # Cross-validation + GridSearchCV
│
├── templates/
│   ├── index.html            # Halaman prediksi (UI premium dark mode)
│   └── eda.html              # Halaman visualisasi EDA
│
├── static/
│   └── img/eda/              # 9 plot visualisasi (PNG)
│       ├── 01_distribusi_target.png
│       ├── 02_distribusi_numerik.png
│       ├── 03_distribusi_kategorikal.png
│       ├── 04_heatmap_korelasi.png
│       ├── 05_boxplot.png
│       ├── 06_smote_balance.png
│       ├── 07_confusion_matrix.png
│       ├── 08_roc_curve.png
│       └── 09_cv_scores.png
│
└── notebooks/
    └── analisis_lengkap.ipynb    # Notebook Jupyter lengkap
```

---

## 🚀 Cara Menjalankan Lokal

### 1. Clone repositori
```bash
git clone <repo-url>
cd TA-03_Customer_Churn
```

### 2. Install dependensi
```bash
py -m pip install -r requirements.txt
```

### 3. Generate visualisasi & model (jika belum ada)
```bash
py src/01_eksplorasi_data.py
py src/03_pelatihan_model.py
py src/05_validasi_model.py
```

### 4. Jalankan aplikasi
```bash
py app.py
```

### 5. Buka browser
```
http://127.0.0.1:5000/
```

---

## ☁️ Deployment ke Vercel

1. Push semua file ke GitHub (termasuk `models/` dan `static/img/eda/`)
2. Buka [vercel.com](https://vercel.com) → **New Project** → Import repo
3. Vercel otomatis mendeteksi `vercel.json` dan mendeploy Flask app
4. Klik **Deploy** → URL live tersedia dalam ~2 menit

---

## 🛠️ Teknologi

| Kategori | Library |
|----------|---------|
| **ML & Data** | Scikit-Learn, Pandas, NumPy |
| **SMOTE** | imbalanced-learn |
| **Visualisasi** | Matplotlib, Seaborn |
| **Web App** | Flask |
| **Model Storage** | Joblib |
| **Deployment** | Vercel |

---

## 📈 Langkah Analisis

1. **EDA** — distribusi fitur, korelasi, visualisasi kelas
2. **Preprocessing** — encoding kategorikal, scaling numerik, SMOTE
3. **Training** — Logistic Regression pipeline
4. **Evaluasi** — Confusion Matrix, Precision, Recall, F1, AUC
5. **Validasi** — GridSearchCV 5-fold cross-validation
6. **Deployment** — Flask + Vercel

---

*Dibuat untuk TA-03 · Program Studi Teknik Informatika*
