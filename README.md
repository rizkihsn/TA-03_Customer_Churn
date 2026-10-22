# ⚡ ChurnSense — Prediksi Customer Churn

> **TA-03 · Tugas Akhir Machine Learning · Regresi Logistik**

Aplikasi web prediksi risiko pelanggan berhenti berlangganan (*customer churn*) menggunakan **Logistic Regression** + **SMOTE**, dibangun dengan Flask, bertema antarmuka **Dark Red**, dan siap di-*deploy* ke **Vercel**.

---

## 🎯 Deskripsi Proyek

Proyek ini membangun sistem prediksi churn pelanggan telekomunikasi berbasis klasifikasi biner. Pengguna dapat memasukkan parameter data pelanggan melalui form web interaktif dan memperoleh hasil evaluasi instan beserta tingkat risiko:

- ✅ **Tidak Berisiko Churn** (probabilitas < 40%) — Pelanggan cenderung loyal.
- ⚠️ **Berisiko Sedang** (probabilitas 40–70%) — Perlu perhatian khusus dan penawaran retensi.
- 🚨 **Berisiko Tinggi** (probabilitas > 70%) — Sangat rentan berhenti berlangganan; butuh intervensi segera.

---

## 📊 Dataset

| Properti | Detail |
|----------|--------|
| **Nama** | Telco Customer Churn |
| **Sumber** | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Ukuran** | 7,043 baris × 21 kolom (7,032 baris setelah penanganan missing values) |
| **Target** | `Churn` (Yes = 1, No = 0) |
| **Fitur Numerik** | `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` |
| **Fitur Kategorikal** | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |

---

## 🤖 Model & Hasil Evaluasi

Model dibangun menggunakan pipeline terintegrasi: **One-Hot Encoding** + **StandardScaler** + penanganan class imbalance dengan **SMOTE** + **Logistic Regression (C=10)**.

| Metrik Evaluasi | Nilai (Data Uji) | Keterangan |
|-----------------|------------------|------------|
| **Akurasi Data Uji** | **73.28%** | Performa klasifikasi keseluruhan pada data uji (1.407 sampel) |
| **Recall (Churn)** | **78.07%** | ⭐ **Metrik paling kritis** (berhasil mendeteksi 292 dari 374 pelanggan churn) |
| **Precision (Churn)**| **49.83%** | Proporsi prediksi churn yang tepat |
| **F1-Score (Churn)** | **60.83%** | Harmonic mean precision & recall |
| **ROC-AUC** | **83.29%** | Kemampuan diskriminasi model sangat baik (> 0.8) |
| **Validation AUC (5-Fold)** | **84.55%** | Skor rata-rata Stratified K-Fold CV (C optimal = 10) |

> 💡 **Alasan Prioritas Recall:**  
> Pada domain retensi pelanggan telekomunikasi, **False Negative (FN)** — pelanggan yang sebenarnya akan berhenti berlangganan tetapi terlewat dan diprediksi aman — merupakan kerugian terbesar karena perusahaan kehilangan pendapatan jangka panjang (*loss of customer lifetime value*). Sebaliknya, **False Positive (FP)** hanya menimbulkan biaya retensi kecil (misal: pemberian voucher diskon kepada pelanggan yang sebenarnya loyal).

---

### ⚖️ Aturan Segmentasi Risiko Bisnis vs Ambang Batas Klasifikasi Biner

Penting untuk dicatat bahwa model **Logistic Regression** secara standar matematis menggunakan ambang batas klasifikasi biner **50% (0.5)**:
- Probabilitas $\ge 50\%$ diklasifikasikan sebagai **Churn (1)**.
- Probabilitas $< 50\%$ diklasifikasikan sebagai **Tidak Churn (0)**.

Namun, untuk kebutuhan operasional manajemen bisnis di dunia nyata, aplikasi ini menerapkan **aturan segmentasi risiko manajerial (*business decision threshold*)**:
1. **Risiko Rendah ($< 40\%$)**: Pelanggan aman dan loyal. Diberikan program apresiasi standar.
2. **Risiko Sedang ($40\% - 70\%$)**: Pelanggan mulai menunjukkan tanda ketidakpuasan layanan. Meskipun probabilitasnya belum tentu melampaui 50%, tindakan pencegahan proaktif (seperti customer care check-in atau opsi paket fleksibel) perlu dimulai lebih awal.
3. **Risiko Tinggi ($\ge 70\%$)**: Pelanggan hampir pasti akan berhenti berlangganan dalam waktu dekat. Memerlukan intervensi agresif dari tim retensi khusus.

*Pengelompokan 40% dan 70% ini merupakan aturan bisnis aplikasi untuk strategi preventif, bukan batas yang otomatis diputuskan oleh algoritma Logistic Regression.*

---

## 🗂️ Struktur Proyek

```text
TA-03_Customer_Churn/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset Telco Customer Churn
│
├── models/
│   ├── feature_names.pkl                     # Daftar nama fitur setelah encoding
│   └── model_churn.pkl                       # Model Pipeline final (SMOTE + Logistic Regression)
│
├── src/
│   ├── 01_eksplorasi_data.py                 # EDA dan 5 visualisasi eksplorasi awal
│   ├── 02_preprocessing.py                   # Preprocessing data & simulasi SMOTE
│   ├── 03_pelatihan_model.py                 # Pelatihan, matriks evaluasi & ROC curve
│   ├── 04_simpan_model.py                    # Pelatihan full data & serialisasi pipeline
│   └── 05_validasi_model.py                  # 5-Fold Stratified CV & Hyperparameter Tuning
│
├── static/
│   └── img/eda/                              # 9 aset visualisasi beresolusi tinggi (PNG)
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
├── templates/
│   ├── index.html                            # Form input & kartu hasil prediksi (Tema Dark Red)
│   └── eda.html                              # Dashboard visualisasi EDA 4 tab (Tema Dark Red)
│
├── app.py                                    # Aplikasi web Flask server
├── requirements.txt                          # Daftar dependensi Python lengkap
├── vercel.json                               # Konfigurasi deploy serverless ke Vercel
├── .gitignore                                # Filter cache python, .venv, dan konfigurasi IDE
└── README.md                                 # Dokumentasi proyek
```

---

## 🎨 Tema UI/UX: Dark Red Edition

Aplikasi dirancang dengan tampilan modern dan profesional:
- **Warna Dasar:** Deep wine/black base (`#0d0608`) dipadukan dengan card burgundy (`#1a0b10`).
- **Aksen:** Gradien merah menyala (`#f87171` → `#dc2626` → `#991b1b`).
- **Elemen Interaktif:** Probability progress ring SVG animasi, input focus ring merah, hover micro-animations, dan tab navigasi dinamis pada visualisasi EDA.

---

## 🚀 Cara Menjalankan Secara Lokal

### 1. Clone repositori
```bash
git clone https://github.com/rizkihsn/TA-03_Customer_Churn.git
cd TA-03_Customer_Churn
```

### 2. Buat virtual environment & install dependensi
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Jalankan pipeline (opsional jika ingin melatih ulang)
```bash
python src/01_eksplorasi_data.py
python src/02_preprocessing.py
python src/03_pelatihan_model.py
python src/04_simpan_model.py
python src/05_validasi_model.py
```

### 4. Jalankan server web Flask
```bash
python app.py
```

### 5. Akses melalui browser
- Form Prediksi: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- Dashboard Visualisasi EDA: [http://127.0.0.1:5000/eda](http://127.0.0.1:5000/eda)

### 📦 Tips Pembuatan Arsip ZIP untuk Pengumpulan Tugas
Jika mengumpulkan proyek dalam bentuk file ZIP ke dosen atau platform e-learning:
- **Jangan menyertakan folder `.venv/`** karena berukuran ratusan MB dan berisi library lokal yang spesifik untuk mesin Anda.
- Gunakan perintah `git archive` di terminal agar ZIP yang dihasilkan bersih dan hanya berisi file kode proyek:
  ```powershell
  git archive -o TA-03_Customer_Churn.zip HEAD
  ```
  *(Perintah ini menghasilkan file zip ringan hanya sekitar ~1–2 MB tanpa folder `.venv`)*

---

## ☁️ Deployment ke Vercel

Proyek ini telah dilengkapi file [vercel.json](file:///d:/UNIBBA%20%28%20TI%20%29/Tugas%20Perkulliahan/Tugas%20Perkuliahan%20Semester%205/P.Machine%20Learning/Modul%203/TA-03_Customer_Churn/vercel.json):
1. Pastikan repository sudah di-push ke GitHub.
2. Buka dashboard [Vercel](https://vercel.com) dan buat **New Project**.
3. Import repository `TA-03_Customer_Churn`.
4. Vercel akan otomatis mengenali WSGI Flask melalui `app.py`.
5. Klik tombol **Deploy**; aplikasi akan tayang dalam hitungan menit.

---

## 🛠️ Teknologi & Pustaka

| Komponen | Teknologi |
|----------|-----------|
| **Bahasa Pemrograman** | Python 3.12+ |
| **Machine Learning** | `scikit-learn`, `imbalanced-learn` (SMOTE) |
| **Manipulasi Data** | `pandas`, `numpy` |
| **Visualisasi Data** | `matplotlib`, `seaborn` |
| **Web Framework** | `Flask`, `Jinja2` |
| **Styling & UI** | Vanilla CSS (Dark Red Glassmorphism & Micro-animations) |
| **Penyimpanan Model** | `joblib` |
| **Cloud Hosting** | `Vercel` |

---

*Dibuat untuk memenuhi Tugas Akhir 03 (TA-03) · Praktikum Machine Learning · Teknik Informatika*
