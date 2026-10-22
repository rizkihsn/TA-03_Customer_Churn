
import pandas as pd

# Membaca dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Menghapus kolom ID karena tidak membantu prediksi
df = df.drop(columns=["customerID"])

# Mengubah TotalCharges menjadi numerik
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Memeriksa nilai kosong setelah konversi
print("Nilai kosong sebelum penanganan:")
print(df.isnull().sum())

# Menghapus baris yang memiliki nilai kosong
df = df.dropna()

# Memeriksa data duplikat
print("\nJumlah data duplikat:", df.duplicated().sum())

# Memisahkan fitur dan target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

print("\nUkuran data setelah preprocessing:", df.shape)
print("\nDistribusi target:")
print(y.value_counts())

print("\nPreprocessing selesai.")
