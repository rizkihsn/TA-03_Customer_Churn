
import pandas as pd

# Membaca dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Menampilkan informasi dataset
print("=== 5 DATA PERTAMA ===")
print(df.head())

print("\n=== UKURAN DATASET ===")
print(df.shape)

print("\n=== INFORMASI KOLOM ===")
df.info()

print("\n=== NILAI KOSONG ===")
print(df.isnull().sum())

print("\n=== DISTRIBUSI TARGET CHURN ===")
print(df["Churn"].value_counts())

print("\n=== STATISTIK DESKRIPTIF ===")
print(df.describe())
