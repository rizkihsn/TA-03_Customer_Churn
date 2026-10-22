
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

# 1. Membaca dan membersihkan dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna()

# 2. Memisahkan fitur dan target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 3. Menentukan kolom numerik dan kategorikal
numeric_features = [
    "SeniorCitizen", "tenure",
    "MonthlyCharges", "TotalCharges"
]
categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

# 4. Membagi data secara stratifikasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

# 6. Membuat pipeline dan melatih model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000, class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# 7. Prediksi data latih dan data uji
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]

print("Akurasi data latih:",
      round(accuracy_score(y_train, train_pred), 4))
print("Akurasi data uji:",
      round(accuracy_score(y_test, test_pred), 4))

print("\nClassification Report:")
print(classification_report(
    y_test, test_pred, target_names=["Tidak Churn", "Churn"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("ROC-AUC:",
      round(roc_auc_score(y_test, test_prob), 4))
