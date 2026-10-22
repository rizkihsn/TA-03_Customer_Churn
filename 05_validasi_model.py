
import pandas as pd

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score
)

# 1. Baca dan bersihkan dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna()

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 2. Tentukan fitur
numeric_features = [
    "SeniorCitizen", "tenure",
    "MonthlyCharges", "TotalCharges"
]
categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

# 3. Pisahkan data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Buat pipeline preprocessing dan model
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000, class_weight="balanced"
    ))
])

# 5. Uji beberapa nilai regularisasi
parameter_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="roc_auc",
    cv=cv,
    return_train_score=True,
    n_jobs=-1
)

print("Sedang melakukan validasi silang...")
search.fit(X_train, y_train)

# 6. Tampilkan hasil cross-validation
results = pd.DataFrame(search.cv_results_)

print("\n=== HASIL VALIDASI SILANG ===")
for i, row in results.iterrows():
    print(
        f"C={row['param_classifier__C']}: "
        f"Train AUC={row['mean_train_score']:.4f}, "
        f"Validation AUC={row['mean_test_score']:.4f}"
    )

print("\nNilai C terbaik:", search.best_params_["classifier__C"])
print("Rata-rata Validation AUC terbaik:",
      round(search.best_score_, 4))

# 7. Evaluasi model terbaik pada data uji
best_model = search.best_estimator_
pred = best_model.predict(X_test)
prob = best_model.predict_proba(X_test)[:, 1]

print("\n=== EVALUASI DATA UJI ===")
print("Akurasi:", round(accuracy_score(y_test, pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
print("\nClassification Report:")
print(classification_report(
    y_test, pred, target_names=["Tidak Churn", "Churn"]
))

# 8. Simpan model terbaik
import joblib
joblib.dump(best_model, "model_churn.pkl")
print("Model terbaik disimpan ke model_churn.pkl")
