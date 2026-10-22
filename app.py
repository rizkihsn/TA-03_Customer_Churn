"""
=============================================================
  TA-03 | Customer Churn Prediction
  app.py — Flask Web Application
=============================================================
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── Setup ───────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_churn.pkl")

app   = Flask(__name__)
model = joblib.load(MODEL_PATH)

# Statistik model untuk ditampilkan di UI
MODEL_STATS = {
    "accuracy" : 73.13,
    "recall"   : 77.54,
    "auc"      : 83.34,
    "f1"       : 60.54,
    "algorithm": "Logistic Regression + SMOTE",
    "dataset"  : "Telco Customer Churn (7,032 records)",
}

# ══════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/", methods=["GET", "POST"])
def index():
    hasil       = None
    probabilitas = None
    risk_level  = None
    form_data   = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        data = {
            "gender"          : form_data.get("gender", "Male"),
            "SeniorCitizen"   : int(form_data.get("SeniorCitizen", 0)),
            "Partner"         : form_data.get("Partner", "No"),
            "Dependents"      : form_data.get("Dependents", "No"),
            "tenure"          : int(form_data.get("tenure", 12)),
            "PhoneService"    : form_data.get("PhoneService", "Yes"),
            "MultipleLines"   : form_data.get("MultipleLines", "No"),
            "InternetService" : form_data.get("InternetService", "DSL"),
            "OnlineSecurity"  : form_data.get("OnlineSecurity", "No"),
            "OnlineBackup"    : form_data.get("OnlineBackup", "No"),
            "DeviceProtection": form_data.get("DeviceProtection", "No"),
            "TechSupport"     : form_data.get("TechSupport", "No"),
            "StreamingTV"     : form_data.get("StreamingTV", "No"),
            "StreamingMovies" : form_data.get("StreamingMovies", "No"),
            "Contract"        : form_data.get("Contract", "Month-to-month"),
            "PaperlessBilling": form_data.get("PaperlessBilling", "Yes"),
            "PaymentMethod"   : form_data.get("PaymentMethod", "Electronic check"),
            "MonthlyCharges"  : float(form_data.get("MonthlyCharges", 70)),
            "TotalCharges"    : float(form_data.get("TotalCharges", 840)),
        }

        X_input      = pd.DataFrame([data])
        prediksi     = model.predict(X_input)[0]
        probabilitas = round(model.predict_proba(X_input)[0][1] * 100, 2)

        if probabilitas >= 70:
            risk_level = "high"
            hasil      = "Berisiko Tinggi Churn"
        elif probabilitas >= 40:
            risk_level = "medium"
            hasil      = "Berisiko Sedang Churn"
        else:
            risk_level = "low"
            hasil      = "Tidak Berisiko Churn"

    return render_template(
        "index.html",
        hasil        = hasil,
        probabilitas = probabilitas,
        risk_level   = risk_level,
        form_data    = form_data,
        stats        = MODEL_STATS,
    )


@app.route("/eda")
def eda():
    """Halaman visualisasi EDA."""
    return render_template("eda.html", stats=MODEL_STATS)


if __name__ == "__main__":
    app.run(debug=True)
