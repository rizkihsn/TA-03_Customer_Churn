
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model_churn.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    probabilitas = None

    if request.method == "POST":
        data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        X_input = pd.DataFrame([data])
        prediksi = model.predict(X_input)[0]
        probabilitas = model.predict_proba(X_input)[0][1] * 100

        hasil = (
            "Berisiko Churn"
            if prediksi == 1
            else "Tidak Berisiko Churn"
        )

    return render_template(
        "index.html",
        hasil=hasil,
        probabilitas=probabilitas
    )

if __name__ == "__main__":
    app.run(debug=True)
