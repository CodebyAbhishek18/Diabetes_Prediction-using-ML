
from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import pandas as pd
import os

# Load trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        input_features = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["blood_pressure"]),
            float(request.form["skin_thickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["pedigree"]),
            float(request.form["age"]),
        ]

        # Transform input using the scaler
        input_features = np.array(input_features).reshape(1, -1)
        input_features = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Interpret the result
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        # Save result to a CSV file
        data = {
            "Pregnancies": [request.form["pregnancies"]],
            "Glucose": [request.form["glucose"]],
            "Blood Pressure": [request.form["blood_pressure"]],
            "Skin Thickness": [request.form["skin_thickness"]],
            "Insulin": [request.form["insulin"]],
            "BMI": [request.form["bmi"]],
            "Diabetes Pedigree": [request.form["pedigree"]],
            "Age": [request.form["age"]],
            "Prediction": [result]
        }
        df = pd.DataFrame(data)
        df.to_csv("result.csv", index=False)

        return render_template("form.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/download")
def download():
    return send_file("result.csv", as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use PORT from environment
    app.run(host="0.0.0.0", port=port, debug=True)
