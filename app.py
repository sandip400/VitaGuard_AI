from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import joblib
import numpy as np
from dotenv import load_dotenv
from predict import predict_from_base64

load_dotenv()

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Get Groq API Key from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Page Routes ──────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", groq_api_key=GROQ_API_KEY)

@app.route("/risk-analysis")
def risk_analysis():
    return render_template("risk_analysis.html")

@app.route("/symptom-analyzer")
def symptom_analyzer():
    return render_template("symptom_analyzer.html", groq_api_key=GROQ_API_KEY)

@app.route("/first-aid-assistant")
def first_aid_assistant():
    return render_template("first_aid_assistant.html", groq_api_key=GROQ_API_KEY)

@app.route("/mindguard-ai")
def mindguard_ai():
    return render_template("mindguard_ai.html", groq_api_key=GROQ_API_KEY)

@app.route("/skin-detect")
def skin_detect():
    return render_template("skin_detect.html", groq_api_key=GROQ_API_KEY)

# ── ML Prediction ────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[
        data["HighBP"], data["HighChol"], data["BMI"],
        data["Smoker"], data["PhysActivity"], data["Fruits"],
        data["Veggies"], data["HvyAlcoholConsump"], data["GenHlth"],
        data["MentHlth"], data["PhysHlth"], data["DiffWalk"],
        data["Sex"], data["Age"], data["Education"], data["Income"]
    ]])
    features_scaled = scaler.transform(features)
    probabilities = model.predict_proba(features_scaled)[0]
    return jsonify({"risk_probability": float(probabilities[2])})

@app.route("/skin-predict", methods=["POST"])
def skin_predict():
    data = request.json
    b64_image = data.get("image", "")

    if not b64_image:
        return jsonify({"error": "No image provided"}), 400

    try:
        label_code, readable_name, confidence = predict_from_base64(b64_image)
        return jsonify({
            "label_code": label_code,       # e.g. "mel"
            "disease": readable_name,        # e.g. "Melanoma"
            "confidence": confidence         # e.g. 91.4
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)