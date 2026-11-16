
import re
import json
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, StrictStr
from scipy.sparse import hstack


# Initialize FastAPI
app = FastAPI(
    title="SMS Spam Detection API",
    description="Machine Learning Based SMS Spam Detection System",
    version="1.0.0"
)


# Define project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "sms_spam_model.joblib"
LOG_PATH = BASE_DIR / "prediction_logs.jsonl"


# Load the saved model and preprocessing components
model_bundle = joblib.load(MODEL_PATH)

tfidf = model_bundle["tfidf"]
scaler = model_bundle["scaler"]
classifier = model_bundle["classifier"]


# Define input validation schema
class SMSRequest(BaseModel):
    message: StrictStr = Field(
        ...,
        min_length=1,
        max_length=500,
        description="SMS message to classify"
    )


# Extract numerical features from SMS
def extract_features(message):

    return {
        "message_length": len(message),
        "word_count": len(message.split()),
        "digit_count": sum(char.isdigit() for char in message),
        "uppercase_count": sum(char.isupper() for char in message),
        "url_count": len(
            re.findall(r"http[s]?://|www\.", message)
        )
    }


# Generate safety tips
def generate_tips(prediction):

    if prediction == "Spam":
        return [
            "Do not click suspicious links.",
            "Never share OTPs or banking details.",
            "Block and report suspicious senders."
        ]

    return [
        "Verify unexpected requests for personal information.",
        "Avoid clicking links from unknown senders."
    ]


# Health check endpoint
@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "loaded"
    }


# Prediction endpoint
@app.post("/predict")
def predict_sms(request: SMSRequest):

    message = request.message.strip()

    if not message:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Message cannot contain only spaces."
        )

    # Extract numerical features
    features = pd.DataFrame(
        [extract_features(message)],
        columns=[
            "message_length",
            "word_count",
            "digit_count",
            "uppercase_count",
            "url_count"
        ]
    )

    # Apply preprocessing
    text_features = tfidf.transform([message])
    numeric_features = scaler.transform(features)

    # Combine features
    final_features = hstack([
        text_features,
        numeric_features
    ])

    # Predict
    result = classifier.predict(final_features)[0]

    prediction = "Spam" if result == 1 else "Ham"

    # Generate safety tips
    tips = generate_tips(prediction)

    # Log prediction for monitoring
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "prediction": prediction
    }

    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

    return {
        "message": message,
        "prediction": prediction,
        "tips": tips
    }


# Home endpoint
@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")
