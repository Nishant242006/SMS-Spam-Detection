
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Make the project folder available for importing app.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app, extract_features

client = TestClient(app)


# Test 1: Health endpoint
def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Test 2: Frontend page
def test_home_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "SMS Spam Detection" in response.text


# Test 3: SMS prediction
def test_sms_prediction():
    response = client.post(
        "/predict",
        json={"message": "Congratulations! You have won a free prize. Claim now!"}
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in ["Spam", "Ham"]
    assert isinstance(response.json()["tips"], list)


# Test 4: Feature extraction
def test_feature_extraction():
    features = extract_features("Hello 123 www.example.com")

    assert features["message_length"] == 25
    assert features["word_count"] == 3
    assert features["digit_count"] == 3
    assert features["uppercase_count"] == 1
    assert features["url_count"] == 1


# Test 5: Invalid input boundary
def test_message_too_long():
    response = client.post(
        "/predict",
        json={"message": "A" * 501}
    )

    assert response.status_code == 422
