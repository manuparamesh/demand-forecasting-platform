from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    payload = {
        "lag_1": 120.0,
        "lag_2": 118.0,
        "lag_3": 122.0,
        "lag_6": 115.0,
        "lag_12": 110.0,
        "rolling_mean_3": 120.0,
        "rolling_std_3": 2.0,
        "rolling_mean_6": 118.5,
        "rolling_std_6": 3.1,
        "month": 7,
        "quarter": 3,
        "year": 2025,
        "promo_flag": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_sales" in body
    assert "model_name" in body
    assert "model_version" in body