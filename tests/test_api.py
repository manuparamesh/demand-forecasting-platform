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
        "rolling_mean_3": 120.0,
        "month": 7,
        "promo_flag": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_sales" in response.json()