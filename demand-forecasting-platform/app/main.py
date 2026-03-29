from fastapi import FastAPI
from app.schemas import ForecastRequest, ForecastResponse
from app.predictor import ForecastService

app = FastAPI(
    title="Demand Forecasting Platform",
    description="Advanced production-oriented ML inference API for demand forecasting",
    version="2.0.0",
)

forecast_service = ForecastService()


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    result = forecast_service.predict(request.model_dump())
    return ForecastResponse(**result)