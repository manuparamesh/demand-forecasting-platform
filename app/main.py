from fastapi import FastAPI
from app.schemas import ForecastRequest, ForecastResponse
from app.predictor import ForecastService

app = FastAPI(
    title="Demand Forecasting Platform",
    description="Production-oriented ML inference API for demand forecasting",
    version="1.0.0",
)

forecast_service = ForecastService()


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    prediction = forecast_service.predict(
        lag_1=request.lag_1,
        lag_2=request.lag_2,
        lag_3=request.lag_3,
        rolling_mean_3=request.rolling_mean_3,
        month=request.month,
        promo_flag=request.promo_flag,
    )
    return ForecastResponse(predicted_sales=prediction)