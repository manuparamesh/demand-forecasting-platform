from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_6: float
    lag_12: float
    rolling_mean_3: float
    rolling_std_3: float
    rolling_mean_6: float
    rolling_std_6: float
    month: int = Field(..., ge=1, le=12)
    quarter: int = Field(..., ge=1, le=4)
    year: int
    promo_flag: int = Field(..., ge=0, le=1)


class ForecastResponse(BaseModel):
    predicted_sales: float
    model_name: str
    model_version: str