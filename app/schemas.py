from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    lag_1: float = Field(..., description="Sales value from previous period")
    lag_2: float = Field(..., description="Sales value from two periods ago")
    lag_3: float = Field(..., description="Sales value from three periods ago")
    rolling_mean_3: float = Field(..., description="Rolling mean over last 3 periods")
    month: int = Field(..., ge=1, le=12, description="Month number")
    promo_flag: int = Field(..., ge=0, le=1, description="Promotion flag")


class ForecastResponse(BaseModel):
    predicted_sales: float