import joblib
import pandas as pd

from app.config import MODEL_PATH


class ForecastService:
    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)

    def predict(
        self,
        lag_1: float,
        lag_2: float,
        lag_3: float,
        rolling_mean_3: float,
        month: int,
        promo_flag: int,
    ) -> float:
        input_df = pd.DataFrame(
            [
                {
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "lag_3": lag_3,
                    "rolling_mean_3": rolling_mean_3,
                    "month": month,
                    "promo_flag": promo_flag,
                }
            ]
        )
        prediction = self.model.predict(input_df)[0]
        return float(prediction)