import json
import joblib
import pandas as pd

from app.config import MODEL_PATH, METADATA_PATH


class ForecastService:
    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def predict(self, payload: dict) -> dict:
        input_df = pd.DataFrame([payload])
        prediction = self.model.predict(input_df)[0]

        return {
            "predicted_sales": float(prediction),
            "model_name": self.metadata["model_name"],
            "model_version": self.metadata["model_version"],
        }