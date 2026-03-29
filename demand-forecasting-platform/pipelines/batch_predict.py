import pandas as pd

from app.config import RAW_DATA_PATH
from app.predictor import ForecastService
from app.utils import create_features, load_config, setup_logger


def main() -> None:
    logger = setup_logger("batch_prediction")
    config = load_config()

    logger.info("Loading data for batch scoring")
    df = pd.read_csv(RAW_DATA_PATH)
    df = create_features(df, config)

    feature_cols = [col for col in df.columns if col not in ["date", "sales"]]

    service = ForecastService()

    results = []
    for _, row in df.iterrows():
        prediction = service.predict(row[feature_cols].to_dict())
        results.append(prediction["predicted_sales"])

    df["predicted_sales"] = results

    logger.info("Batch prediction complete")
    print(df[["date", "sales", "predicted_sales"]].tail(10))


if __name__ == "__main__":
    main()