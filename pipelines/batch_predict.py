import pandas as pd
from app.predictor import ForecastService
from app.config import RAW_DATA_PATH
from app.utils import create_features


def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH)
    df = create_features(df)

    feature_cols = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "month", "promo_flag"]

    service = ForecastService()
    df["predicted_sales"] = df[feature_cols].apply(
        lambda row: service.predict(
            lag_1=row["lag_1"],
            lag_2=row["lag_2"],
            lag_3=row["lag_3"],
            rolling_mean_3=row["rolling_mean_3"],
            month=int(row["month"]),
            promo_flag=int(row["promo_flag"]),
        ),
        axis=1,
    )

    print(df[["date", "sales", "predicted_sales"]].tail(10))


if __name__ == "__main__":
    main()