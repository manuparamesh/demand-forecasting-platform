import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from app.config import RAW_DATA_PATH, MODEL_DIR, MODEL_PATH
from app.utils import create_features


def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH)
    df = create_features(df)

    feature_cols = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "month", "promo_flag"]
    target_col = "sales"

    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model trained successfully.")
    print(f"MAE: {mae:.4f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()