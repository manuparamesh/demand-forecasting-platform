from datetime import datetime

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from app.config import METADATA_PATH, MODEL_DIR, MODEL_PATH, RAW_DATA_PATH
from app.utils import create_features, load_config, save_metadata, setup_logger


def main() -> None:
    logger = setup_logger("training")
    config = load_config()

    logger.info("Loading raw dataset from %s", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    df = create_features(df, config)

    target_col = config["training"]["target_column"]
    feature_cols = [col for col in df.columns if col not in ["date", target_col]]

    train_size = int(len(df) * (1 - config["training"]["test_size_ratio"]))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    params = config["model"]["params"]
    model = XGBRegressor(**params)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info("Starting MLflow run")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("feature_count", len(feature_cols))

        logger.info("Training model")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        metadata = {
            "model_name": config["model"]["name"],
            "model_version": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
            "trained_at_utc": datetime.utcnow().isoformat(),
            "metrics": {
                "mae": mae,
                "rmse": rmse,
            },
            "features": feature_cols,
        }
        save_metadata(METADATA_PATH, metadata)

        logger.info("Training complete | MAE=%.4f | RMSE=%.4f", mae, rmse)
        logger.info("Model saved to %s", MODEL_PATH)
        logger.info("Metadata saved to %s", METADATA_PATH)


if __name__ == "__main__":
    main()