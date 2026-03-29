import json
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import yaml

from app.config import CONFIG_PATH, LOG_DIR


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str = "forecasting") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler = RotatingFileHandler(
            LOG_DIR / "app.log",
            maxBytes=1_000_000,
            backupCount=3,
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def create_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    lags = config["features"]["lags"]
    rolling_windows = config["features"]["rolling_windows"]

    for lag in lags:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["sales"].shift(1).rolling(window).std()

    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    df = df.dropna().reset_index(drop=True)
    return df


def save_metadata(metadata_path, payload: dict) -> None:
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)