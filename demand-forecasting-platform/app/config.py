from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sales_data.csv"
MODEL_PATH = MODEL_DIR / "forecast_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
CONFIG_PATH = BASE_DIR / "configs" / "training_config.yaml"
LOG_DIR = BASE_DIR / "logs"