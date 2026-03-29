import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["lag_1"] = df["sales"].shift(1)
    df["lag_2"] = df["sales"].shift(2)
    df["lag_3"] = df["sales"].shift(3)
    df["rolling_mean_3"] = df["sales"].shift(1).rolling(3).mean()
    df["month"] = df["date"].dt.month

    df = df.dropna().reset_index(drop=True)
    return df