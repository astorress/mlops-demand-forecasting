"""Feature engineering *dentro* del modelo: los mismos encodings cíclicos e
interacciones de open-source/src/data/preprocessing.py, aplicados como primer
paso del Pipeline de scikit-learn. Así corren idénticos al entrenar y al
servir, y los clientes solo envían las columnas curadas (schema.FEATURES).

Se usa por nombre dentro del modelo serializado: renombrar o mover esta
función obliga a reentrenar.
"""
import numpy as np
import pandas as pd

from src.features.schema import FEATURES

ENGINEERED_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "temp_x_hour",
    "is_peak_hour",
]

# Mismas 17 columnas y mismo orden que FEATURE_COLS del proyecto open source.
MODEL_FEATURES = FEATURES + ENGINEERED_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(df, columns=FEATURES).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["temp_x_hour"] = df["temperature_c"] * df["hour"]
    df["is_peak_hour"] = df["hour"].between(18, 21).astype(int)
    return df[MODEL_FEATURES]
