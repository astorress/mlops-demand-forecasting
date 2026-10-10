"""
Preprocesamiento y feature engineering para el dataset de demanda eléctrica.

Funciones principales:
- load_raw_data: carga el CSV crudo
- create_features: ingeniería de features
- train_val_test_split: split temporal
- save_processed_data: guarda datos procesados

No se escalan las features: LightGBM es un modelo de árboles y es invariante a
transformaciones monótonas. Así el modelo trabaja en kWh reales y la API puede
recibir los valores tal cual, sin riesgo de training-serving skew.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columnas usadas como features para el modelo (sin el target)
FEATURE_COLS = [
    "temperature_c", "hour", "day_of_week", "month",
    "is_weekend", "is_holiday",
    "demand_lag_24h", "demand_lag_168h", "demand_rolling_mean_7d",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "temp_x_hour", "is_peak_hour",
]

TARGET_COL = "demand_kwh"


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Carga el dataset crudo desde CSV.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        DataFrame con timestamp como datetime.
    """
    path = os.environ.get("DATA_RAW_PATH", path)
    logger.info(f"Cargando datos desde: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info(f"Datos cargados: {len(df):,} registros, {df.shape[1]} columnas")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features adicionales a partir del dataset crudo.

    Agrega:
    - Encodings cíclicos: hour_sin/cos, dow_sin/cos, month_sin/cos
    - Interacciones: temp_x_hour, is_peak_hour

    Args:
        df: DataFrame crudo.

    Returns:
        DataFrame con features adicionales.
    """
    df = df.copy()

    # --- Encodings cíclicos (capturan periodicidad) ---
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Features de interacción ---
    df["temp_x_hour"] = df["temperature_c"] * df["hour"]
    df["is_peak_hour"] = ((df["hour"] >= 18) & (df["hour"] <= 21)).astype(int)

    logger.info(f"Features creadas. Shape final: {df.shape}")
    return df


def train_val_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal del dataset:
    - Train: 2022-01-01 a 2023-12-31
    - Val:   2024-01-01 a 2024-06-30
    - Test:  2024-07-01 a 2024-12-31

    Se usa split temporal (no aleatorio) para evitar data leakage en series temporales.

    Args:
        df: DataFrame completo con columna 'timestamp'.

    Returns:
        Tupla (train, val, test).
    """
    train_end = "2023-12-31 23:59:59"
    val_end = "2024-06-30 23:59:59"

    train = df[df["timestamp"] <= train_end].copy()
    val = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)].copy()
    test = df[df["timestamp"] > val_end].copy()

    logger.info(
        f"Split temporal: "
        f"train={len(train):,} | val={len(val):,} | test={len(test):,}"
    )

    return train, val, test


def save_processed_data(
    df: pd.DataFrame,
    path: str,
    filename: str = "features.parquet",
) -> str:
    """
    Guarda datos procesados en formato Parquet.

    Args:
        df: DataFrame procesado.
        path: Directorio de destino.
        filename: Nombre del archivo.

    Returns:
        Ruta completa del archivo guardado.
    """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    df.to_parquet(output_path, index=False)
    logger.info(f"Datos guardados en: {output_path}")
    return str(output_path)


def run_preprocessing(
    raw_path: str = "data/raw/demand.csv",
    output_dir: str = "data/processed/",
) -> dict[str, str]:
    """
    Ejecuta el pipeline completo de preprocesamiento.

    Args:
        raw_path: Ruta al CSV crudo.
        output_dir: Directorio de salida.

    Returns:
        Dict con las rutas de los archivos generados.
    """
    # 1. Cargar datos
    df = load_raw_data(raw_path)

    # 2. Feature engineering
    df = create_features(df)

    # 3. Split temporal
    train, val, test = train_val_test_split(df)

    # 4. Guardar
    paths = {}
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        paths[name] = save_processed_data(split_df, output_dir, f"{name}.parquet")

    logger.info("Preprocesamiento completado.")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Preprocesamiento del dataset de demanda eléctrica")
    parser.add_argument("--raw-path", default="data/raw/demand.csv")
    parser.add_argument("--output-dir", default="data/processed/")
    args = parser.parse_args()

    paths = run_preprocessing(
        raw_path=args.raw_path,
        output_dir=args.output_dir,
    )
    print("Archivos generados:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
