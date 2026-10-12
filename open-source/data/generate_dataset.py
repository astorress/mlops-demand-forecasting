"""
Generador de dataset sintético de demanda eléctrica horaria.

Genera ~26,136 registros (3 años, 2022-2024, menos la primera semana por los lags) con las siguientes columnas:
- timestamp: datetime horario
- demand_kwh: demanda eléctrica (target)
- temperature_c: temperatura correlacionada con demanda
- hour, day_of_week, month, is_weekend, is_holiday
- demand_lag_24h, demand_lag_168h, demand_rolling_mean_7d

Uso:
    python data/generate_dataset.py
    python data/generate_dataset.py --output data/raw/demand.csv --seed 42
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Colombian holidays (approximate) - month, day pairs
COLOMBIAN_HOLIDAYS = [
    (1, 1),   # Año Nuevo
    (1, 6),   # Reyes Magos (aproximado)
    (3, 25),  # San José (aproximado)
    (4, 13),  # Jueves Santo (aproximado)
    (4, 14),  # Viernes Santo (aproximado)
    (5, 1),   # Día del Trabajo
    (5, 19),  # Ascensión (aproximado)
    (6, 9),   # Corpus Christi (aproximado)
    (6, 16),  # Sagrado Corazón (aproximado)
    (6, 29),  # San Pedro y San Pablo
    (7, 20),  # Independencia
    (8, 7),   # Batalla de Boyacá
    (8, 19),  # Asunción (aproximado)
    (10, 12), # Día de la Raza (aproximado)
    (11, 2),  # Todos los Santos (aproximado)
    (11, 11), # Independencia de Cartagena (aproximado)
    (12, 8),  # Inmaculada Concepción
    (12, 25), # Navidad
]

def is_holiday(dt: pd.Timestamp) -> int:
    """Retorna 1 si el timestamp corresponde a un feriado colombiano."""
    return int((dt.month, dt.day) in COLOMBIAN_HOLIDAYS)

def generate_demand(
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Genera la serie de demanda eléctrica con múltiples componentes:
    - Tendencia lineal suave creciente
    - Estacionalidad anual (verano/invierno colombiano)
    - Estacionalidad semanal (fin de semana vs laborable)
    - Estacionalidad diaria (hora pico 18-21h, valle 2-5h)
    - Ruido gaussiano controlado
    """
    n = len(timestamps)
    t = np.arange(n)

    # 1. Nivel base de demanda (kWh)
    base = 1000.0

    # 2. Tendencia lineal (crece ~10% en 3 años)
    trend = 0.0038 * t  # ~100 kWh de incremento total

    # 3. Estacionalidad anual (Colombia: más calor = más A/C = más demanda en junio-agosto)
    day_of_year = timestamps.dayofyear.values
    annual_seasonality = 80 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # 4. Estacionalidad semanal (laborables más demanda que fines de semana)
    dow = timestamps.dayofweek.values
    weekly_seasonality = np.where(dow < 5, 60, -80)  # +60 laborable, -80 fin de semana

    # 5. Estacionalidad diaria (pico 18-21h, valle 2-5h)
    hour = timestamps.hour.values
    # Curva de carga diaria con doble pico (mañana y tarde)
    daily_seasonality = (
        120 * np.sin(np.pi * (hour - 6) / 12) * (hour >= 6) * (hour <= 22)
        + 40 * np.sin(np.pi * (hour - 10) / 6) * (hour >= 8) * (hour <= 14)
        - 150 * ((hour >= 2) & (hour <= 5)).astype(float)
    )

    # 6. Efecto feriados (reducción de demanda similar a fin de semana)
    holiday_effect = np.array([is_holiday(ts) * (-70) for ts in timestamps])

    # 7. Ruido gaussiano (σ = 50 kWh)
    noise = rng.normal(0, 50, n)

    demand = base + trend + annual_seasonality + weekly_seasonality + daily_seasonality + holiday_effect + noise

    # Asegurar que la demanda sea siempre positiva
    demand = np.maximum(demand, 50)

    return demand

def generate_temperature(
    timestamps: pd.DatetimeIndex,
    demand: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Genera temperatura correlacionada con demanda.
    Colombia: ~15-32°C según altitud y época, aquí simulamos ciudad costera.
    """
    hour = timestamps.hour.values
    day_of_year = timestamps.dayofyear.values

    # Base temperatura (ciudad costera colombiana ~26°C)
    base_temp = 26.0

    # Variación diaria (más calor al mediodía)
    daily_var = 6 * np.sin(np.pi * (hour - 6) / 12) * (hour >= 6).astype(float)

    # Variación anual
    annual_var = 3 * np.sin(2 * np.pi * (day_of_year - 60) / 365)

    # Correlación con demanda (normalizada)
    demand_normalized = (demand - demand.mean()) / demand.std()
    demand_corr = 0.3 * demand_normalized

    # Ruido
    noise = rng.normal(0, 1.5, len(timestamps))

    temperature = base_temp + daily_var + annual_var + demand_corr + noise
    temperature = np.clip(temperature, -10, 50)  # límites realistas

    return temperature

def generate_dataset(output_path: str, seed: int = 42) -> pd.DataFrame:
    """
    Genera el dataset completo de demanda eléctrica horaria.

    Args:
        output_path: Ruta donde guardar el CSV.
        seed: Semilla para reproducibilidad.

    Returns:
        DataFrame con todas las columnas.
    """
    rng = np.random.default_rng(seed)

    print("Generando timestamps (2022-01-01 a 2024-12-31)...")
    timestamps = pd.date_range(
        start="2022-01-01 00:00:00",
        end="2024-12-31 23:00:00",
        freq="h"
    )

    print(f"  Total de registros: {len(timestamps):,}")

    print("Generando serie de demanda eléctrica...")
    demand = generate_demand(timestamps, rng)

    print("Generando temperatura correlacionada...")
    temperature = generate_temperature(timestamps, demand, rng)

    print("Construyendo DataFrame...")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_kwh": np.round(demand, 2),
        "temperature_c": np.round(temperature, 2),
        "hour": timestamps.hour,
        "day_of_week": timestamps.dayofweek,
        "month": timestamps.month,
        "is_weekend": (timestamps.dayofweek >= 5).astype(int),
        "is_holiday": [is_holiday(ts) for ts in timestamps],
    })

    # Lags
    print("Calculando lags y medias móviles...")
    df["demand_lag_24h"] = df["demand_kwh"].shift(24)
    df["demand_lag_168h"] = df["demand_kwh"].shift(168)
    # shift(1): la media móvil solo usa horas anteriores; incluir la hora actual
    # filtraría el target en las features (data leakage).
    df["demand_rolling_mean_7d"] = (
        df["demand_kwh"]
        .shift(1)
        .rolling(window=168, min_periods=1)
        .mean()
        .round(2)
    )

    # Eliminar filas con NaN en lags críticos (primera semana)
    df = df.dropna(subset=["demand_lag_168h"]).reset_index(drop=True)

    print(f"  Registros finales: {len(df):,}")
    print(f"  Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print(f"  demand_kwh — min: {df['demand_kwh'].min():.1f}, max: {df['demand_kwh'].max():.1f}, mean: {df['demand_kwh'].mean():.1f}")

    # Guardar
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset guardado en: {output_path}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera el dataset sintético de demanda eléctrica")
    parser.add_argument("--output", default="data/raw/demand.csv", help="Ruta de salida del CSV")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    df = generate_dataset(output_path=args.output, seed=args.seed)
    print("\nPrimeras filas:")
    print(df.head())
    print("\nTipos de datos:")
    print(df.dtypes)
