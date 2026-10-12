"""Mirror en pandas del job de curación de Glue (src/jobs/glue/
curate_demand.py), solo para experimentación local (src.scripts.train_local)
y tests. Mantener ambos sincronizados: en AWS manda el job de Glue.

raw (timestamp, demand_kwh, temperature_c) → curated:
    * tipos, timestamps únicos y ordenados,
    * calendario (hora, día de la semana, mes, fin de semana, festivo),
    * lags por *tiempo* (t − 24 h, t − 168 h) y media móvil de las 168 horas
      anteriores (sin incluir t: evita fuga del target),
    * descarta la primera semana (sin lag de 168 h).
"""
import pandas as pd

from src.features.calendar import COLOMBIAN_HOLIDAYS
from src.features.schema import (
    FEATURES,
    TARGET,
    TIMESTAMP,
    TIMESTAMP_FORMAT,
)


def curate(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[[TIMESTAMP, TARGET, "temperature_c"]].copy()
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    df[TARGET] = pd.to_numeric(df[TARGET])
    df["temperature_c"] = pd.to_numeric(df["temperature_c"])
    df = df.drop_duplicates(subset=[TIMESTAMP], keep="last")
    df = df.sort_values(TIMESTAMP).reset_index(drop=True)

    moments = df[TIMESTAMP].dt
    df["hour"] = moments.hour
    df["day_of_week"] = moments.dayofweek
    df["month"] = moments.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = [
        int((month, day) in COLOMBIAN_HOLIDAYS)
        for month, day in zip(moments.month, moments.day)
    ]

    demand = df.set_index(TIMESTAMP)[TARGET]
    for hours in (24, 168):
        lagged = demand.reindex(df[TIMESTAMP] - pd.Timedelta(hours=hours))
        df[f"demand_lag_{hours}h"] = lagged.to_numpy()
    df["demand_rolling_mean_7d"] = (demand.rolling(
        "168h", closed="left").mean().round(2).to_numpy())

    df = df.dropna(subset=["demand_lag_168h"]).reset_index(drop=True)
    df[TIMESTAMP] = df[TIMESTAMP].dt.strftime(TIMESTAMP_FORMAT)
    return df[[TIMESTAMP] + FEATURES + [TARGET]]
