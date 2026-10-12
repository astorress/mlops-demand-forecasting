"""Contrato de datos compartido por todos los pasos de ML (preprocess, train,
inference, evaluate) y por los clientes online/batch.

El orden de FEATURES es el orden de una fila CSV sin encabezado enviada al
modelo (endpoint, Batch Transform y monitor de drift). Cambiarlo rompe a
todos los consumidores: nueva versión del modelo y redeploy de todos juntos.

FEATURES son las columnas de la tabla curada (calendario, temperatura, lags y
media móvil). Los encodings cíclicos e interacciones del proyecto open source
se calculan dentro del modelo (src/features/engineering.py), así el cliente
nunca los envía y no hay training/serving skew.
"""
TIMESTAMP = "timestamp"
TARGET = "demand_kwh"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
]

LAG_FEATURES = [
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_mean_7d",
]

FEATURES = ["temperature_c"] + CALENDAR_FEATURES + LAG_FEATURES

# Todas las features son numéricas (el calendario ya viene codificado).
NUMERIC_FEATURES = FEATURES
CATEGORICAL_FEATURES: list[str] = []

# Features vigiladas por el monitor de drift. Las de calendario se excluyen:
# su distribución depende de *qué horas* se piden en la ventana analizada, no
# de un cambio en los datos.
MONITORED_FEATURES = ["temperature_c"] + LAG_FEATURES
