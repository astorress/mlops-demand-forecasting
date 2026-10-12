"""Features de calendario en Python puro (sin dependencias): las usan el
generador de datos, la Lambda de predicción y el mirror local de la curación.

El job de Glue (src/jobs/glue/curate_demand.py) repite la lista de festivos:
mantener ambos sincronizados.
"""
from datetime import datetime

from src.features.schema import TIMESTAMP_FORMAT

# Festivos colombianos (aproximados, fecha fija) como pares (mes, día): los
# mismos del proyecto open source.
COLOMBIAN_HOLIDAYS = {
    (1, 1),  # Año Nuevo
    (1, 6),  # Reyes Magos
    (3, 25),  # San José
    (4, 13),  # Jueves Santo
    (4, 14),  # Viernes Santo
    (5, 1),  # Día del Trabajo
    (5, 19),  # Ascensión
    (6, 9),  # Corpus Christi
    (6, 16),  # Sagrado Corazón
    (6, 29),  # San Pedro y San Pablo
    (7, 20),  # Independencia
    (8, 7),  # Batalla de Boyacá
    (8, 19),  # Asunción
    (10, 12),  # Día de la Raza
    (11, 2),  # Todos los Santos
    (11, 11),  # Independencia de Cartagena
    (12, 8),  # Inmaculada Concepción
    (12, 25),  # Navidad
}

_INPUT_FORMATS = (
    TIMESTAMP_FORMAT,
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S.%f",
)


def is_holiday(moment: datetime) -> int:
    return int((moment.month, moment.day) in COLOMBIAN_HOLIDAYS)


def parse_timestamp(value) -> datetime:
    """'2024-07-01 18:00:00' (también con T, sin segundos o con fracción)."""
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for pattern in _INPUT_FORMATS:
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    raise ValueError(f"Invalid timestamp: {value!r}")


def calendar_features(moment: datetime) -> dict:
    """day_of_week: lunes = 0 … domingo = 6 (convención de pandas)."""
    day_of_week = moment.weekday()
    return {
        "hour": moment.hour,
        "day_of_week": day_of_week,
        "month": moment.month,
        "is_weekend": int(day_of_week >= 5),
        "is_holiday": is_holiday(moment),
    }
