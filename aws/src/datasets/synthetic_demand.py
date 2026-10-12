"""Generador de la serie horaria sintética de demanda eléctrica, en Python
puro para correr dentro de la Lambda de ingesta sin numpy/pandas.

Es la misma receta del proyecto open source (open-source/data/
generate_dataset.py): nivel base + tendencia + estacionalidad anual, semanal y
diaria + festivos + ruido, y una temperatura correlacionada con la demanda.
La diferencia es *qué* produce: aquí solo las mediciones crudas (timestamp,
demand_kwh, temperature_c). El calendario, los lags y la media móvil los
calcula la curación (Glue), como pasaría con datos reales de un medidor.
"""
import math
import random
from datetime import datetime, timedelta

from src.features.calendar import is_holiday
from src.features.schema import (
    TARGET,
    TIMESTAMP,
    TIMESTAMP_FORMAT,
)

COLUMNS = [TIMESTAMP, TARGET, "temperature_c"]

# Deriva simulada (simulate_drift): ola de calor + crecimiento de la demanda
# en los últimos DRIFT_DAYS días de la serie.
DRIFT_DAYS = 120
DRIFT_DEMAND_FACTOR = 1.18
DRIFT_TEMPERATURE_SHIFT = 4.0


def _hours(start: datetime, end: datetime) -> list:
    count = int((end - start).total_seconds() // 3600) + 1
    return [start + timedelta(hours=offset) for offset in range(count)]


def _daily_profile(hour: int) -> float:
    """Curva de carga diaria: pico de la tarde, hombro de mediodía y valle
    de madrugada."""
    value = 0.0
    if 6 <= hour <= 22:
        value += 120 * math.sin(math.pi * (hour - 6) / 12)
    if 8 <= hour <= 14:
        value += 40 * math.sin(math.pi * (hour - 10) / 6)
    if 2 <= hour <= 5:
        value -= 150
    return value


def _demand(moment: datetime, index: int, rng: random.Random) -> float:
    day_of_year = moment.timetuple().tm_yday
    components = [
        1000.0,  # nivel base (kWh)
        0.0038 * index,  # tendencia: ~+10 % en 3 años
        80 * math.sin(2 * math.pi * (day_of_year - 80) / 365),  # anual
        60 if moment.weekday() < 5 else -80,  # laborable vs fin de semana
        _daily_profile(moment.hour),  # curva diaria
        -70 * is_holiday(moment),  # festivos
        rng.gauss(0, 50),  # ruido
    ]
    return max(sum(components), 50.0)


def _temperature(moment: datetime, demand_z: float,
                 rng: random.Random) -> float:
    """Ciudad costera (~26 °C) con ciclo diario/anual, correlacionada con la
    demanda estandarizada."""
    day_of_year = moment.timetuple().tm_yday
    daily = 0.0
    if moment.hour >= 6:
        daily = 6 * math.sin(math.pi * (moment.hour - 6) / 12)
    annual = 3 * math.sin(2 * math.pi * (day_of_year - 60) / 365)
    value = 26.0 + daily + annual + 0.3 * demand_z + rng.gauss(0, 1.5)
    return min(max(value, -10.0), 50.0)


def generate(start: datetime,
             end: datetime,
             seed: int = 42,
             simulate_drift: bool = False) -> list:
    """Filas {timestamp, demand_kwh, temperature_c}, una por hora, de start a
    end (ambos incluidos)."""
    rng = random.Random(seed)
    moments = _hours(start, end)
    demand = [
        _demand(moment, index, rng) for index, moment in enumerate(moments)
    ]

    mean = sum(demand) / len(demand)
    std = math.sqrt(sum((value - mean)**2 for value in demand) / len(demand))
    std = std or 1.0
    temperature = [
        _temperature(moment, (value - mean) / std, rng)
        for moment, value in zip(moments, demand)
    ]

    if simulate_drift:
        drift_start = end - timedelta(days=DRIFT_DAYS)
        for index, moment in enumerate(moments):
            if moment > drift_start:
                demand[index] *= DRIFT_DEMAND_FACTOR
                temperature[index] = min(
                    temperature[index] + DRIFT_TEMPERATURE_SHIFT, 50.0)

    return [{
        TIMESTAMP: moment.strftime(TIMESTAMP_FORMAT),
        TARGET: round(value, 2),
        "temperature_c": round(temp, 2),
    } for moment, value, temp in zip(moments, demand, temperature)]
