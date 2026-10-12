import pytest

from src.features.engineering import MODEL_FEATURES, add_features
from src.features.schema import FEATURES

# FEATURE_COLS de open-source/src/data/preprocessing.py.
OPEN_SOURCE_FEATURES = [
    "temperature_c", "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "demand_lag_24h", "demand_lag_168h", "demand_rolling_mean_7d", "hour_sin",
    "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "temp_x_hour",
    "is_peak_hour"
]


def test_same_features_as_open_source_project():
    assert MODEL_FEATURES == OPEN_SOURCE_FEATURES


def test_add_features(curated_df):
    df = add_features(curated_df[FEATURES].head(48))
    assert list(df.columns) == MODEL_FEATURES
    peak = df[df["hour"] == 19].iloc[0]
    assert peak["is_peak_hour"] == 1
    assert peak["temp_x_hour"] == pytest.approx(peak["temperature_c"] * 19)
    assert df["hour_sin"].between(-1, 1).all()
