import numpy as np

from src.data.preprocessing import FEATURE_COLS, create_features, train_val_test_split
from src.data.validation import validate_dataset


def test_dataset_shape_and_columns(raw_df):
    # 2022-2024 horario (2024 es bisiesto: 26 304 h) menos la primera semana (lags de 168 h)
    assert len(raw_df) == 26_304 - 168
    for col in ["timestamp", "demand_kwh", "temperature_c", "demand_lag_24h", "demand_lag_168h"]:
        assert col in raw_df.columns
    assert raw_df["timestamp"].is_unique


def test_rolling_mean_does_not_leak_target(raw_df):
    # La media móvil de la hora t solo usa horas anteriores a t
    expected = raw_df["demand_kwh"].shift(1).rolling(168, min_periods=1).mean().round(2)
    assert np.allclose(raw_df["demand_rolling_mean_7d"].iloc[200:], expected.iloc[200:], atol=0.01)


def test_validation_passes_on_generated_data(raw_df):
    result = validate_dataset(raw_df)
    assert result["passed"], result["summary"]
    # Valida con Great Expectations, no con el fallback manual
    assert "modo manual" not in result["summary"]


def test_validation_detects_bad_rows(raw_df):
    bad = raw_df.head(100).copy()
    bad.loc[0, "hour"] = 30
    bad.loc[1, "demand_kwh"] = -5
    result = validate_dataset(bad)
    assert not result["passed"]
    assert {"hour_range", "demand_kwh_positive"} <= set(result["failed_checks"])


def test_features_keep_real_units(raw_df):
    df = create_features(raw_df)
    assert set(FEATURE_COLS) <= set(df.columns)
    # Sin escalado: el target y los lags siguen en kWh
    assert df["demand_kwh"].equals(raw_df["demand_kwh"])
    assert df["is_peak_hour"].isin([0, 1]).all()


def test_temporal_split_has_no_overlap(raw_df):
    train, val, test = train_val_test_split(create_features(raw_df))
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(val) + len(test) == len(raw_df)
