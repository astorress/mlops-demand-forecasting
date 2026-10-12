import pandas as pd

from src.features.calendar import calendar_features, parse_timestamp
from src.features.curation import curate
from src.features.schema import (
    FEATURES,
    TARGET,
    TIMESTAMP,
)


def test_curated_schema_and_first_week_dropped(raw_df, curated_df):
    assert list(curated_df.columns) == [TIMESTAMP] + FEATURES + [TARGET]
    assert len(curated_df) == len(raw_df) - 168
    assert curated_df[FEATURES].notna().all().all()
    assert curated_df[TIMESTAMP].iloc[0] == "2023-01-08 00:00:00"


def test_lags_are_by_time(raw_df, curated_df):
    demand = raw_df.set_index(TIMESTAMP)[TARGET]
    row = curated_df.iloc[500]
    moment = pd.Timestamp(row[TIMESTAMP])
    lag_24 = (moment - pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    assert row["demand_lag_24h"] == demand[lag_24]


def test_rolling_mean_excludes_current_hour(raw_df, curated_df):
    demand = raw_df[TARGET]
    # Fila curada i = fila cruda i + 168: media de las 168 horas anteriores.
    expected = round(demand.iloc[100:268].mean(), 2)
    assert curated_df["demand_rolling_mean_7d"].iloc[100] == expected


def test_gaps_leave_lags_empty(raw_df):
    # Sin la hora t − 24 h, el lag de t queda nulo (no se corre de posición).
    gap = raw_df.drop(index=300)
    curated = curate(gap)
    moment = pd.Timestamp(raw_df[TIMESTAMP].iloc[324])
    row = curated[curated[TIMESTAMP] == moment.strftime("%Y-%m-%d %H:%M:%S")]
    assert row["demand_lag_24h"].isna().all()


def test_calendar_matches_python_calendar(curated_df):
    for _, row in curated_df.sample(50, random_state=0).iterrows():
        expected = calendar_features(parse_timestamp(row[TIMESTAMP]))
        assert {
            name: row[name]
            for name in expected
        } == expected
