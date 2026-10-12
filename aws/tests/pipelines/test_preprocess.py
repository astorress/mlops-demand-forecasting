import gzip
import json

import pandas as pd
import pytest

from src.features.schema import (
    FEATURES,
    TARGET,
    TIMESTAMP,
)
from src.pipelines.training.preprocess import (
    DataValidationError,
    read_dataset,
    split,
    validate,
    write_outputs,
)


def test_reads_athena_json_lines_and_gzip(tmp_path, curated_df):
    records = curated_df.head(10).to_dict("records")
    plain = "\n".join(json.dumps(record) for record in records[:5])
    (tmp_path / "part-0").write_text(plain)
    (tmp_path / "part-1").write_bytes(
        gzip.compress("\n".join(json.dumps(r) for r in records[5:]).encode()))
    (tmp_path / "part-1.metadata").write_text("ignored")
    assert len(read_dataset(str(tmp_path))) == 10


def test_temporal_split_without_overlap(splits):
    train, validation, test = splits
    assert train[TIMESTAMP].max() < validation[TIMESTAMP].min()
    assert validation[TIMESTAMP].max() < test[TIMESTAMP].min()
    assert test[TIMESTAMP].dt.normalize().nunique() == 30
    assert test[TIMESTAMP].min() == pd.Timestamp("2023-06-01")


def test_default_split_matches_open_source_dates():
    hours = pd.date_range("2022-01-08", "2024-12-31 23:00", freq="h")
    df = pd.DataFrame({
        TIMESTAMP: hours
    })
    train, validation, test = split(df, 184, 182)
    assert train[TIMESTAMP].max() == pd.Timestamp("2023-12-31 23:00")
    assert validation[TIMESTAMP].min() == pd.Timestamp("2024-01-01")
    assert test[TIMESTAMP].min() == pd.Timestamp("2024-07-01")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda df: df.drop(columns=["demand_lag_24h"]), "Missing columns"),
        (lambda df: df.assign(demand_kwh=-1), "Non-positive"),
        (lambda df: df.assign(temperature_c=80), "temperature_c"),
        (lambda df: df.assign(demand_lag_168h=None), "Null values"),
    ],
)
def test_validation_gate(curated_df, change, message):
    with pytest.raises(DataValidationError, match=message):
        validate(change(curated_df.copy()), min_rows=100)


def test_too_few_rows(curated_df):
    with pytest.raises(DataValidationError, match="at least"):
        validate(curated_df.head(50), min_rows=100)


def test_outputs(tmp_path, splits):
    write_outputs(*splits, str(tmp_path))
    train = pd.read_csv(tmp_path / "train" / "train.csv")
    baseline = pd.read_csv(tmp_path / "baseline" / "baseline.csv")
    assert list(train.columns) == [TIMESTAMP] + FEATURES + [TARGET]
    assert list(baseline.columns) == FEATURES
