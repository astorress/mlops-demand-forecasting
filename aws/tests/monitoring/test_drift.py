import pytest

from src.features.schema import MONITORED_FEATURES
from src.monitoring.drift import (
    NULL,
    OTHER,
    build_profile,
    categorical_distribution,
    evaluate,
    numeric_distribution,
    psi,
)


@pytest.fixture
def rows(curated_df):
    return curated_df.to_dict("records")


@pytest.fixture
def profile(rows):
    return build_profile(rows, MONITORED_FEATURES, [])


def test_psi_is_zero_for_identical_distributions():
    assert psi({
        "a": 0.5,
        "b": 0.5
    }, {
        "a": 0.5,
        "b": 0.5
    }) == pytest.approx(0)


def test_numeric_distribution_buckets_and_nulls():
    distribution = numeric_distribution([1, 5, 9, None, "\\N"], [3, 7])
    assert distribution == {
        "0": 0.2,
        "1": 0.2,
        "2": 0.2,
        NULL: 0.4,
    }


def test_categorical_distribution_groups_unseen_values():
    distribution = categorical_distribution(["a", "b", "z"], {"a", "b"})
    assert distribution[OTHER] == pytest.approx(1 / 3)


def test_profile_covers_monitored_features(profile, rows):
    assert profile["samples"] == len(rows)
    assert set(profile["numeric"]) == set(MONITORED_FEATURES)
    assert sum(profile["numeric"]["temperature_c"]["expected"].values()) == (
        pytest.approx(1))


def test_same_data_does_not_drift(profile, rows):
    report = evaluate(profile, {}, rows[::3])
    assert report["status"] == "ok"
    assert report["max_psi"] < 0.1


def test_heat_wave_drifts(profile, rows):
    hot = [{
        **row, "temperature_c": row["temperature_c"] + 5
    } for row in rows]
    report = evaluate(profile, {}, hot)
    assert report["status"] == "drift"
    assert "temperature_c" in report["drifted_features"]


def test_too_few_rows_is_insufficient_data(profile, rows):
    report = evaluate(profile, {
        "min_samples": 30
    }, rows[:5])
    assert report["status"] == "insufficient_data"
