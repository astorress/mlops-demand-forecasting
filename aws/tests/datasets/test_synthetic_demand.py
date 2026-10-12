from datetime import datetime

from src.datasets.synthetic_demand import COLUMNS, generate


def _generate(**kwargs):
    return generate(datetime(2024, 1, 1), datetime(2024, 3, 31, 23), **kwargs)


def test_one_row_per_hour_with_raw_columns():
    rows = _generate()
    assert len(rows) == 91 * 24
    assert list(rows[0]) == COLUMNS
    assert rows[0]["timestamp"] == "2024-01-01 00:00:00"
    assert rows[-1]["timestamp"] == "2024-03-31 23:00:00"
    assert all(row["demand_kwh"] > 0 for row in rows)


def test_seed_makes_it_reproducible():
    assert _generate(seed=1) == _generate(seed=1)
    assert _generate(seed=1) != _generate(seed=2)


def test_daily_pattern_peaks_in_the_evening():
    rows = _generate()
    by_hour = {
        hour: [
            row["demand_kwh"]
            for row in rows
            if row["timestamp"][11:13] == f"{hour:02d}"
        ]
        for hour in (3, 19)
    }
    assert sum(by_hour[19]) / len(by_hour[19]) > sum(by_hour[3]) / len(
        by_hour[3])


def test_drift_only_changes_the_last_months():
    start, end = datetime(2023, 1, 1), datetime(2023, 12, 31, 23)
    base = generate(start, end, seed=3)
    drifted = generate(start, end, seed=3, simulate_drift=True)
    assert drifted[-1]["demand_kwh"] > base[-1]["demand_kwh"]
    assert drifted[-1]["temperature_c"] > base[-1]["temperature_c"]
    assert drifted[0] == base[0]
