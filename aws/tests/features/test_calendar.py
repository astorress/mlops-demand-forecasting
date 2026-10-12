from datetime import datetime

import pytest

from src.features.calendar import calendar_features, parse_timestamp


@pytest.mark.parametrize("value", [
    "2024-12-25 18:00:00",
    "2024-12-25 18:00",
    "2024-12-25T18:00:00",
    "2024-12-25 18:00:00.000",
])
def test_parse_timestamp_formats(value):
    assert parse_timestamp(value) == datetime(2024, 12, 25, 18)


def test_parse_timestamp_rejects_garbage():
    with pytest.raises(ValueError):
        parse_timestamp("ayer")


def test_calendar_features_christmas_wednesday():
    features = calendar_features(datetime(2024, 12, 25, 18))
    assert features == {
        "hour": 18,
        "day_of_week": 2,
        "month": 12,
        "is_weekend": 0,
        "is_holiday": 1,
    }


def test_sunday_is_weekend():
    assert calendar_features(datetime(2024, 7, 7, 3))["is_weekend"] == 1
