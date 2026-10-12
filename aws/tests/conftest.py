import os
from datetime import datetime

import pandas as pd
import pytest

# Los services crean sus clientes boto3 al importarse.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")

TEST_DAYS = 30
VALIDATION_DAYS = 30


def make_raw(start: str = "2023-01-01",
             end: str = "2023-06-30",
             seed: int = 0) -> pd.DataFrame:
    """Serie cruda sintética (mismo generador de la Lambda de ingesta)."""
    from src.datasets.synthetic_demand import generate

    rows = generate(
        datetime.fromisoformat(start),
        datetime.fromisoformat(end).replace(hour=23),
        seed,
    )
    return pd.DataFrame.from_records(rows)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def curated_df(raw_df) -> pd.DataFrame:
    from src.features.curation import curate

    return curate(raw_df)


@pytest.fixture(scope="session")
def splits(curated_df):
    from src.pipelines.training.preprocess import split, validate

    return split(validate(curated_df, min_rows=100), TEST_DAYS, VALIDATION_DAYS)


@pytest.fixture(scope="session")
def trained_model(splits):
    from src.models.train import train

    return train(splits[0], {
        "n_estimators": 200
    }, splits[1])
