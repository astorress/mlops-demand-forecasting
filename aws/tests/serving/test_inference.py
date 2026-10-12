import json

import pytest

from src.features.schema import FEATURES
from src.serving.inference import (
    input_fn,
    output_fn,
    predict_fn,
)


def test_csv_and_json_inputs_match(curated_df):
    rows = curated_df[FEATURES].head(3)
    csv_body = rows.to_csv(header=False, index=False)
    json_body = json.dumps({
        "instances": rows.to_dict("records")
    })
    assert input_fn(csv_body.encode(),
                    "text/csv").equals(input_fn(json_body, "application/json"))


def test_json_missing_feature():
    with pytest.raises(ValueError, match="Missing features"):
        input_fn(json.dumps([{
            "hour": 1
        }]), "application/json")


def test_unsupported_content_type():
    with pytest.raises(ValueError, match="Unsupported"):
        input_fn("x", "text/plain")


def test_predict_and_output_formats(trained_model, curated_df):
    data = input_fn(
        curated_df[FEATURES].head(2).to_csv(header=False, index=False),
        "text/csv")
    prediction = predict_fn(data, trained_model)
    body, mimetype = output_fn(prediction, "application/json")
    assert mimetype == "application/json"
    assert len(json.loads(body)["predictions"]) == 2
    body, mimetype = output_fn(prediction, "text/csv")
    assert mimetype == "text/csv"
    assert len(body.splitlines()) == 2
