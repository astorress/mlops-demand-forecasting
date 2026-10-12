import json
import sys

from src.features.schema import MONITORED_FEATURES
from src.pipelines.training import baseline


def test_writes_profile_and_constraints(tmp_path, monkeypatch, splits):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    splits[0].to_csv(input_dir / "baseline.csv", index=False)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "baseline.py", "--input-dir",
        str(input_dir), "--output-dir",
        str(output_dir)
    ])

    baseline.main()

    profile = json.loads((output_dir / "statistics.json").read_text())
    constraints = json.loads((output_dir / "constraints.json").read_text())
    assert set(profile["numeric"]) == set(MONITORED_FEATURES)
    assert profile["categorical"] == {}
    assert constraints == {
        "psi_threshold": 0.2,
        "min_samples": 30,
    }
