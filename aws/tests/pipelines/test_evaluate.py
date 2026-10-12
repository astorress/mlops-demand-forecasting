import json
import sys
import tarfile

import joblib

from src.pipelines.training import evaluate


def test_reads_model_tarball_and_writes_report(tmp_path, monkeypatch,
                                               trained_model, splits):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    joblib.dump(trained_model, tmp_path / "model.joblib")
    with tarfile.open(model_dir / "model.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "model.joblib", arcname="model.joblib")
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    splits[2].to_csv(test_dir / "test.csv", index=False)
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--model-dir",
        str(model_dir), "--test-dir",
        str(test_dir), "--output-dir",
        str(tmp_path / "evaluation")
    ])

    evaluate.main()

    report = json.loads(
        (tmp_path / "evaluation" / "evaluation.json").read_text())
    assert report["regression_metrics"]["mape"]["value"] > 0
    assert report["samples"] == len(splits[2])
