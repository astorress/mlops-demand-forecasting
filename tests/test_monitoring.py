import pandas as pd

from src.monitoring.drift_detector import _detect_drift_manual, run_drift_report


def test_no_drift_between_identical_samples(processed_dir, tmp_path):
    train = pd.read_parquet(processed_dir / "train.parquet")
    result = run_drift_report(train, train.copy(), output_path=str(tmp_path / "report.html"))
    assert not result["drift_detected"]
    assert (tmp_path / "report.html").exists()


def test_detects_shifted_temperature(processed_dir):
    train = pd.read_parquet(processed_dir / "train.parquet")
    shifted = train.copy()
    shifted["temperature_c"] = shifted["temperature_c"] + 10
    result = _detect_drift_manual(train, shifted, drift_threshold=0.2)
    assert "temperature_c" in result["drifted_features"]


def test_evidently_report_lists_drifted_features(processed_dir, tmp_path):
    train = pd.read_parquet(processed_dir / "train.parquet")
    shifted = train.copy()
    shifted["temperature_c"] = shifted["temperature_c"] + 10
    result = run_drift_report(train, shifted, output_path=str(tmp_path / "report.html"))
    assert "temperature_c" in result["drifted_features"]
