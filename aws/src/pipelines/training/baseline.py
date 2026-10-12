"""Paso Processing de SageMaker: perfil de referencia para el monitoreo de
data drift.

Entrada (/opt/ml/processing/input/baseline_dataset_input/baseline.csv): las
        features de entrenamiento que escribe PrepareData.
Salida (/opt/ml/processing/output):
    statistics.json   perfil PSI de referencia (bins y frecuencias por feature)
    constraints.json  umbrales de drift que usa la Lambda DataDrift

Ambos archivos se registran con la versión del modelo como sus baselines de
drift y se publican en el contrato de despliegue cuando se aprueba.
"""
import argparse
import csv
import json
import logging
from pathlib import Path

from src.features.schema import MONITORED_FEATURES
from src.monitoring.drift import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_PSI_THRESHOLD,
    build_profile,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path("/opt/ml/processing")


def read_rows(input_dir: Path) -> list:
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as file:
            rows.extend(csv.DictReader(file))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(BASE_DIR / "input" / "baseline_dataset_input"),
    )
    parser.add_argument("--output-dir", default=str(BASE_DIR / "output"))
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=DEFAULT_PSI_THRESHOLD,
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(Path(args.input_dir))
    if not rows:
        raise ValueError(f"No baseline rows in {args.input_dir}")
    profile = build_profile(rows, MONITORED_FEATURES, [], args.bins)
    constraints = {
        "psi_threshold": args.psi_threshold,
        "min_samples": args.min_samples,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "statistics.json").write_text(json.dumps(profile, indent=2))
    (output_dir / "constraints.json").write_text(
        json.dumps(constraints, indent=2))
    logger.info("Profile of %s rows written to %s", len(rows), output_dir)


if __name__ == "__main__":
    main()
