"""
Detección de data drift usando Evidently AI.

Compara la distribución de los datos de referencia (training) con los datos
actuales (producción) y detecta si hay drift significativo.

Uso:
    python src/monitoring/drift_detector.py
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Features monitoreadas (las más importantes del modelo)
MONITORED_FEATURES = [
    "temperature_c",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_mean_7d",
]

TARGET_COL = "demand_kwh"
DEFAULT_DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.2"))


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "artifacts/drift_report.html",
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    column_mapping: Optional[dict] = None,
) -> dict:
    """
    Genera un reporte de data drift con Evidently AI.

    Compara la distribución de las features entre el dataset de referencia
    (normalmente el training set) y el dataset actual (producción).

    El drift se considera detectado si el dataset drift score supera el threshold,
    o si más del 20% de las features muestran drift individual.

    Args:
        reference_df: DataFrame de referencia (training set).
        current_df: DataFrame actual (datos de producción).
        output_path: Ruta donde guardar el reporte HTML.
        drift_threshold: Umbral de drift score (default: 0.2 equivalente a PSI).
        column_mapping: Mapeo opcional de columnas para Evidently.

    Returns:
        Dict con {drift_detected, drifted_features, drift_score, report_path}
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report
    except ImportError:
        logger.warning("evidently no instalado. Usando detección manual de drift.")
        return _detect_drift_manual(reference_df, current_df, drift_threshold)

    # Seleccionar solo las columnas monitoreadas que existen en ambos DataFrames
    monitored = [c for c in MONITORED_FEATURES if c in reference_df.columns and c in current_df.columns]

    ref = reference_df[monitored + ([TARGET_COL] if TARGET_COL in reference_df.columns else [])].copy()
    cur = current_df[monitored + ([TARGET_COL] if TARGET_COL in current_df.columns else [])].copy()

    # Configurar column mapping
    cm = ColumnMapping()
    if TARGET_COL in ref.columns:
        cm.target = TARGET_COL
    cm.numerical_features = [c for c in monitored if ref[c].dtype in ["float64", "float32"]]
    cm.categorical_features = [c for c in monitored if ref[c].dtype in ["int64", "int32", "object"]]

    # Generar reporte con DataDrift + DataQuality
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=ref, current_data=cur, column_mapping=cm)

    # Guardar reporte HTML
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info(f"Reporte guardado en: {output_path}")

    # Extraer resultados
    result_dict = report.as_dict()

    # Parsear resultados del drift
    drifted_features = []
    drift_scores = []

    try:
        drift_metrics = result_dict.get("metrics", [])
        for metric in drift_metrics:
            if metric.get("metric") == "DatasetDriftMetric":
                drift_result = metric.get("result", {})
                dataset_drift = drift_result.get("dataset_drift", False)
                share_drifted = drift_result.get("share_of_drifted_columns", 0.0)
                drift_scores.append(share_drifted)

            # DataDriftPreset reporta el drift por columna en DataDriftTable
            if metric.get("metric") == "DataDriftTable":
                by_column = metric.get("result", {}).get("drift_by_columns", {})
                drifted_features.extend(
                    col for col, col_result in by_column.items()
                    if col in monitored and col_result.get("drift_detected", False)
                )

    except Exception as e:
        logger.warning(f"Error parseando resultados de Evidently: {e}")

    overall_drift_score = float(drift_scores[0]) if drift_scores else 0.0
    drift_detected = overall_drift_score > drift_threshold or len(drifted_features) > len(monitored) * 0.2

    result = {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "drift_score": overall_drift_score,
        "n_drifted_features": len(drifted_features),
        "n_monitored_features": len(monitored),
        "drift_threshold": drift_threshold,
        "report_path": str(output_path),
    }

    logger.info(
        f"Drift {'DETECTADO' if drift_detected else 'NO detectado'} | "
        f"Score: {overall_drift_score:.3f} | "
        f"Features drifteadas: {drifted_features}"
    )

    return result


def _detect_drift_manual(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_threshold: float,
) -> dict:
    """
    Detección manual de drift usando PSI (Population Stability Index).
    Fallback cuando Evidently no está instalado.
    """
    import numpy as np

    def compute_psi(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
        """Calcula PSI entre dos distribuciones."""
        # Crear bins basados en la referencia
        min_val = reference.min()
        max_val = reference.max()
        if min_val == max_val:
            return 0.0

        bins = np.linspace(min_val, max_val, buckets + 1)
        ref_counts = np.histogram(reference, bins=bins)[0]
        cur_counts = np.histogram(current, bins=bins)[0]

        # Normalizar (evitar división por cero)
        ref_pct = (ref_counts + 1e-6) / len(reference)
        cur_pct = (cur_counts + 1e-6) / len(current)

        # PSI = sum((cur - ref) * ln(cur/ref))
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    monitored = [c for c in MONITORED_FEATURES if c in reference_df.columns and c in current_df.columns]
    drifted_features = []
    psi_scores = {}

    for col in monitored:
        psi = compute_psi(reference_df[col].dropna(), current_df[col].dropna())
        psi_scores[col] = psi
        if psi > drift_threshold:
            drifted_features.append(col)
            logger.info(f"  {col}: PSI={psi:.3f} > threshold={drift_threshold} -> DRIFT")
        else:
            logger.debug(f"  {col}: PSI={psi:.3f} <= threshold={drift_threshold} -> OK")

    overall_score = float(sum(psi_scores.values()) / len(psi_scores)) if psi_scores else 0.0
    drift_detected = len(drifted_features) > 0

    return {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "drift_score": overall_score,
        "psi_by_feature": psi_scores,
        "n_drifted_features": len(drifted_features),
        "n_monitored_features": len(monitored),
        "drift_threshold": drift_threshold,
        "report_path": None,
        "method": "manual_psi",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Detección de data drift con Evidently")
    parser.add_argument("--reference", default="data/processed/train.parquet")
    parser.add_argument("--current", default="data/processed/test.parquet")
    parser.add_argument("--output", default="artifacts/drift_report.html")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    ref_df = pd.read_parquet(args.reference)
    cur_df = pd.read_parquet(args.current)

    result = run_drift_report(ref_df, cur_df, args.output, args.threshold)
    print("\nResultado de detección de drift:")
    print(json.dumps({k: v for k, v in result.items() if k != "psi_by_feature"}, indent=2))
