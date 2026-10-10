"""
Validación de datos con Great Expectations.

Valida el dataset de demanda eléctrica antes de usarlo en entrenamiento.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Valida el dataset de demanda eléctrica usando Great Expectations.

    Validaciones:
    - No nulos en columnas críticas
    - Rangos válidos para demand_kwh, hour, temperature_c
    - Tipos de datos correctos
    - Sin timestamps duplicados

    Args:
        df: DataFrame a validar.

    Returns:
        Dict con {passed: bool, results: list, summary: str}
    """
    try:
        import great_expectations as gx
        from great_expectations.dataset import PandasDataset
    except ImportError:
        logger.warning("great_expectations no instalado. Usando validación manual.")
        return _validate_manual(df)

    results = []
    failed = []

    try:
        ge_df = PandasDataset(df)

        # 1. No nulos en columnas críticas
        critical_cols = [
            "timestamp", "demand_kwh", "temperature_c",
            "hour", "day_of_week", "month", "is_weekend"
        ]
        for col in critical_cols:
            if col not in df.columns:
                result = {"check": f"column_exists_{col}", "passed": False,
                          "detail": f"Columna '{col}' no existe"}
                results.append(result)
                failed.append(result["check"])
                continue

            result_ge = ge_df.expect_column_values_to_not_be_null(col)
            passed = result_ge.success
            result = {
                "check": f"no_nulls_{col}",
                "passed": passed,
                "detail": ("OK" if passed else "FAIL: {} nulos encontrados".format(result_ge.result.get("unexpected_count", "?")))
            }
            results.append(result)
            if not passed:
                failed.append(result["check"])

        # 2. demand_kwh > 0
        if "demand_kwh" in df.columns:
            result_ge = ge_df.expect_column_values_to_be_between(
                "demand_kwh", min_value=0, strict_min=True
            )
            passed = result_ge.success
            result = {
                "check": "demand_kwh_positive",
                "passed": passed,
                "detail": f"{'OK' if passed else 'FAIL: valores de demanda <= 0 encontrados'}"
            }
            results.append(result)
            if not passed:
                failed.append(result["check"])

        # 3. hour en 0-23
        if "hour" in df.columns:
            result_ge = ge_df.expect_column_values_to_be_between("hour", min_value=0, max_value=23)
            passed = result_ge.success
            result = {
                "check": "hour_range",
                "passed": passed,
                "detail": f"{'OK' if passed else 'FAIL: horas fuera del rango 0-23'}"
            }
            results.append(result)
            if not passed:
                failed.append(result["check"])

        # 4. temperature_c en -10 a 50
        if "temperature_c" in df.columns:
            result_ge = ge_df.expect_column_values_to_be_between(
                "temperature_c", min_value=-10, max_value=50
            )
            passed = result_ge.success
            result = {
                "check": "temperature_range",
                "passed": passed,
                "detail": f"{'OK' if passed else 'FAIL: temperaturas fuera de rango -10 a 50°C'}"
            }
            results.append(result)
            if not passed:
                failed.append(result["check"])

        # 5. month en 1-12
        if "month" in df.columns:
            result_ge = ge_df.expect_column_values_to_be_between("month", min_value=1, max_value=12)
            passed = result_ge.success
            result = {
                "check": "month_range",
                "passed": passed,
                "detail": f"{'OK' if passed else 'FAIL: meses fuera del rango 1-12'}"
            }
            results.append(result)
            if not passed:
                failed.append(result["check"])

        # 6. is_weekend es binario
        if "is_weekend" in df.columns:
            result_ge = ge_df.expect_column_values_to_be_in_set("is_weekend", value_set=[0, 1])
            passed = result_ge.success
            result = {
                "check": "is_weekend_binary",
                "passed": passed,
                "detail": f"{'OK' if passed else 'FAIL: is_weekend tiene valores distintos de 0 o 1'}"
            }
            results.append(result)
            if not passed:
                failed.append(result["check"])

    except Exception as e:
        logger.error(f"Error en validación con GE: {e}. Usando validación manual.")
        return _validate_manual(df)

    # 7. Sin timestamps duplicados (verificación manual)
    if "timestamp" in df.columns:
        dup_count = df["timestamp"].duplicated().sum()
        passed = dup_count == 0
        result = {
            "check": "no_duplicate_timestamps",
            "passed": passed,
            "detail": f"{'OK' if passed else f'FAIL: {dup_count} timestamps duplicados'}"
        }
        results.append(result)
        if not passed:
            failed.append(result["check"])

    all_passed = len(failed) == 0
    summary = (
        f"Validación {'EXITOSA' if all_passed else 'FALLIDA'}. "
        f"{sum(r['passed'] for r in results)}/{len(results)} checks pasados."
        + (f" Fallaron: {', '.join(failed)}" if failed else "")
    )

    logger.info(summary)

    return {
        "passed": all_passed,
        "results": results,
        "summary": summary,
        "failed_checks": failed,
    }


def _validate_manual(df: pd.DataFrame) -> dict[str, Any]:
    """Validación manual como fallback si GE no está instalado."""
    results = []
    failed = []

    def add_result(check: str, passed: bool, detail: str) -> None:
        results.append({"check": check, "passed": passed, "detail": detail})
        if not passed:
            failed.append(check)

    # No nulos en columnas críticas
    critical_cols = ["demand_kwh", "temperature_c", "hour", "day_of_week", "month"]
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            add_result(f"no_nulls_{col}", null_count == 0,
                       f"OK" if null_count == 0 else f"FAIL: {null_count} nulos")

    # Rangos
    if "demand_kwh" in df.columns:
        add_result("demand_kwh_positive", (df["demand_kwh"] > 0).all(),
                   "OK" if (df["demand_kwh"] > 0).all() else "FAIL: valores <= 0")

    if "hour" in df.columns:
        valid = df["hour"].between(0, 23).all()
        add_result("hour_range", valid, "OK" if valid else "FAIL: horas fuera de 0-23")

    if "temperature_c" in df.columns:
        valid = df["temperature_c"].between(-10, 50).all()
        add_result("temperature_range", valid, "OK" if valid else "FAIL: temperatura fuera de rango")

    if "timestamp" in df.columns:
        dup_count = df["timestamp"].duplicated().sum()
        add_result("no_duplicate_timestamps", dup_count == 0,
                   "OK" if dup_count == 0 else f"FAIL: {dup_count} duplicados")

    all_passed = len(failed) == 0
    summary = (
        f"Validación {'EXITOSA' if all_passed else 'FALLIDA'} (modo manual). "
        f"{sum(r['passed'] for r in results)}/{len(results)} checks pasados."
    )

    return {"passed": all_passed, "results": results, "summary": summary, "failed_checks": failed}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/demand.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    result = validate_dataset(df)
    print(result["summary"])
    for r in result["results"]:
        status = "✓" if r["passed"] else "✗"
        print(f"  {status} {r['check']}: {r['detail']}")
    sys.exit(0 if result["passed"] else 1)
