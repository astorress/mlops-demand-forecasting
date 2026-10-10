# MLflow Models Serving

Guía de referencia para servir modelos directamente con el comando `mlflow models serve`, sin necesidad de código adicional.

---

## ¿Qué es MLflow Models Serving?

MLflow incluye un servidor HTTP integrado que expone cualquier modelo registrado como una API REST con un solo comando. El servidor utiliza `gunicorn` (Linux/Mac) o `waitress` (Windows) internamente y expone dos endpoints principales:

- `POST /invocations` — realiza predicciones
- `GET /ping` — health check

El servidor sabe cómo servir el modelo porque MLflow almacena el *flavor* (sklearn, xgboost, pyfunc, etc.) junto con el artefacto. No se requiere código personalizado.

**Cuándo usarlo:**
- Prototipado rápido y demos
- Entornos de staging donde no se necesita lógica de negocio adicional
- Evaluación del modelo antes de construir una API propia

---

## Comando para servir el modelo

### Por stage (Production / Staging)

```bash
mlflow models serve \
  --model-uri "models:/demand_forecaster/Production" \
  --host 0.0.0.0 \
  --port 8001 \
  --env-manager local
```

### Por alias (@champion)

```bash
mlflow models serve \
  --model-uri "models:/demand_forecaster@champion" \
  --host 0.0.0.0 \
  --port 8001 \
  --env-manager local
```

### Por run_id (útil para reproducibilidad exacta)

```bash
mlflow models serve \
  --model-uri "runs:/<run_id>/model" \
  --host 0.0.0.0 \
  --port 8001 \
  --env-manager local
```

### Por ruta local (para desarrollo)

```bash
mlflow models serve \
  --model-uri "mlruns/0/<run_id>/artifacts/model" \
  --host 127.0.0.1 \
  --port 8001 \
  --env-manager local
```

> `--env-manager local` usa el entorno activo en vez de crear uno nuevo (virtualenv/conda). Asegúrate de tener las dependencias instaladas.
>
> **En este proyecto** basta con `make serve-mlflow`: sirve `models:/demand_forecaster/Production` en http://localhost:8001 desde el contenedor `app`, que ya tiene todo instalado.

**Variables de entorno relevantes:**

```bash
export MLFLOW_TRACKING_URI=http://localhost:5050   # o "mlruns" para local
mlflow models serve --model-uri "models:/demand_forecaster/Production" --port 8001 --env-manager local
```

---

## Formato de requests

El endpoint `/invocations` acepta tres formatos de entrada:

### 1. `dataframe_records` (recomendado para registros individuales o pequeños batches)

Cada elemento del array es un diccionario `{feature: valor}`.

```json
{
  "dataframe_records": [
    {
      "temperature_c": 22.5,
      "hour": 14,
      "day_of_week": 2,
      "month": 6,
      "is_weekend": 0,
      "is_holiday": 0,
      "demand_lag_24h": 1180.0,
      "demand_lag_168h": 1150.0,
      "demand_rolling_mean_7d": 1090.0,
      "hour_sin": 0.866,
      "hour_cos": -0.5,
      "dow_sin": 0.782,
      "dow_cos": 0.623,
      "month_sin": 0.866,
      "month_cos": 0.5,
      "temp_x_hour": 315.0,
      "is_peak_hour": 0
    }
  ]
}
```

### 2. `dataframe_split` (más eficiente para batches grandes)

Separa columnas y datos para reducir la redundancia en el JSON.

```json
{
  "dataframe_split": {
    "columns": [
      "temperature_c", "hour", "day_of_week", "month",
      "is_weekend", "is_holiday",
      "demand_lag_24h", "demand_lag_168h", "demand_rolling_mean_7d",
      "hour_sin", "hour_cos", "dow_sin", "dow_cos",
      "month_sin", "month_cos", "temp_x_hour", "is_peak_hour"
    ],
    "data": [
      [22.5, 14, 2, 6, 0, 0, 1180.0, 1150.0, 1090.0, 0.866, -0.5, 0.782, 0.623, 0.866, 0.5, 315.0, 0],
      [18.0, 19, 4, 6, 1, 0, 1050.0, 1020.0, 1070.0, -0.259, -0.966, -0.975, 0.223, 0.866, 0.5, 342.0, 1]
    ]
  }
}
```

### 3. `instances` (formato TensorFlow Serving, menos común con sklearn)

```json
{
  "instances": [
    [22.5, 14, 2, 6, 0, 0, 1180.0, 1150.0, 1090.0, 0.866, -0.5, 0.782, 0.623, 0.866, 0.5, 315.0, 0]
  ]
}
```

**Formato de respuesta:**

```json
{
  "predictions": [0.7234]
}
```

---

## Ejemplos con curl

### Health check

```bash
curl -s http://localhost:8001/ping
# Respuesta esperada: "OK"
```

### Predicción individual (dataframe_records)

```bash
curl -s -X POST http://localhost:8001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [{
      "temperature_c": 22.5,
      "hour": 14,
      "day_of_week": 2,
      "month": 6,
      "is_weekend": 0,
      "is_holiday": 0,
      "demand_lag_24h": 1180.0,
      "demand_lag_168h": 1150.0,
      "demand_rolling_mean_7d": 1090.0,
      "hour_sin": 0.866,
      "hour_cos": -0.5,
      "dow_sin": 0.782,
      "dow_cos": 0.623,
      "month_sin": 0.866,
      "month_cos": 0.5,
      "temp_x_hour": 315.0,
      "is_peak_hour": 0
    }]
  }' | python -m json.tool
```

### Predicción batch (dataframe_split)

```bash
curl -s -X POST http://localhost:8001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": [
        "temperature_c","hour","day_of_week","month","is_weekend","is_holiday",
        "demand_lag_24h","demand_lag_168h","demand_rolling_mean_7d",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "month_sin","month_cos","temp_x_hour","is_peak_hour"
      ],
      "data": [
        [22.5, 14, 2, 6, 0, 0, 1180.0, 1150.0, 1090.0, 0.866, -0.5, 0.782, 0.623, 0.866, 0.5, 315.0, 0],
        [18.0, 19, 4, 6, 1, 0, 1050.0, 1020.0, 1070.0, -0.259, -0.966, -0.975, 0.223, 0.866, 0.5, 342.0, 1]
      ]
    }
  }' | python -m json.tool
```

### Con variable de entorno para tracking URI remoto

```bash
MLFLOW_TRACKING_URI=http://localhost:5050 \
curl -s -X POST http://localhost:8001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"temperature_c": 25.0, "hour": 20, ...}]}'
```

---

## Cómo servir por stage, run_id o alias

| Método | URI | Caso de uso |
|--------|-----|-------------|
| Por stage | `models:/demand_forecaster/Production` | Siempre sirve el modelo en producción |
| Por stage | `models:/demand_forecaster/Staging` | Pruebas de integración antes de promover |
| Por alias | `models:/demand_forecaster@champion` | Recomendado en MLflow >= 2.0 (stages deprecados) |
| Por versión | `models:/demand_forecaster/3` | Servir una versión específica exacta |
| Por run_id | `runs:/abc123def456/model` | Reproducibilidad absoluta, debugging |
| Por ruta | `mlruns/0/abc123/artifacts/model` | Desarrollo local sin registry |

**Cambiar el modelo servido sin reiniciar:**
MLflow Models Serving no soporta recarga en caliente. Para cambiar el modelo hay que reiniciar el servidor con el nuevo URI. Si necesitas recarga dinámica, usa `flask_app.py` con `ModelManager.load()`.

---

## Limitaciones vs Flask personalizado

| Característica | `mlflow models serve` | `flask_app.py` |
|---|---|---|
| Setup inicial | 1 comando | Requiere código |
| Validación de inputs | No incluida | Pydantic (tipada, con mensajes) |
| Feature engineering | No incluido | `build_features()` integrado |
| Endpoints personalizados | Solo `/invocations` y `/ping` | Ilimitados (`/health`, `/model-info`, etc.) |
| Recarga en caliente del modelo | No | Posible con `model_manager.load()` |
| Autenticación / rate limiting | No | Implementable con middleware Flask |
| Logging estructurado | Básico | Configurable por endpoint |
| Batch con validación | No (acepta cualquier JSON) | Limitado a 10,000 registros con error claro |
| Métricas de latencia | No | Incluidas en cada respuesta |
| Integración con monitoreo | No | Extensible (Prometheus, Datadog, etc.) |
| Producción en alta escala | No recomendado | Extensible con gunicorn/uvicorn |

**Recomendación:** usar `mlflow models serve` para demos y validación rápida. Para producción, usar `flask_app.py` o un servidor ASGI (FastAPI + uvicorn).

---

## Script de prueba

Guárdalo como `test_mlflow_serve.py` (no está incluido en el repositorio):

```python
#!/usr/bin/env python3
"""
test_mlflow_serve.py
Script para probar el servidor mlflow models serve.

Uso:
    # 1. Iniciar el servidor en otra terminal:
    mlflow models serve --model-uri "models:/demand_forecaster/Production" --port 8001 --env-manager local

    # 2. Ejecutar este script:
    python test_mlflow_serve.py
    python test_mlflow_serve.py --port 8001 --host localhost
"""

import argparse
import json
import sys
import time

import requests

# Features de ejemplo (con feature engineering ya aplicado)
SAMPLE_RECORD = {
    "temperature_c": 22.5,
    "hour": 14,
    "day_of_week": 2,
    "month": 6,
    "is_weekend": 0,
    "is_holiday": 0,
    "demand_lag_24h": 1180.0,
    "demand_lag_168h": 1150.0,
    "demand_rolling_mean_7d": 1090.0,
    "hour_sin": 0.8660254,
    "hour_cos": -0.5,
    "dow_sin": 0.7818315,
    "dow_cos": 0.6234898,
    "month_sin": 0.8660254,
    "month_cos": 0.5,
    "temp_x_hour": 315.0,
    "is_peak_hour": 0,
}


def test_ping(base_url: str) -> bool:
    """Verifica que el servidor está corriendo."""
    try:
        resp = requests.get(f"{base_url}/ping", timeout=5)
        if resp.status_code == 200:
            print(f"  [OK] /ping -> {resp.text.strip()}")
            return True
        else:
            print(f"  [FAIL] /ping -> HTTP {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  [FAIL] No se puede conectar a {base_url}. ¿Está corriendo el servidor?")
        return False


def test_single_prediction(base_url: str) -> bool:
    """Prueba predicción individual con dataframe_records."""
    payload = {"dataframe_records": [SAMPLE_RECORD]}
    try:
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/invocations",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        latency_ms = (time.time() - t0) * 1000

        if resp.status_code == 200:
            result = resp.json()
            predictions = result.get("predictions", [])
            print(f"  [OK] /invocations (dataframe_records) -> {predictions} ({latency_ms:.0f}ms)")
            return True
        else:
            print(f"  [FAIL] /invocations -> HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_batch_prediction(base_url: str) -> bool:
    """Prueba predicción batch con dataframe_split."""
    payload = {
        "dataframe_split": {
            "columns": list(SAMPLE_RECORD.keys()),
            "data": [
                list(SAMPLE_RECORD.values()),
                # Segundo registro: hora pico, fin de semana
                [18.0, 19, 5, 6, 1, 0, 1050.0, 1020.0, 1070.0,
                 -0.2588190, -0.9659258, -0.9749279, 0.2225209,
                 0.8660254, 0.5, 342.0, 1],
            ],
        }
    }
    try:
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/invocations",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        latency_ms = (time.time() - t0) * 1000

        if resp.status_code == 200:
            result = resp.json()
            predictions = result.get("predictions", [])
            print(f"  [OK] /invocations (dataframe_split, 2 registros) -> {predictions} ({latency_ms:.0f}ms)")
            return True
        else:
            print(f"  [FAIL] /invocations (batch) -> HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test mlflow models serve")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"\nProbando servidor en: {base_url}\n")

    results = []
    results.append(("ping", test_ping(base_url)))
    results.append(("single_prediction", test_single_prediction(base_url)))
    results.append(("batch_prediction", test_batch_prediction(base_url)))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\nResultado: {passed}/{total} tests pasaron")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
```
