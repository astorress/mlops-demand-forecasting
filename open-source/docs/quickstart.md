# Quickstart — MLOps Demand Forecasting

Guía de inicio rápido para levantar el entorno completo y correr el primer experimento.
Todo corre en Docker: no necesitas instalar Python ni dependencias en tu máquina.

## Prerrequisitos

| Herramienta | Versión mínima | Verificar |
|-------------|---------------|-----------|
| Docker Desktop (o Docker Engine) | 24.0 | `docker --version` |
| Docker Compose v2 | 2.20 | `docker compose version` |
| Make | cualquiera | `make --version` |
| Git | cualquiera | `git --version` |

Recursos recomendados para Docker: 4 CPU y 6 GB de RAM.

> En macOS: instala [Docker Desktop](https://www.docker.com/products/docker-desktop/) (`make` viene con las Xcode Command Line Tools)
> En Ubuntu: `sudo apt install docker.io docker-compose-v2 make`

## Paso 1 — Obtener el código

```bash
git clone https://github.com/astorress/mlops-demand-forecasting.git
cd mlops-demand-forecasting/open-source
```

## Paso 2 — Levantar la infraestructura

```bash
make up
```

Esto:
1. Crea `.envs/config.env` desde `.envs/config.env.example` (con tu UID para que Airflow pueda escribir en `data/`)
2. Construye las imágenes `app` y `airflow` (la primera vez tarda ~3-5 minutos)
3. Levanta los contenedores y espera a que estén *healthy*:

| Contenedor | Servicio | Puerto en tu máquina |
|-----------|---------|--------|
| `mlops-postgres` | PostgreSQL (metadata de Airflow + backend de MLflow) | — (solo red interna) |
| `mlops-mlflow` | MLflow Server (tracking, registry y artefactos) | 5050 |
| `mlops-airflow-init` | Migra la base de datos y crea el usuario admin | — |
| `mlops-airflow-webserver` | Airflow UI | 8080 |
| `mlops-airflow-scheduler` | Airflow Scheduler | — |
| `mlops-jupyter` | JupyterLab | 8888 |

> Si quieres cambiar puertos o credenciales, edita `.envs/config.env` antes de `make up`.

## Paso 3 — Verificar que todo funciona

```bash
make status
```

| URL | Qué deberías ver | Credenciales |
|-----|-----------------|--------------|
| http://localhost:5050 | MLflow UI | — |
| http://localhost:8080 | Airflow UI con 3 DAGs | admin / admin |
| http://localhost:8888 | JupyterLab | — |

Si algún servicio no responde, ver los logs:
```bash
make logs
# O solo de un servicio:
docker compose logs mlflow-server
```

## Paso 4 — Flujo manual paso a paso

Cada comando corre dentro del contenedor `app` y registra en el servidor de MLflow:

```bash
make generate-data   # data/raw/demand.csv: ~26,136 registros horarios (2022-2024)
make validate        # validación con Great Expectations
make preprocess      # features + split temporal → data/processed/{train,val,test}.parquet
make train           # LightGBM con parámetros por defecto
make hpo             # Optuna, 30 trials (make hpo N_TRIALS=10 para algo más rápido)
make drift           # reporte de Evidently → artifacts/drift_report.html
```

`make train` imprime el Run ID. Para evaluarlo en test y compararlo con el champion:
```bash
make evaluate RUN_ID=<run_id>
```

## Paso 5 — Notebooks

Abre http://localhost:8888 y recorre `notebooks/` en orden (00 → 08). Los runs que crean
los notebooks aparecen en la UI de MLflow (http://localhost:5050).

## Paso 6 — Correr el pipeline completo desde Airflow

```bash
make pipeline
```

O desde la UI:
1. Ir a http://localhost:8080 (admin/admin)
2. Buscar el DAG `mlops_full_pipeline`
3. Activarlo (toggle) y dispararlo (botón ▶)
4. Ver el progreso en la pestaña **Graph**

El pipeline tarda ~1-2 minutos (incluye HPO con 20 trials).

> Al activar un DAG con schedule, Airflow programa también la ejecución del último intervalo.
> Las ejecuciones se encolan (`max_active_runs=1`), así que no se pisan.

## Paso 7 — Ver el modelo en el Registry

1. Ir a http://localhost:5050
2. Click en **"Models"**
3. Deberías ver `demand_forecaster` con una versión en `Production` y el alias `@champion`

## Paso 8 — Servir el modelo

**Con Flask** (API propia con validación Pydantic):
```bash
make serve-flask

curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature_c": 28.5,
    "hour": 19,
    "day_of_week": 1,
    "month": 7,
    "is_weekend": 0,
    "is_holiday": 0,
    "demand_lag_24h": 1450.0,
    "demand_lag_168h": 1420.0,
    "demand_rolling_mean_7d": 1300.0
  }'
```

**Con MLflow Serving** (sin código, en http://localhost:8001): `make serve-mlflow`.
Ver [`src/serving/mlflow_serve.md`](../src/serving/mlflow_serve.md).

## Paso 9 — Monitoreo de drift

El DAG `mlops_monitoring` corre todos los días a las 8AM cuando está activo. Compara los
últimos 7 días con el training set y, si detecta drift, dispara `mlops_retraining`.

Para dispararlo manualmente: Airflow UI → `mlops_monitoring` → activar y disparar.
Activa también `mlops_retraining` para que el reentrenamiento se ejecute.

Los reportes se guardan en `artifacts/drift_report_YYYYMMDD.html`.

---

## Comandos útiles

```bash
make help          # ver todos los comandos disponibles
make up            # levantar servicios
make down          # detener servicios (conserva datos y modelos)
make destroy       # detener y BORRAR volúmenes (bases de datos y artefactos)
make logs          # ver logs en tiempo real
make status        # estado de servicios y URLs
make test          # correr los tests
make clean         # limpiar datos generados y reportes
```

---

## Solución de problemas comunes

### "Puerto ya en uso"
Cambia el puerto en `.envs/config.env` (`MLFLOW_PORT`, `AIRFLOW_PORT`, `JUPYTER_PORT`, `FLASK_PORT`) y vuelve a correr `make up`.
MLflow usa el 5050 por defecto porque en macOS el 5000 lo ocupa AirPlay Receiver.

### "Airflow UI no carga"
```bash
docker compose logs airflow-webserver
# Verificar que airflow-init terminó correctamente
docker compose logs airflow-init | tail -20
```

### "El DAG no aparece en Airflow"
```bash
docker compose exec airflow-scheduler airflow dags list-import-errors
docker compose logs airflow-scheduler | grep ERROR
```

### "La API responde 503: Modelo no disponible"
No hay un modelo en Production todavía. Corre `make pipeline` y espera a que termine,
luego reinicia la API: `make stop-flask && make serve-flask`.

### Empezar de cero
```bash
make destroy && make clean && make up
```
