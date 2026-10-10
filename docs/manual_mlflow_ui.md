# Manual de la UI de MLflow

Este manual describe cómo usar la interfaz gráfica de MLflow para gestionar experimentos, comparar modelos y administrar el Model Registry.

---

## Sección 1 — Cómo Acceder

### Opción A: Modo local (notebook o línea de comandos)

1. Abrir una terminal en la carpeta del proyecto
2. Ejecutar:
   ```bash
   mlflow ui
   ```
3. Abrir el navegador en: **http://localhost:5000**

> Si el puerto 5000 está ocupado, usar `mlflow ui --port 5001`

### Opción B: Desde el docker-compose del proyecto

1. Levantar los servicios:
   ```bash
   make up
   ```
2. Abrir: **http://localhost:5050** (se publica en el 5050 porque en macOS el 5000 lo ocupa AirPlay)

El servidor MLflow del docker-compose usa:
- **Backend store**: PostgreSQL (`mlflow` database)
- **Artifact store**: volumen Docker `mlflow-artifacts`, servido por el propio MLflow (`--serve-artifacts`)

### Opción C: Desde Google Colab

En Colab no se puede abrir localhost. Opciones:

**Opción 1 — ngrok** (túnel público):
```python
!pip install pyngrok -q
from pyngrok import ngrok
import subprocess

subprocess.Popen(["mlflow", "ui", "--port", "5000"])
public_url = ngrok.connect(5000)
print(f"MLflow UI: {public_url}")
```

**Opción 2 — Servidor remoto**:
```python
import mlflow
mlflow.set_tracking_uri("http://TU_SERVIDOR:5000")
# Todos los runs se guardan en el servidor remoto
```

---

## Sección 2 — Experiments View

La vista principal de MLflow muestra todos los experimentos y sus runs.

```
┌─────────────────────────────────────────────────────────────────────┐
│  MLflow                                         Experiments  Models  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Experiments                                  + Create Experiment    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ □ Default (1234)                                               │ │
│  │ □ demand_forecasting (5678)                             ●      │ │
│  │ □ demand_forecasting_hpo (9012)                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Runs in: demand_forecasting                    □ Compare  ↓ CSV   │
│                                                                      │
│  Columns: Name | Date | Duration | Source | Params | Metrics        │
│  ┌─────────────┬──────────┬──────┬────────┬──────────┬───────────┐ │
│  │ Name        │ Created  │ Dur. │ n_est. │ val_rmse │ val_r2    │ │
│  ├─────────────┼──────────┼──────┼────────┼──────────┼───────────┤ │
│  │ best_model  │ 2h ago   │ 3m12s│ 500    │ 42.31    │ 0.94      │ │
│  │ trial_15    │ 3h ago   │ 1m45s│ 320    │ 45.22    │ 0.92      │ │
│  │ config_2    │ 5h ago   │ 2m30s│ 300    │ 48.10    │ 0.91      │ │
│  └─────────────┴──────────┴──────┴────────┴──────────┴───────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Qué es un Experiment

Un experimento es un **grupo de runs relacionados**. Úsalos para organizar tus pruebas:
- Un experimento por proyecto: `demand_forecasting`
- Un experimento por etapa: `demand_forecasting_hpo`
- Un experimento por versión de dataset: `demand_forecasting_v2`

### Crear un experimento

```python
# Desde Python:
mlflow.create_experiment("mi_experimento")
mlflow.set_experiment("mi_experimento")

# O directamente al crear un run:
with mlflow.start_run(experiment_id=mlflow.create_experiment("nuevo")):
    pass
```

Desde la UI: click en **"+ Create Experiment"** (esquina superior derecha).

### Columnas de la lista de runs

| Columna | Descripción |
|---------|-------------|
| **Name** | Nombre del run (auto-generado o definido con `run_name=`) |
| **Date** | Cuándo se creó el run |
| **Duration** | Tiempo de ejecución |
| **Source** | Qué script o notebook creó el run |
| **Version** | Git commit (si el proyecto usa git) |
| **Metrics** | Métricas finales logueadas (personalizables) |

### Filtrar y buscar runs

**Con la barra de búsqueda** (sintaxis SQL-like):
```
# Solo runs con val_rmse menor a 50:
metrics.val_rmse < 50

# Runs con learning_rate específico:
params.learning_rate = "0.05"

# Combinaciones:
metrics.val_rmse < 50 AND params.n_estimators > 200

# Por tag:
tags.model_type = "LightGBM"
```

**Ordenar**: click en el encabezado de cualquier columna de métrica.

### Comparar runs

1. Seleccionar 2+ runs con el checkbox a la izquierda
2. Click en el botón **"Compare"**
3. Verás:

```
┌──────────────────────────────────────────────────────┐
│  Compare Runs                                         │
│                                                      │
│  Parameters:                                         │
│  ┌──────────────┬──────────┬──────────┬──────────┐  │
│  │ Parameter    │ Run A    │ Run B    │ Run C    │  │
│  ├──────────────┼──────────┼──────────┼──────────┤  │
│  │ n_estimators │ 200      │ 500      │ 300      │  │
│  │ learning_rate│ 0.1      │ 0.05     │ 0.01     │  │
│  └──────────────┴──────────┴──────────┴──────────┘  │
│                                                      │
│  Metrics:                                            │
│  ┌──────────────┬──────────┬──────────┬──────────┐  │
│  │ Metric       │ Run A    │ Run B    │ Run C    │  │
│  ├──────────────┼──────────┼──────────┼──────────┤  │
│  │ val_rmse     │ 55.2     │ 42.3     │ 48.1     │  │
│  │ val_r2       │ 0.89     │ 0.94     │ 0.92     │  │
│  └──────────────┴──────────┴──────────┴──────────┘  │
│                                                      │
│  [Parallel Coordinates Plot]  [Scatter Plot]         │
└──────────────────────────────────────────────────────┘
```

---

## Sección 3 — Run Detail View

Al hacer click en un run individual, ves todos sus detalles.

```
┌──────────────────────────────────────────────────────────────────┐
│  Run: best_model_from_hpo                                         │
│  Status: FINISHED  │  Started: 2024-01-15 10:30  │  Duration: 3m │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Parameters              Metrics              Tags               │
│  ─────────────           ────────────         ────────────────   │
│  n_estimators: 500       val_rmse: 42.31      model_type: LGB    │
│  learning_rate: 0.05     val_r2: 0.94         origin: hpo        │
│  num_leaves: 63          train_rmse: 38.22    target: demand_kwh │
│  subsample: 0.8          test_rmse: 44.10                        │
│                                                                   │
│  Artifacts                                                        │
│  ─────────────────────────────────────────────────────────────   │
│  📁 model/                                                        │
│  │  ├── MLmodel              (metadatos del modelo)              │
│  │  ├── model.lgb            (modelo serializado)                │
│  │  └── conda.yaml           (dependencias)                      │
│  📁 plots/                                                        │
│     └── feature_importance.png                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Ver métricas en el tiempo

Si logueaste una métrica en múltiples steps:
```python
for epoch in range(100):
    mlflow.log_metric("loss", loss_value, step=epoch)
```

MLflow muestra una gráfica interactiva de la métrica vs step.

### Descargar artefactos

- Click en cualquier archivo del panel de artefactos
- O desde Python:
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.download_artifacts(run_id, "plots/feature_importance.png", "/tmp/")
```

---

## Sección 4 — Model Registry

El Model Registry es el **catálogo central de modelos listos para producción**.

### Cómo acceder

Click en la pestaña **"Models"** en la barra de navegación superior.

```
┌──────────────────────────────────────────────────────────────────┐
│  MLflow                                       Experiments  Models │
│                                                            ↑ AQUÍ│
├──────────────────────────────────────────────────────────────────┤
│  Registered Models                                                │
│                                                                   │
│  ┌──────────────────────┬──────────┬──────────┬──────────────┐  │
│  │ Name                 │ Versions │  Latest  │ Last Modified│  │
│  ├──────────────────────┼──────────┼──────────┼──────────────┤  │
│  │ demand_forecaster    │    5     │    v5    │ 2h ago       │  │
│  │ demand_forecaster_v2 │    2     │    v2    │ 1d ago       │  │
│  └──────────────────────┴──────────┴──────────┴──────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Model Version Detail

Al hacer click en un modelo y luego en una versión:

```
┌──────────────────────────────────────────────────────────────────┐
│  demand_forecaster  >  Version 5                                  │
│                                                                   │
│  Stage: Production          [Transition to: ▼]                   │
│  Aliases: @champion                                               │
│  Run ID: 3f4a2b1c...        [View Run]                           │
│  Created: 2024-01-15                                              │
│                                                                   │
│  Description:                                                     │
│  Modelo LightGBM para predicción de demanda eléctrica horaria.   │
│                                                                   │
│  Tags:                                                            │
│  training_dataset: synthetic_2022_2024                           │
│  rmse_test: 42.31                                                 │
│                                                                   │
│  Schema:                                                          │
│  Inputs:  hour (long), temperature_c (double), is_peak (long)... │
│  Outputs: prediction (double)                                     │
└──────────────────────────────────────────────────────────────────┘
```

### Transicionar stages desde la UI

1. En el detalle de una versión, click en **"Transition to:"**
2. Seleccionar el stage destino: `Staging`, `Production`, `Archived`
3. Confirmar en el dialog

### Asignar aliases desde la UI

1. En el detalle de una versión, click en **"Add alias"**
2. Escribir el nombre del alias (ej: `champion`, `challenger`)
3. Click **"Add"**

### Activity Log

En la página del modelo (no de la versión), abajo encontrarás un historial de todas las acciones:
- Cuándo se crearon versiones
- Cuándo se cambiaron stages
- Quién hizo cada cambio

---

## Sección 5 — Desde Notebooks y Scripts

### Modo local vs servidor remoto

```python
# Modo local (guarda en ./mlruns/)
mlflow.set_tracking_uri("mlruns")

# Servidor del docker-compose, desde tu máquina
mlflow.set_tracking_uri("http://localhost:5050")

# Servidor remoto (desde dentro de Docker)
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

### Buscar runs programáticamente

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Buscar todos los runs de un experimento, ordenados por métrica
runs = mlflow.search_runs(
    experiment_names=["demand_forecasting"],
    filter_string="metrics.val_rmse < 50",
    order_by=["metrics.val_rmse ASC"],
    max_results=10,
)

# Obtener el mejor run
best_run_id = runs.iloc[0]["run_id"]
best_rmse = runs.iloc[0]["metrics.val_rmse"]
print(f"Mejor run: {best_run_id} con RMSE={best_rmse:.4f}")
```

### Cargar modelo por stage o alias

```python
# Por stage (clásico)
model = mlflow.pyfunc.load_model("models:/demand_forecaster/Production")

# Por alias (recomendado en MLflow >= 2.8)
model = mlflow.pyfunc.load_model("models:/demand_forecaster@champion")

# Por versión específica
model = mlflow.pyfunc.load_model("models:/demand_forecaster/5")

# Por run_id
model = mlflow.pyfunc.load_model("runs:/3f4a2b1c.../model")
```

### Ver el mismo experimento en la UI y en Python

Ambas interfaces ven los mismos datos porque usan el mismo tracking URI:

```python
mlflow.set_tracking_uri("http://localhost:5050")

# Este run aparecerá en la UI inmediatamente
with mlflow.start_run():
    mlflow.log_metric("mi_metrica", 0.95)
```
