# Manual de la UI de Apache Airflow

Este manual describe cómo usar la interfaz gráfica de Airflow para gestionar, monitorear y depurar pipelines MLOps.

---

## Sección 1 — Cómo Acceder

### Con el docker-compose del proyecto

1. Levantar los servicios:
   ```bash
   make up
   ```
2. Abrir: **http://localhost:8080**
3. Credenciales:
   - Usuario: `admin`
   - Password: `admin`

> El primer inicio puede tardar 2-3 minutos mientras `airflow-init` configura la base de datos.

### Diferencia entre modos

| Modo | Cuándo usar |
|------|-------------|
| **Docker-compose** (este proyecto) | Desarrollo y demo |
| **Airflow standalone** (`airflow standalone`) | Pruebas locales rápidas |
| **Kubernetes (Helm chart)** | Producción |
| **Amazon MWAA / Cloud Composer** | Producción managed |

---

## Sección 2 — DAGs View (Página Principal)

Al iniciar sesión ves la lista de todos los DAGs.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Airflow          DAGs   Browse▼  Admin▼  Docs▼           admin ▼   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  DAGs     ○ Auto Refresh    Search: [____________]  [Filter Tags ▼]  │
│                                                                       │
│ ┌──────────────────┬───────┬────────┬──────────┬─────────┬────────┐ │
│ │ DAG              │ Owner │ Runs   │ Schedule │Last Run │ State  │ │
│ ├──────────────────┼───────┼────────┼──────────┼─────────┼────────┤ │
│ │▶ mlops_full_pipe │mlops  │ ●●●○   │Mon 2AM   │10:30 AM │success │ │
│ │▶ mlops_monitoring│mlops  │●●●●●●●●│0 8 * * * │08:05 AM │success │ │
│ │▶ mlops_retrain   │mlops  │  ●     │ None     │  3d ago │success │ │
│ └──────────────────┴───────┴────────┴──────────┴─────────┴────────┘ │
│                                                                       │
│ Íconos de estado:  ● success  ● running  ● failed  ○ queued         │
│                    ● up_for_retry        ● skipped                   │
└──────────────────────────────────────────────────────────────────────┘
```

### Columnas de la tabla

| Columna | Descripción |
|---------|-------------|
| **Toggle (▶/⏸)** | Activa o pausa el DAG. Un DAG pausado no corre automáticamente |
| **DAG** | Nombre del DAG (el `dag_id` en el código) |
| **Owner** | El campo `owner` en `default_args` |
| **Runs** | Historial visual de las últimas ejecuciones (círculos coloreados) |
| **Schedule** | Expresión cron o intervalo |
| **Last Run** | Cuándo corrió por última vez |
| **Recent Tasks** | Estado de las tasks en la última ejecución |

### Pausar/Reanudar un DAG

- Click en el toggle **▶/⏸** a la izquierda del nombre del DAG
- Cuando está **azul**: activo (correrá según el schedule)
- Cuando está **gris**: pausado (no correrá automáticamente)

> **Nota**: Los DAGs nuevos se crean pausados por defecto (`dags_are_paused_at_creation=True`). Debes activarlos manualmente.

### Disparar un DAG manualmente

1. Click en el botón **▶** (play) en la columna de acciones
2. Opcional: agregar configuración JSON en el campo "Configuration JSON"
3. Click **"Trigger DAG"**

---

## Sección 3 — DAG Detail View

Al hacer click en el nombre de un DAG, entras a su vista detallada.

### Pestaña Graph

Visualización del grafo de dependencias del DAG.

```
┌──────────────────────────────────────────────────────────────────┐
│  mlops_full_pipeline  > Graph                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [generate_data]──▶[validate_data]──▶[preprocess_data]──▶[run_hpo]│
│      ✓ success          ✓ success       ✓ success      ⟳ running  │
│                                                              │     │
│                                                     [train_model] │
│                                                              │     │
│                                                   [evaluate_model]│
│                                                              │     │
│                                                   [branch_promote]│
│                                                    /          \    │
│                                          [promote_model] [skip_pr]│
│                                                    \          /    │
│                                              [notify_completion]  │
│                                                                   │
│  Leyenda: ✓=success  ⟳=running  ✗=failed  ○=queued  —=skipped   │
└──────────────────────────────────────────────────────────────────┘
```

**Colores de los nodos:**
- **Verde**: task completada exitosamente
- **Rojo**: task fallida
- **Amarillo/Dorado**: task en ejecución
- **Gris claro**: task en cola (queued)
- **Rosa/Salmón**: task con upstream fallido (upstream_failed)
- **Celeste**: task saltada (skipped, en branches)

**Cómo interactuar con el grafo:**
- Click en un nodo → popup con acciones (View Log, Clear, Mark Success)
- Scroll para hacer zoom
- Arrastrar para mover el grafo

### Pestaña Grid

Vista de cuadrícula que muestra el historial de todas las ejecuciones.

```
┌──────────────────────────────────────────────────────────────────┐
│  mlops_full_pipeline  > Grid                                      │
├──────────────────────────────────────────────────────────────────┤
│                    Run 1   Run 2   Run 3   Run 4   Run 5         │
│                    Jan 6   Jan 13  Jan 20  Jan 27  Feb 3         │
│ generate_data       ●       ●       ●       ●       ●            │
│ validate_data       ●       ●       ●       ●       ●            │
│ preprocess_data     ●       ●       ●       ●       ●            │
│ run_hpo             ●       ●       ✗       ●       ●            │
│ train_model         ●       ●       -       ●       ●            │
│ evaluate_model      ●       ●       -       ●       ●            │
│ branch_promote      ●       ●       -       ●       ●            │
│ promote_model       ●       -       -       ●       -            │
│ skip_promote        -       ●       -       -       ●            │
│ notify_completion   ●       ●       -       ●       ●            │
│                                                                   │
│ ● success  ✗ failed  - skipped  (vacío = no ejecutada)          │
└──────────────────────────────────────────────────────────────────┘
```

**Cómo leer el Grid:**
- Cada **fila** es una task del DAG
- Cada **columna** es un DAG run (ejecución)
- El color de cada celda indica el estado de esa (task, run)
- Click en una celda → detalle de esa ejecución específica

**Patrones de fallo a identificar:**
- La misma task falla en múltiples runs → problema sistemático en esa task
- Un run completo falla desde una task en adelante → error en esa task que no se recupera

### Pestaña Code

Muestra el código Python del DAG tal como Airflow lo cargó. Es **read-only** (no editable desde la UI).

Útil para verificar que Airflow está leyendo la versión correcta del archivo.

### Pestaña Audit Log

Historial de todas las acciones manuales sobre el DAG:
- Cuándo se disparó manualmente
- Cuándo se marcó una task como success/failed
- Cuándo se hizo clear de una task

---

## Sección 4 — Task Instance Detail

Al hacer click en una celda del Grid (o en un nodo del Graph y luego en "Log"):

```
┌──────────────────────────────────────────────────────────────────┐
│  Task Instance: run_hpo                                           │
│  DAG Run: scheduled__2024-01-20T02:00:00+00:00                   │
│  State: FAILED  │  Duration: 12m 34s                             │
├──────────────────────────────────────────────────────────────────┤
│  [Log] [Rendered Template] [XCom] [Details]                      │
├──────────────────────────────────────────────────────────────────┤
│  Log:                                                             │
│  ─────────────────────────────────────────────────────────────  │
│  [2024-01-20 02:05:23] INFO - Marking task as RUNNING            │
│  [2024-01-20 02:05:24] INFO - HPO iniciado: 20 trials            │
│  [2024-01-20 02:05:24] INFO - Importando mlflow...               │
│  [2024-01-20 02:05:25] ERROR - ModuleNotFoundError: optuna       │
│  [2024-01-20 02:17:57] ERROR - Task failed with exception        │
│  Traceback (most recent call last):                               │
│    File "src/models/hpo.py", line 10, in <module>                │
│      import optuna                                                │
│  ModuleNotFoundError: No module named 'optuna'                   │
└──────────────────────────────────────────────────────────────────┘
```

### Cómo leer los logs

**Estructura de un log de Airflow:**
```
[timestamp] NIVEL - Mensaje
[2024-01-20 02:05:24] INFO  - Ejecutando task...
[2024-01-20 02:05:25] ERROR - Error en la task
```

**Qué buscar cuando una task falla:**
1. Busca la línea con `ERROR` o `CRITICAL`
2. Busca `Traceback` para ver el stack trace completo
3. Mira el último mensaje antes del error para contexto

**Botones del log:**
- **"Download"**: descargar el log completo como texto
- **"Wrap"**: envolver líneas largas
- **"Raw Log"**: ver el log sin formato HTML

### Pestaña XCom

Muestra todos los valores que esta task instancia pusheó o pulleó vía XCom.

```
┌──────────────────────────────────────────────────────────────────┐
│  XCom                                                             │
│  ┌─────────────────┬────────────────────────────────────────┐   │
│  │ Key             │ Value                                  │   │
│  ├─────────────────┼────────────────────────────────────────┤   │
│  │ best_params     │ {"n_estimators": 320, "lr": 0.05...}  │   │
│  │ return_value    │ {"n_estimators": 320, "lr": 0.05...}  │   │
│  └─────────────────┴────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Sección 5 — Acciones de Recuperación

### Clear Task (Reejecutar una task)

**Cuándo usar**: una task falló pero el problema ya fue corregido (ej: se instaló una dependencia faltante).

1. Click en la celda de la task en el Grid (o en el nodo en Graph)
2. Click en **"Clear"**
3. Elegir opciones:
   - **"Past"**: limpiar también runs anteriores
   - **"Future"**: limpiar runs futuros
   - **"Upstream"**: limpiar tasks que dependen de esta hacia arriba
   - **"Downstream"**: limpiar también las tasks que vienen después
4. Click **"OK"**

> El Clear cambia el estado de la task a `None` y el scheduler la vuelve a encolar.

### Mark as Success / Mark as Failed

**Mark as Success**: fuerza el estado de la task a `success` sin ejecutarla.
- Usar cuando: la task ya se hizo fuera de Airflow y quieres que el pipeline continúe.

**Mark as Failed**: fuerza el estado a `failed`.
- Usar cuando: quieres que el pipeline se detenga en ese punto.

### Backfill

Para ejecutar un DAG para fechas pasadas que no se ejecutaron:

Desde la UI: click en **"DAG Runs"** → **"+ New DAG Run"** → especificar `logical_date`.

Desde la terminal:
```bash
airflow dags backfill mlops_full_pipeline --start-date 2024-01-01 --end-date 2024-01-31
```

---

## Sección 6 — Admin

### Variables

Las Variables son pares clave-valor globales accesibles desde todos los DAGs.

**Crear una variable:**
1. Admin → Variables → **"+"**
2. Key: `mlflow_model_name`
3. Value: `demand_forecaster`
4. Click **"Save"**

**Usar en un DAG:**
```python
from airflow.models import Variable

model_name = Variable.get("mlflow_model_name", default_var="demand_forecaster")
```

### Connections

Las Connections almacenan credenciales de manera segura para bases de datos, APIs, etc.

**Crear una connection para MLflow:**
1. Admin → Connections → **"+"**
2. Connection ID: `mlflow_default`
3. Connection Type: `HTTP`
4. Host: `mlflow-server`
5. Port: `5000`
6. Click **"Save"**

**Usar en un DAG con MLflowHook:**
```python
from plugins.mlflow_hook import MLflowHook

hook = MLflowHook(mlflow_conn_id="mlflow_default")
tracking_uri = hook.get_conn()
```

### XComs (admin global)

Admin → XComs muestra **todos** los XComs del sistema (de todos los DAGs y runs).

Útil para:
- Depurar por qué un XCom no está llegando a la task correcta
- Ver qué valores se están pasando entre tasks
- Limpiar XComs viejos para liberar espacio en la base de datos

---

## Sección 7 — Desde Notebooks y Scripts

### Simular la ejecución de un DAG sin Airflow

```python
# Llamar las funciones del DAG directamente (para testing)
import sys
sys.path.insert(0, "/ruta/al/proyecto")

from airflow.dags.dag_full_pipeline import (
    task_generate_data,
    task_validate_data,
    task_preprocess_data,
)

# Crear un mock de contexto
class MockContext:
    class ti:
        @staticmethod
        def xcom_push(key, value): print(f"XCom: {key}={value}")
        @staticmethod
        def xcom_pull(key, task_ids=None): return None
    ds = "2024-01-15"

ctx = MockContext()
task_generate_data(**ctx.__dict__)
task_validate_data(**ctx.__dict__)
```

### Disparar un DAG via REST API

```python
import requests

# Disparar el pipeline
response = requests.post(
    "http://localhost:8080/api/v1/dags/mlops_full_pipeline/dagRuns",
    auth=("admin", "admin"),
    headers={"Content-Type": "application/json"},
    json={
        "conf": {
            "triggered_by": "notebook",
            "note": "Test manual desde notebook"
        }
    },
)
print(response.status_code, response.json())

# Ver el estado del DAG run
dag_run_id = response.json()["dag_run_id"]
status = requests.get(
    f"http://localhost:8080/api/v1/dags/mlops_full_pipeline/dagRuns/{dag_run_id}",
    auth=("admin", "admin"),
).json()
print(f"Estado: {status['state']}")
```
