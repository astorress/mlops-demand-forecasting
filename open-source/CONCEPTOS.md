# Conceptos de las herramientas open source

Qué es cada herramienta que usa este proyecto, los conceptos propios de cada una y cómo se usan aquí. Los conceptos generales de MLOps (drift, model registry, quality gate, split temporal…) están en el [`CONCEPTOS.md` de la raíz](../CONCEPTOS.md).

- [Conceptos de las herramientas open source](#conceptos-de-las-herramientas-open-source)
  - [Mapa de herramientas](#mapa-de-herramientas)
  - [Docker y Docker Compose](#docker-y-docker-compose)
  - [PostgreSQL](#postgresql)
  - [MLflow](#mlflow)
  - [Apache Airflow](#apache-airflow)
  - [Great Expectations](#great-expectations)
  - [pandas y Parquet](#pandas-y-parquet)
  - [Optuna](#optuna)
  - [LightGBM](#lightgbm)
  - [scikit-learn](#scikit-learn)
  - [Evidently AI](#evidently-ai)
  - [Flask y Pydantic](#flask-y-pydantic)
  - [JupyterLab](#jupyterlab)
  - [Make](#make)

---

## Mapa de herramientas

| Etapa                 | Herramienta                     | Versión | Dónde                                   |
| --------------------- | ------------------------------- | ------- | --------------------------------------- |
| Infraestructura local | Docker Compose                  | —       | `docker-compose.yml`, `docker/`         |
| Base de datos         | PostgreSQL                      | 15      | servicio `postgres`                     |
| Orquestación          | Apache Airflow                  | 2.10    | `airflow/dags/`                         |
| Validación de datos   | Great Expectations              | 0.18    | `src/data/validation.py`                |
| Feature engineering   | pandas + Parquet                | —       | `src/data/preprocessing.py`             |
| HPO                   | Optuna                          | 4.1     | `src/models/hpo.py`                     |
| Modelo                | LightGBM                        | 4.5     | `src/models/train.py`                   |
| Tracking y registry   | MLflow                          | 2.17    | servicio `mlflow-server`, `src/models/` |
| Serving               | Flask (+ `mlflow models serve`) | 3.0     | `src/serving/flask_app.py`              |
| Drift                 | Evidently AI                    | 0.4     | `src/monitoring/drift_detector.py`      |
| Exploración           | JupyterLab                      | 4.2     | `notebooks/`                            |

Las versiones están fijas en `requirements*.txt`: Great Expectations 1.x, Evidently 0.7+ y MLflow 3 cambiaron sus APIs.

## Docker y Docker Compose

- **Imagen:** plantilla inmutable con un sistema operativo mínimo, librerías y código. Se construye desde un `Dockerfile`. Aquí hay dos: `docker/app.Dockerfile` (MLflow, Jupyter, API y comandos) y `docker/airflow.Dockerfile` (Airflow con las librerías del pipeline).
- **Contenedor:** una instancia en ejecución de una imagen, aislada del resto del sistema.
- **Docker Compose:** describe en un solo archivo (`docker-compose.yml`) varios contenedores que trabajan juntos, sus puertos, variables y dependencias, y los levanta con un comando.
- **Servicio:** cada contenedor definido en el Compose (`postgres`, `mlflow-server`, `airflow-webserver`, `airflow-scheduler`, `jupyter`, `api`…).
- **Volumen:** almacenamiento persistente que sobrevive a que el contenedor se borre (`postgres-data`, `mlflow-artifacts`). `make destroy` los elimina.
- **Bind mount:** una carpeta de tu máquina montada dentro del contenedor (`./data`, `./src`, `./artifacts`): lo que el contenedor escribe aparece en tu disco y viceversa.
- **Healthcheck:** comando que Compose ejecuta periódicamente para saber si un servicio está listo; `depends_on: condition: service_healthy` espera a que lo esté antes de arrancar el siguiente.
- **Profiles:** servicios que solo se levantan si se piden (`api` con el perfil `serving`, `app` con `tools`).
- **Red interna:** los contenedores se encuentran por nombre de servicio (`http://mlflow-server:5000`) sin exponer puertos al exterior.
- **`--env-file`:** archivo con variables que Compose usa para completar `${VARIABLE}` en el `docker-compose.yml`. Aquí es `.envs/config.env`, y el `Makefile` lo pasa siempre.

## PostgreSQL

Base de datos relacional. Aquí no guarda datos del negocio: es el **almacenamiento interno** de las otras herramientas.

- **Metadata de Airflow** (base `airflow`): DAGs, ejecuciones, estado de cada tarea, XComs, usuarios.
- **Backend store de MLflow** (base `mlflow`): experimentos, runs, parámetros, métricas y el model registry. `docker/postgres/init-db.sql` crea esta segunda base.

## MLflow

Plataforma para gestionar el ciclo de vida de modelos. El proyecto usa tres de sus componentes.

**Tracking**

- **Tracking server:** servicio central (`mlflow-server`, puerto 5050 en tu máquina) al que los procesos envían lo que registran. La UI permite comparar runs.
- **Experiment:** agrupación de runs de un mismo problema (`demand_forecasting`).
- **Run:** una ejecución de entrenamiento. Guarda **parámetros** (`learning_rate`, `num_leaves`), **métricas** (`val_rmse`, `test_mape`), **tags** (tipo de modelo, DAG que lo lanzó) y **artefactos**.
- **Artefactos:** archivos asociados a un run: el modelo, el gráfico de importancia de features. Con `--serve-artifacts` el servidor los guarda en un volumen y los clientes los suben por HTTP, sin credenciales de almacenamiento.
- **Backend store vs. artifact store:** los metadatos (parámetros, métricas, registry) van en PostgreSQL; los archivos, en el artifact store.

**Models**

- **MLflow Model / flavor:** formato estándar para guardar un modelo con sus dependencias. El _flavor_ `lightgbm` guarda el modelo nativo; el _flavor_ `pyfunc` es una interfaz genérica `predict()` que cualquier herramienta de MLflow sabe usar.
- **Signature:** el esquema de entrada y salida del modelo (columnas y tipos), inferido de los datos de entrenamiento. MLflow lo usa para validar las peticiones.
- **Model URI:** forma de referirse a un modelo: `runs:/<run_id>/model`, `models:/demand_forecaster/Production` o `models:/demand_forecaster@champion`.

**Model Registry**

- **Registered model:** un nombre (`demand_forecaster`) bajo el cual se registran versiones.
- **Model version:** cada modelo registrado recibe un número incremental y apunta al run que lo produjo (linaje).
- **Stages:** etapas `None`, `Staging`, `Production`, `Archived`. Promover = mover una versión a `Production` y archivar la anterior. MLflow las está reemplazando por alias, por eso el proyecto usa ambos.
- **Alias:** etiqueta móvil que apunta a una versión (`@champion`). La API carga siempre `@champion`, así que cambiar el alias cambia el modelo servido sin tocar la API.

**Serving**

- **`mlflow models serve`:** levanta un servidor REST para cualquier modelo del registry, con el endpoint estándar `/invocations`. Es la alternativa "sin código" a la API Flask (`make serve-mlflow`).

## Apache Airflow

Orquestador de flujos de trabajo escritos en Python.

- **DAG:** el flujo completo, definido como código en `airflow/dags/`. Hay tres: `mlops_full_pipeline` (lunes 2 a. m.), `mlops_monitoring` (diario 8 a. m.) y `mlops_retraining` (sin calendario, solo se dispara).
- **Task:** un paso del DAG (`validate_data`, `train_model`…). Las dependencias se declaran con `>>`.
- **Operator:** la plantilla de una task:
  - `PythonOperator`: ejecuta una función de Python.
  - `BranchPythonOperator`: ejecuta una función que **elige qué rama seguir** (promover o no; reentrenar o no). Las tasks de la rama no elegida quedan _skipped_.
  - `TriggerDagRunOperator`: dispara otro DAG; así el monitoreo lanza el reentrenamiento.
  - `EmptyOperator`: task vacía que marca el final de una rama.
- **Scheduler:** el proceso que decide qué tasks están listas y las ejecuta. **Webserver:** la UI (puerto 8080).
- **Executor:** cómo se ejecutan las tasks. `LocalExecutor` las corre como procesos en la misma máquina del scheduler; en producción se usan `CeleryExecutor` o `KubernetesExecutor` para repartirlas.
- **Schedule:** cuándo corre un DAG, con sintaxis cron (`0 2 * * 1` = lunes a las 2:00). `catchup=False` evita que corra todas las fechas pasadas al activarlo.
- **DAG run / task instance:** una ejecución concreta del DAG y de cada task, con su estado (_success_, _failed_, _skipped_…).
- **XCom (cross-communication):** mecanismo para pasar valores pequeños entre tasks (el `run_id` del entrenamiento, si el modelo es mejor, el score de drift). No sirve para datos grandes: esos van a archivos.
- **`conf`:** parámetros que se pasan al disparar un DAG (`triggered_by: drift_detection`).
- **Hook:** cliente reutilizable para hablar con un sistema externo; `airflow/plugins/mlflow_hook.py` encapsula el acceso a MLflow.

## Great Expectations

Librería de validación de datos.

- **Expectation:** una afirmación verificable sobre los datos: `expect_column_values_to_not_be_null`, `expect_column_values_to_be_between`, `expect_column_values_to_be_in_set`.
- **Validation result:** cada expectation devuelve si se cumplió y cuántos valores la rompieron.
- **Gate:** si alguna expectation falla, la task `validate_data` falla y el DAG se detiene antes de entrenar.
- Aquí se usa la API clásica `PandasDataset` (versión 0.18). Si la librería no está instalada, el código cae a una validación manual equivalente.

## pandas y Parquet

- **pandas:** librería para manipular tablas (`DataFrame`) en memoria. Hace el feature engineering: encodings cíclicos, interacciones y el split temporal.
- **Parquet:** formato de archivo columnar, comprimido y tipado. Los splits `train/val/test` se guardan como `.parquet`: se leen mucho más rápido que un CSV y conservan los tipos (fechas, enteros).

## Optuna

Librería de optimización de hiperparámetros.

- **Study:** una búsqueda completa, con una dirección (`minimize` el RMSE de validación).
- **Trial:** un intento con una combinación de hiperparámetros. Cada trial queda registrado como un run de MLflow.
- **Search space:** los rangos que puede proponer cada hiperparámetro (`trial.suggest_float`, `trial.suggest_int`).
- **Sampler TPE (Tree-structured Parzen Estimator):** muestreo bayesiano: modela qué regiones del espacio dieron buenos resultados y propone más cerca de ellas.
- **Pruner (`MedianPruner`):** corta un trial si a mitad de camino va peor que la mediana de los anteriores, para no gastar tiempo en combinaciones malas.
- **Best params:** al final, el DAG pasa los mejores hiperparámetros a `train_model` por XCom.

## LightGBM

Implementación de **gradient boosting** sobre árboles de decisión, de Microsoft.

- **Boosting:** entrena árboles en secuencia; cada uno aprende a corregir el error residual de los anteriores.
- **Crecimiento por hoja (leaf-wise):** a diferencia de otros algoritmos, divide la hoja que más reduce el error, lo que lo hace rápido y preciso; `num_leaves` controla la complejidad.
- **Hiperparámetros principales:** `n_estimators` (número de árboles), `learning_rate` (cuánto aporta cada árbol), `num_leaves`, `min_child_samples`, `subsample`/`colsample_bytree` (submuestreo de filas/columnas), `reg_alpha`/`reg_lambda` (regularización).
- **Early stopping:** con `eval_set` y `early_stopping(50)` el entrenamiento se detiene si el RMSE de validación no mejora en 50 rondas.
- **Invariante a escalas:** como divide por umbrales, no necesita normalizar las features; por eso la API recibe los valores en kWh y °C tal cual.

## scikit-learn

Librería general de ML. Aquí se usa para las **métricas** (`mean_squared_error`, `mean_absolute_error`, `r2_score`) con las que se evalúa el modelo y se compara con el champion (por RMSE en test).

## Evidently AI

Librería para evaluar y monitorear datos y modelos.

- **Report:** documento (HTML o diccionario) con un conjunto de métricas sobre dos datasets: **referencia** (el split de entrenamiento) y **actual** (los últimos 7 días).
- **Preset:** conjunto predefinido de métricas. Se usan `DataDriftPreset` (drift por columna) y `DataQualityPreset` (nulos, rangos, valores únicos).
- **Prueba estadística por columna:** Evidently elige la prueba según el tipo y tamaño de la columna (Kolmogorov-Smirnov, Wasserstein, chi-cuadrado, Jensen-Shannon…) y marca cada columna como con o sin drift.
- **Dataset drift / share of drifted columns:** la proporción de columnas con drift. Es el **score** que usa el DAG de monitoreo: si supera el umbral (`DRIFT_THRESHOLD`, 0.2), dispara el reentrenamiento.
- **Column mapping:** indica qué columna es el target, cuáles son numéricas o categóricas y cuál es la fecha.

## Flask y Pydantic

- **Flask:** micro-framework web de Python. `src/serving/flask_app.py` expone `/health`, `/model-info`, `/predict` y `/predict-batch`, y carga el modelo `@champion` desde MLflow al arrancar.
- **Pydantic:** valida y tipa el cuerpo de las peticiones (horas entre 0 y 23, listas no vacías) y devuelve errores claros (HTTP 400) antes de llegar al modelo.
- **Mismo feature engineering que en entrenamiento:** la API agrega los encodings cíclicos con la misma lógica para evitar training/serving skew.

## JupyterLab

Entorno de notebooks en el navegador (puerto 8888). Los notebooks `00` a `08` recorren cada pieza del stack paso a paso contra los mismos servicios (MLflow, Airflow), así que lo que se ve en un notebook queda registrado igual que si lo hubiera hecho el pipeline.

## Make

`Makefile` agrupa los comandos largos de Docker Compose en objetivos cortos (`make up`, `make pipeline`, `make test`, `make serve-flask`). Cada objetivo documentado con `## nombre: descripción` aparece en `make help`.
