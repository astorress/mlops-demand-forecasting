# MLOps Demand Forecasting

Monorepo con **dos implementaciones del mismo proyecto MLOps**: pronóstico horario de demanda eléctrica con LightGBM, desde los datos hasta el serving, el monitoreo de drift y el reentrenamiento automático.

Los dos usan la misma serie sintética (3 años de demanda horaria con estacionalidad diaria, semanal y anual, festivos colombianos y temperatura), los mismos 17 features, el mismo modelo y el mismo split temporal. Así se pueden comparar las herramientas sin que cambie el problema.

## Proyectos

| Proyecto                                | Enfoque                                                                                                                                | Dónde corre                                | Documentación                                                           |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------- |
| [`open-source/`](open-source/README.md) | Herramientas open source autoalojadas: MLflow, Airflow, Great Expectations, Optuna, Evidently, Flask                                   | Tu máquina, con Docker Compose (`make up`) | [README](open-source/README.md) · [Conceptos](open-source/CONCEPTOS.md) |
| [`aws/`](aws/README.md)                 | Servicios administrados de AWS: S3, Glue, Athena, SageMaker Pipelines, Model Registry, endpoints, Batch Transform, Lambda, EventBridge | Tu cuenta de AWS, con Serverless Framework | [README](aws/README.md) · [Conceptos](aws/CONCEPTOS.md)                 |

- **Para aprender MLOps sin cuenta de nube:** empieza por [`open-source/README.md`](open-source/README.md) y su guía [`docs/quickstart.md`](open-source/docs/quickstart.md).
- **Para ver el mismo flujo en AWS:** [`aws/README.md`](aws/README.md) tiene la arquitectura, el despliegue, la operación y los costos. También se puede entrenar en local sin AWS (`npm run local:generate && npm run local:train`).
- **Si algún término no te suena** (data drift, model registry, training/serving skew, blue/green…): [`CONCEPTOS.md`](CONCEPTOS.md) explica el flujo MLOps y sus conceptos generales, y el `CONCEPTOS.md` de cada proyecto explica las herramientas y servicios que usa.

## Diagramas

La arquitectura de ambos proyectos está en [`docs/architecture.drawio`](docs/architecture.drawio), con una página por flujo (entrenamiento y serving/inferencia de cada proyecto). Para verlo en el navegador, abre [`docs/architecture.html`](docs/architecture.html).

## Estructura

```text
mlops-demand-forecasting/
├── CONCEPTOS.md                 # teoría general del flujo MLOps
├── docs/                        # diagramas de arquitectura (draw.io)
├── open-source/                 # stack open source (Docker Compose)
│   ├── airflow/dags/            #   DAGs: pipeline completo, monitoreo, reentrenamiento
│   ├── src/                     #   datos, modelos (MLflow + Optuna), serving (Flask), drift (Evidently)
│   ├── notebooks/               #   recorrido explicativo de cada pieza
│   └── CONCEPTOS.md             #   las herramientas open source que usa
└── aws/                         # stack AWS (Serverless Framework)
    ├── serverless/              #   stacks: common, studio, data, image, pipeline, endpoint, batch
    ├── src/                     #   Glue job, pasos del pipeline, serving, Lambdas, drift (PSI)
    ├── container/               #   imagen del modelo para SageMaker (train/serve)
    └── CONCEPTOS.md             #   los servicios de AWS que usa
```

## Comparación por etapa del ciclo de vida

| Etapa                    | `open-source/`                                  | `aws/`                                                                   |
| ------------------------ | ----------------------------------------------- | ------------------------------------------------------------------------ |
| Ingesta de datos         | Script que escribe un CSV local                 | Lambda → zona raw de S3 → EventBridge                                    |
| Catálogo / consultas     | Archivos Parquet locales                        | Glue Data Catalog (crawler + tabla declarada) + Athena                   |
| Feature engineering      | pandas                                          | Glue job (Spark) + transformación dentro del modelo                      |
| Validación de datos      | Great Expectations                              | Glue Data Quality (DQDL) + gate en el pipeline                           |
| Orquestación             | Airflow                                         | Glue workflow + EventBridge + SageMaker Pipelines                        |
| Tracking de experimentos | MLflow Tracking                                 | SageMaker Experiments                                                    |
| HPO                      | Optuna                                          | Hiperparámetros como parámetros del pipeline                             |
| Registro y promoción     | MLflow Registry, champion/challenger automático | SageMaker Model Registry, quality gate por MAPE + aprobación manual      |
| Serving online           | Flask / `mlflow models serve`                   | Endpoint de SageMaker (imagen propia, blue/green) + API Gateway + Lambda |
| Inferencia batch         | —                                               | Batch Transform → tabla `predictions` en Athena                          |
| Monitoreo de drift       | Evidently                                       | Data capture + Lambda (PSI) + alarma de CloudWatch                       |
| Reentrenamiento          | DAG de Airflow disparado por drift              | Reglas de EventBridge: datos nuevos, calendario o alarma de drift        |
| Notificaciones           | Logs de Airflow                                 | SNS (fallas, modelos pendientes, aprobaciones, alarmas)                  |
| Costo                    | Gratis (local)                                  | Por uso; el endpoint ~USD 41/mes mientras exista                         |

## Licencia

[MIT](LICENSE)
