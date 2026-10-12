# CLAUDE.md — aws

## Contexto en el monorepo

Subproyecto AWS del monorepo `mlops-demand-forecasting` (ver `README.md` y `CLAUDE.md` de la raíz). Su par
`open-source/` resuelve el mismo problema con MLflow, Airflow, Great Expectations y Evidently en Docker Compose.

Sigue las convenciones de `mlops-customer-churn` (repo hermano): mismo layout de `src/`, mismos stacks, mismos patrones
(imagen BYOC con tag por hash, contrato de despliegue en SSM, drift con PSI en Python puro). Diferencia a propósito:
aquí la infraestructura base no es un subproyecto aparte sino dos stacks más (`serverless/common`, que se despliega
primero y publica `/mlops-common-infrastructure/*` en SSM, y `serverless/studio`, opcional).

## Qué hace

Pronóstico horario de demanda eléctrica (regresión, LightGBM) sobre la serie sintética del proyecto open source:
Lambda de ingesta → Glue crawler / curación (calendario, lags por tiempo, media móvil) / Data Quality → Athena →
SageMaker Pipeline (PrepareData con split temporal, Baseline, Train, Evaluate, quality gate por MAPE, Register) →
Model Registry con aprobación manual → endpoint (`ml.t2.medium`, blue/green, data capture, Lambda DataDrift diaria) +
API HTTP pública con throttling → Batch Transform programado → reentrenamiento por datos nuevos, schedule o alarma de
drift. Detalle y diagrama: `README.md` → Resumen.

## Estructura

```text
aws/
├── container/       # imagen del modelo: Dockerfile, entrypoint.sh (train/serve), requirements.txt, image-version.js, build-push.sh
├── serverless/      # stacks: common, studio, data, image, pipeline, endpoint, batch (+ shared/)
├── src/
│   ├── features/  models/  monitoring/  pipelines/training/  serving/   # paquetes del modelo (van en la imagen)
│   ├── datasets/                                                       # generador sintético (Python puro, lo usa la Lambda)
│   ├── jobs/glue/                                                      # script del job de Glue
│   ├── functions/{ingestion,registry,inference,monitoring}/            # handlers Lambda
│   ├── services/                                                       # un módulo por servicio de AWS
│   ├── utilities/                                                      # compartido por todas las Lambdas
│   └── scripts/                                                        # tooling local (python3 -m src.scripts.<nombre>)
├── tests/           # pytest, espejo de src/
├── .docker/  .envs/  package.json  pyproject.toml  pytest.ini  eslint.config.mjs  .dockerignore
```

## Contratos que no se rompen a la ligera

- `src/features/schema.py` → `FEATURES` (orden del CSV sin encabezado): lo usan endpoint, Batch Transform, `Predict`,
  `DataDrift` y el perfil de drift. Cambiarlo = nueva versión del modelo y redeploy de todo junto.
- `src/features/engineering.py` → `add_features` va serializada dentro del modelo (por nombre).
- Lista de festivos: `src/features/calendar.py` y `src/jobs/glue/curate_demand.py` (duplicada a propósito: el job de
  Glue es un script suelto). `src/features/curation.py` es el mirror en pandas del job: mantenerlos sincronizados.
- Columnas de `CuratedTable` (`serverless/data/glue.yml`) = salida del job de Glue = validación de `preprocess.py`.
- `timestamp` es string `yyyy-MM-dd HH:mm:ss` y palabra reservada en Athena: siempre entre comillas dobles en SQL.

## Desarrollo local

Devcontainer `MlopsDemandForecasting` (target `img-demandforecasting` de `.docker/Dockerfile`, con el mismo
`container/requirements.txt` de la imagen del modelo). Antes de dar un cambio por terminado: `npm run format`,
`npm run lint`, `npm test`, y `serverless print` / `serverless package` en cada `serverless/<stack>/` tocado.

## Despliegue

Orden y pasos: `README.md` → Despliegue. **Claude nunca ejecuta deploys, `remove`, `invoke`, `image:push`, arranques
del pipeline ni aprobaciones**: entrega el comando al dueño del proyecto. El endpoint cuesta por hora.
