# Conceptos de los servicios de AWS

Qué es cada servicio de AWS que usa este proyecto, sus conceptos propios y para qué se usa aquí. Los conceptos generales de MLOps (drift, model registry, quality gate, split temporal, blue/green…) están en el [`CONCEPTOS.md` de la raíz](../CONCEPTOS.md).

- [Conceptos de los servicios de AWS](#conceptos-de-los-servicios-de-aws)
  - [Mapa de servicios](#mapa-de-servicios)
  - [Infraestructura como código](#infraestructura-como-código)
    - [CloudFormation](#cloudformation)
    - [Serverless Framework](#serverless-framework)
  - [Identidad y configuración](#identidad-y-configuración)
    - [IAM](#iam)
    - [SSM Parameter Store](#ssm-parameter-store)
  - [Datos](#datos)
    - [Amazon S3](#amazon-s3)
    - [AWS Glue](#aws-glue)
    - [Amazon Athena](#amazon-athena)
  - [Eventos, cómputo y avisos](#eventos-cómputo-y-avisos)
    - [Amazon EventBridge](#amazon-eventbridge)
    - [AWS Lambda](#aws-lambda)
    - [Amazon API Gateway](#amazon-api-gateway)
    - [Amazon SNS](#amazon-sns)
    - [Amazon CloudWatch](#amazon-cloudwatch)
  - [Contenedores](#contenedores)
    - [Amazon ECR](#amazon-ecr)
  - [Amazon SageMaker AI](#amazon-sagemaker-ai)
    - [Imagen propia (BYOC)](#imagen-propia-byoc)
    - [Processing y Training jobs](#processing-y-training-jobs)
    - [SageMaker Pipelines](#sagemaker-pipelines)
    - [SageMaker Experiments](#sagemaker-experiments)
    - [Model Registry](#model-registry)
    - [Hosting: Model, EndpointConfig y Endpoint](#hosting-model-endpointconfig-y-endpoint)
    - [Batch Transform](#batch-transform)
    - [SageMaker Studio](#sagemaker-studio)
    - [Model Monitor (no se usa)](#model-monitor-no-se-usa)
  - [Amazon VPC](#amazon-vpc)
  - [Patrones del proyecto](#patrones-del-proyecto)

---

## Mapa de servicios

| Etapa                       | Servicio                                                  | Stack                                   |
| --------------------------- | --------------------------------------------------------- | --------------------------------------- |
| Infraestructura como código | Serverless Framework → CloudFormation                     | todos                                   |
| Almacenamiento              | S3 (data lake, artefactos, resultados Athena)             | `common`                                |
| Catálogo y ETL              | Glue (Data Catalog, crawler, job, workflow, Data Quality) | `data`                                  |
| Consultas SQL               | Athena                                                    | `common`, `data`, `pipeline`            |
| Eventos y calendarios       | EventBridge                                               | `data`, `pipeline`, `endpoint`, `batch` |
| Funciones                   | Lambda                                                    | `data`, `pipeline`, `endpoint`, `batch` |
| Imagen del modelo           | ECR                                                       | `image`                                 |
| Entrenamiento y registro    | SageMaker Pipelines, Experiments, Model Registry          | `pipeline`                              |
| Serving                     | SageMaker Endpoint, API Gateway, Batch Transform          | `endpoint`, `batch`                     |
| Monitoreo y avisos          | CloudWatch, SNS                                           | `common`, `endpoint`                    |
| Configuración entre stacks  | SSM Parameter Store                                       | todos                                   |
| Permisos                    | IAM                                                       | todos                                   |
| Interfaz                    | SageMaker Studio + VPC                                    | `studio` (opcional)                     |

## Infraestructura como código

### CloudFormation

Servicio de AWS que crea y actualiza recursos a partir de una plantilla (YAML/JSON).

- **Stack:** el conjunto de recursos creados desde una plantilla, que se gestionan como una unidad: se crean, actualizan y borran juntos. Aquí hay un stack por carpeta de `serverless/` (`mlops-common-infrastructure`, `mlops-demand-forecasting-data`…).
- **Recurso lógico vs. físico:** el nombre en la plantilla (`CurateJob`) y el recurso real que se crea en la cuenta.
- **Funciones intrínsecas:** `!Ref` (el nombre o id de otro recurso), `!GetAtt` (un atributo, p. ej. un ARN), `!Sub` (interpolar texto).
- **DeletionPolicy: Retain:** el recurso no se borra al borrar el stack (el repositorio ECR, porque los modelos dependen de sus imágenes).
- **Reemplazo vs. actualización:** algunos cambios actualizan el recurso en sitio; otros obligan a crear uno nuevo. El modelo y la configuración del endpoint no tienen nombre fijo a propósito, para que cada versión aprobada cree recursos nuevos.

### Serverless Framework

Herramienta (v4) que genera plantillas de CloudFormation a partir de un `serverless.yml` más corto, y empaqueta y despliega las Lambdas.

- **Service:** un `serverless.yml` = un stack. `stackName: ${self:custom.app}-${self:service}` les da nombres como `mlops-demand-forecasting-pipeline`.
- **Stage:** ambiente (`dev`, `test`, `prod`). Se elige con `--stage`; aquí cada stage va en una cuenta de AWS distinta.
- **Variables:** se resuelven al desplegar:
  - `${self:custom.x}`: otro valor del mismo archivo.
  - `${env:X}`: variable de entorno (`.envs/config.env`).
  - `${ssm:/ruta}`: un parámetro de SSM, normalmente publicado por otro stack.
  - `${file(archivo.js)}`: el resultado de un script de Node. Así se construyen la definición del pipeline (`pipeline-definition.js`), el tag de la imagen (`image-version.js`) y la guarda que impide desplegar el endpoint sin modelo aprobado (`approved-model.js`).
- **`functions` y `resources`:** las Lambdas (con sus eventos) y cualquier otro recurso de CloudFormation.
- **Plugins:** extienden el framework. Aquí: `serverless-deployment-bucket` (bucket de despliegue con nombre propio), `@serverless/safeguards-plugin` (reglas: solo `us-east-1`, solo `dev|test|prod`, sin IAM con comodines…) y `serverless-s3-sync` (sube el script de Glue al desplegar).
- **`sls print` / `sls package` / `sls deploy` / `sls invoke` / `sls remove`:** ver la configuración resuelta, empaquetar sin desplegar, desplegar, ejecutar una función y borrar el stack.

## Identidad y configuración

### IAM

Servicio de identidades y permisos.

- **Rol:** identidad que asume un servicio (una Lambda, un job de Glue, SageMaker) para actuar. Cada pieza tiene su propio rol.
- **Trust policy:** quién puede asumir el rol (`lambda.amazonaws.com`, `sagemaker.amazonaws.com`…).
- **Policy:** lista de permisos (`Action` sobre `Resource`). Se acotan al mínimo: esta Lambda solo puede invocar este endpoint; este job solo escribe en este prefijo.
- **`iam:PassRole`:** permiso para entregarle un rol a otro servicio (la Lambda de batch le pasa el rol de SageMaker al transform job).
- **Managed policy:** política predefinida por AWS (`AmazonSageMakerFullAccess`, `AWSLambdaBasicExecutionRole`).

### SSM Parameter Store

Almacén de parámetros de configuración (parte de AWS Systems Manager).

- **Parámetro:** un par nombre/valor jerárquico (`/mlops-common-infrastructure/DataLakeBucketName`).
- **Comunicación entre stacks:** cada stack publica lo que otros necesitan (nombres de buckets, de tablas, del rol de SageMaker) y los demás lo leen con `${ssm:...}`. Se prefiere a los _exports_ de CloudFormation porque no amarra los stacks entre sí.
- **Contrato de despliegue:** `/mlops-demand-forecasting-pipeline/ApprovedModelPackageArn` y sus baselines. Los escribe la Lambda `OnModelApproved` al aprobar un modelo y los leen el endpoint y el batch.

## Datos

### Amazon S3

Almacenamiento de objetos.

- **Bucket:** contenedor de objetos con nombre único global. Hay tres: data lake, artefactos de ML y resultados de Athena.
- **Objeto y key:** un archivo y su ruta dentro del bucket. Las "carpetas" son solo **prefijos** de la key (`raw/demand_forecasting/ingestion_date=2026-10-11/demand.csv`).
- **Particiones estilo Hive:** prefijos `columna=valor` (`ingestion_date=…`, `run_date=…`) que Glue, Athena y Spark entienden como columnas.
- **Versionado:** cada sobrescritura conserva la versión anterior; la **lifecycle policy** borra las versiones viejas tras N días.
- **Notificaciones a EventBridge:** el bucket emite un evento por cada objeto creado; así la llegada de un archivo raw arranca la ingesta.
- **Cifrado y bloqueo de acceso público:** todos los buckets cifran en reposo (SSE-S3) y bloquean el acceso público.

### AWS Glue

Servicio de integración de datos (ETL) y catálogo.

- **Data Catalog:** metastore central con **bases de datos** y **tablas** (esquema, formato y ubicación en S3). Athena, Spark y SageMaker lo usan para saber cómo leer los datos.
- **Crawler:** recorre una ruta de S3, infiere el esquema y crea o actualiza una tabla (`raw_demand_forecasting`). Útil para explorar datos cuyo esquema no se controla.
- **Tabla declarada:** la tabla curada (`demand`) se define en la plantilla, con columnas y tipos explícitos: es el contrato de datos.
- **Partition projection:** Athena calcula las particiones a partir de un patrón (fechas desde 2024 hasta hoy) en lugar de registrarlas una a una; no hace falta un crawler ni `MSCK REPAIR` para ver particiones nuevas.
- **Job (ETL):** script que corre en un clúster administrado de **Apache Spark** (Glue 5.0, workers `G.1X`). `curate_demand.py` lee el CSV raw, calcula calendario, lags y media móvil, y escribe Parquet en la zona curada.
- **DynamicFrame:** la estructura de datos de Glue sobre los DataFrames de Spark; se usa para evaluar la calidad de datos.
- **Glue Data Quality y DQDL:** reglas declarativas (_Data Quality Definition Language_) como `IsComplete "timestamp"`, `IsUnique "timestamp"`, `ColumnValues "demand_kwh" > 0`. El job las evalúa y falla si alguna no se cumple; el mismo ruleset queda asociado a la tabla para correrlo desde la consola.
- **Workflow y triggers:** orquestación propia de Glue. Un trigger **EVENT** (activado por EventBridge, agrupando eventos durante 120 s) arranca el crawler, y un trigger **CONDITIONAL** arranca el job cuando el crawler termina bien.

### Amazon Athena

Motor SQL serverless (Trino) que consulta directamente archivos en S3 usando el Data Catalog. Se paga por datos escaneados.

- **Workgroup:** agrupa consultas con su configuración: dónde se guardan los resultados, límite de bytes escaneados por consulta, métricas. Hay uno compartido (`mlops`) y otro para el pipeline, que necesita fijar su propia ubicación de resultados.
- **Named query:** consulta guardada en la consola (perfil de demanda por hora, frescura de los datos, raw vs. curated, MAPE del batch).
- **`UNLOAD`:** escribe el resultado de una consulta como archivos en S3 (aquí, CSV comprimido) en vez de devolverlo; así la Lambda de batch prepara la entrada del Batch Transform.
- **Palabras reservadas:** `timestamp` es reservada en SQL y va entre comillas dobles (`"timestamp"`).

## Eventos, cómputo y avisos

### Amazon EventBridge

Bus de eventos y planificador.

- **Evento:** un mensaje JSON que emite un servicio de AWS (objeto creado en S3, job de Glue terminado, model package aprobado, alarma de CloudWatch cambió de estado).
- **Regla (rule):** un **event pattern** que filtra eventos (p. ej. `source: aws.glue`, `state: SUCCEEDED`, `jobName: …CurateDemand`) y uno o más **targets** a los que se envían: un workflow de Glue, un pipeline de SageMaker, una Lambda o un tópico SNS.
- **Input transformer:** arma el mensaje que recibe el target con campos del evento (el texto del correo de "pipeline fallido").
- **Schedule:** reglas por calendario con expresiones `cron(...)` o `rate(...)` (reentrenamiento semanal, batch diario, monitoreo de drift diario).
- **Disparadores del pipeline:** tres reglas arrancan el entrenamiento con un parámetro `TriggeredBy` distinto: `new-data`, `schedule` y `data-drift`.

### AWS Lambda

Funciones serverless: código que corre ante un evento, sin administrar servidores, cobrado por milisegundo.

- **Handler:** la función de entrada (`src/functions/.../x.handler`), que recibe el `event` y el `context`.
- **Runtime y arquitectura:** Python 3.13 sobre arm64 (más barato). Las funciones solo usan la librería estándar y boto3, sin capas ni dependencias extra.
- **Empaquetado individual:** cada función incluye solo los archivos que importa (`package.patterns`).
- **Variables de entorno:** su configuración (bucket, endpoint, nombres de parámetros SSM).
- **Timeout y memoria:** límites por función (p. ej. 120 s y 512 MB para generar el dataset).
- **Las cinco funciones:** `IngestDataset` (genera la serie raw), `OnModelApproved` (publica el contrato de despliegue), `Predict` (API online), `DataDrift` (monitoreo diario) y `BatchScoring` (arranca el batch).

### Amazon API Gateway

Servicio para publicar APIs HTTP.

- **HTTP API:** la variante simple y barata de API Gateway. Expone `POST /predict` y reenvía cada petición a la Lambda `Predict` (_proxy integration_).
- **Stage y throttling:** el stage por defecto limita la API a 10 peticiones por segundo con ráfagas de 20; el exceso recibe `429 Too Many Requests`. Como no hay autenticación, es la protección contra abuso y costos.

### Amazon SNS

Servicio de mensajería publicar/suscribir.

- **Tópico:** canal al que se publican mensajes (`Notifications`).
- **Suscripción:** destino que recibe los mensajes; aquí, correos electrónicos de `NOTIFICATION_EMAILS`, que deben confirmar la suscripción.
- **Topic policy:** quién puede publicar en el tópico (EventBridge y CloudWatch de la misma cuenta).

### Amazon CloudWatch

Monitoreo de AWS.

- **Logs:** cada Lambda, job de Glue y job de SageMaker escribe sus logs en _log groups_, con retención por stage.
- **Métricas:** series de tiempo con **namespace** y **dimensiones**. SageMaker publica `Invocation5XXErrors` y `ModelLatency` del endpoint; la Lambda `DataDrift` publica métricas propias (`MaxFeatureDrift`, `FeatureDrift` por feature, `SampleCount`) en el namespace `Mlops/DemandForecasting`.
- **Alarma:** observa una métrica y cambia de estado (`OK` → `ALARM`) cuando cruza un umbral durante cierto número de periodos. Sus **acciones** avisan al tópico SNS; además, EventBridge reacciona al cambio de estado de la alarma de drift para reentrenar, y la alarma de errores 5XX dispara el rollback del despliegue blue/green.

## Contenedores

### Amazon ECR

Registro privado de imágenes Docker.

- **Repositorio:** `mlops/demand-forecasting`, con la imagen propia del modelo.
- **Tags inmutables:** un tag nunca puede apuntar a otra imagen. El tag es un hash del contenido (Dockerfile, requirements y código del modelo), así cada versión del modelo queda atada a la imagen exacta que la entrenó.
- **Lifecycle policy:** conserva solo las N imágenes más recientes.
- **Scan on push:** analiza vulnerabilidades al subir cada imagen.
- **Manifiesto Docker v2 vs. OCI:** el hosting de SageMaker solo acepta manifiestos Docker v2; `build-push.sh` los fuerza y lo verifica.

## Amazon SageMaker AI

Plataforma de ML administrada. El proyecto usa varias de sus piezas.

### Imagen propia (BYOC)

_Bring Your Own Container_: en vez de los contenedores administrados de AWS, todas las cargas corren la imagen del proyecto (`container/Dockerfile`), con las mismas versiones de LightGBM y scikit-learn del devcontainer. SageMaker solo exige un **contrato**:

- **Training:** corre la imagen con el argumento `train`. Lee los hiperparámetros de `/opt/ml/input/config/hyperparameters.json` (siempre como strings), los datos de `/opt/ml/input/data/<canal>/`, deja el modelo en `/opt/ml/model/` (SageMaker lo comprime en `model.tar.gz`) y, si falla, escribe el motivo en `/opt/ml/output/failure`.
- **Serving:** corre la imagen con `serve`; debe responder `GET /ping` (salud) y `POST /invocations` (predicción) en el puerto 8080. Aquí lo hace Flask detrás de gunicorn.
- **Processing:** se sobrescribe el entrypoint con el script del paso (`python3 -m src.pipelines.training.preprocess`).

### Processing y Training jobs

- **Processing job:** ejecuta un script sobre datos (preparar, generar baselines, evaluar). Las **entradas** se copian desde S3 a `/opt/ml/processing/...` y las **salidas** se suben a S3 al terminar.
- **Dataset definition de Athena:** una entrada de processing que es una consulta SQL; SageMaker la ejecuta y entrega el resultado como archivos al contenedor.
- **Training job:** entrena un modelo en instancias administradas que se apagan solas al terminar (aquí `ml.m5.large`).
- **Canales:** cada entrada de datos de un training job (`train`, `validation`).
- **Metric definitions:** expresiones regulares sobre los logs del entrenamiento (`validation:mape=…`) para extraer métricas y verlas en Experiments y CloudWatch.

### SageMaker Pipelines

Orquestador de flujos de ML de SageMaker.

- **Pipeline:** un DAG de pasos definido en JSON (aquí, escrito a mano en `pipeline-definition.js` para que cada paso se vea como código).
- **Parámetros:** valores que se pueden cambiar en cada ejecución (tipo de instancia, días de test, hiperparámetros, `MaxMape`, `TriggeredBy`).
- **Tipos de paso:** `Processing` (PrepareData, Baseline, Evaluate), `Training` (Train), `Condition` (CheckMape), `RegisterModel` y `Fail` (FailHighMape).
- **Referencias entre pasos:** `Get: Steps.<Paso>.…` usa la salida de un paso como entrada de otro; SageMaker deduce las dependencias.
- **Property file:** un JSON producido por un paso (`evaluation.json`) cuyos valores puede leer otro paso; el `ConditionStep` lee `regression_metrics.mape.value` y decide si registrar o fallar.
- **Ejecución (execution):** cada corrida del pipeline, con su historial y el estado de cada paso, visible en Studio.

### SageMaker Experiments

Registra las ejecuciones del pipeline como **experimentos** y **trials**, con los parámetros y las métricas de cada training job (`PipelineExperimentConfig`). Cumple el papel de MLflow Tracking en el proyecto open source.

### Model Registry

- **Model package group:** el "nombre" del modelo (`demand-forecasting`); agrupa todas las versiones.
- **Model package:** una versión registrada, con la imagen y los artefactos para servirla (**inference specification**), las instancias en que puede correr, sus **métricas** (calidad del modelo y perfil de datos) y sus **drift check baselines**.
- **Estado de aprobación:** `PendingManualApproval`, `Approved` o `Rejected`. Cada cambio emite un evento a EventBridge: el registro avisa al equipo y la aprobación activa `OnModelApproved`.
- **Customer metadata:** propiedades libres; aquí, el tag de la imagen y qué disparó el entrenamiento.

### Hosting: Model, EndpointConfig y Endpoint

Desplegar un modelo en tiempo real usa tres recursos:

- **Model:** une la imagen, los artefactos y un rol. Aquí se crea directamente desde el model package aprobado.
- **EndpointConfig:** define las **production variants** (qué modelo, tipo y cantidad de instancias, peso del tráfico) y el **data capture** (porcentaje de peticiones y respuestas que se guardan en S3).
- **Endpoint:** la URL estable (`demand-forecasting`) que recibe las peticiones. Cambiar de modelo = apuntar el endpoint a una EndpointConfig nueva.
- **Deployment config (blue/green):** al actualizar, SageMaker levanta la flota nueva, le pasa todo el tráfico, la observa durante un tiempo de espera y **vuelve atrás sola** si se dispara la alarma de errores 5XX.
- **Instancia `ml.t2.medium`:** la más pequeña; las instancias burstable (`t2`) no admiten autoescalado, así que el endpoint corre siempre una sola. Se cobra por hora mientras exista.
- **SageMaker Runtime `InvokeEndpoint`:** la llamada que hace la Lambda `Predict`, con `ContentType` (CSV) y `Accept` (JSON).

### Batch Transform

Inferencia batch sobre archivos en S3, sin endpoint permanente: levanta instancias, procesa y se apaga.

- **Transform job:** lee los archivos de entrada, los divide por línea (`SplitType: Line`) y agrupa varias filas por petición (`BatchStrategy: MultiRecord`).
- **Data processing:** filtros JSONPath que permiten mandar al modelo solo parte de cada fila y unir la respuesta al registro original: `InputFilter: $[1:]` quita el timestamp, `JoinSource: Input` une la predicción a la fila de entrada y `OutputFilter: $[0,-1]` deja solo timestamp y pronóstico.

### SageMaker Studio

IDE web de SageMaker (opcional, stack `studio`).

- **Domain:** el entorno de Studio de la cuenta, con su configuración de red y su rol de ejecución.
- **User profile:** el perfil de cada persona (aquí, `DEVELOPER`).
- Sirve para **ver** los flujos: DAG y ejecuciones del pipeline, experimentos, versiones del registry (y aprobarlas), endpoints. Las apps inactivas se apagan solas tras 60 minutos.

### Model Monitor (no se usa)

SageMaker Model Monitor era el servicio administrado para vigilar drift en endpoints, pero **cerró a clientes nuevos**. Este proyecto sigue el patrón alternativo recomendado: data capture → Lambda programada (`DataDrift`, PSI en Python puro) → métricas y alarmas de CloudWatch → SNS.

## Amazon VPC

Red virtual privada. Solo la usa SageMaker Studio, que exige una VPC con subredes. El stack `studio` crea una mínima: dos subredes públicas en dos zonas de disponibilidad, un internet gateway y **sin NAT gateways** (que cobran por hora). Studio corre en modo `PublicInternetOnly`: el tráfico de las apps sale por la red administrada de SageMaker y la VPC solo aloja el disco (EFS) de los usuarios.

## Patrones del proyecto

- **Arquitectura orientada a eventos:** ningún componente llama directamente al siguiente; S3, Glue, SageMaker y CloudWatch emiten eventos y EventBridge los conecta. Agregar una reacción nueva es agregar una regla.
- **Contrato de despliegue en SSM:** aprobar un modelo no lo despliega; publica en SSM _qué_ versión está aprobada. Desplegar el endpoint (o la próxima ejecución del batch) lee ese contrato.
- **Una imagen inmutable por versión:** la misma imagen, identificada por el hash de su contenido, prepara los datos, entrena, evalúa y sirve.
- **Infraestructura base en el mismo proyecto:** `common` y `studio` son stacks del proyecto, no un repositorio aparte; `common` se despliega primero porque los demás leen sus valores por SSM.
- **Un stack por capa:** `data`, `image`, `pipeline`, `endpoint` y `batch` se despliegan por separado, así un cambio en el endpoint no toca los datos ni el pipeline.
