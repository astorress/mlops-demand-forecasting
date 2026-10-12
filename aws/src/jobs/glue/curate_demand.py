"""Job ETL de Glue (Spark): CSV de la zona raw → Parquet de la zona curada.

    1. Lee los CSV crudos de una fecha de ingesta
       (raw/<proyecto>/ingestion_date=YYYY-MM-DD/*.csv).
    2. Tipa columnas y deja un timestamp único por hora.
    3. Calcula el calendario (hora, día de la semana con lunes = 0, mes, fin de
       semana, festivo), los lags por tiempo (t − 24 h, t − 168 h) y la media
       móvil de las 168 horas anteriores (sin incluir t). Descarta la primera
       semana, que no tiene lag de 168 h. Mirror en pandas:
       src/features/curation.py — mantener sincronizados.
    4. Evalúa el ruleset de Glue Data Quality; una regla que falle hace fallar
       el job, así ningún entrenamiento corre con datos malos.
    5. Sobrescribe la partición de esa fecha en la tabla curada (Parquet), que
       Athena lee con partition projection.

Argumentos: --RAW_PATH, --CURATED_PATH, --DQ_RULESET, --INGESTION_DATE
("latest" toma la partición raw más reciente).
"""
import sys

from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from awsgluedq.transforms import EvaluateDataQuality
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql import functions as F

TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss"
HOUR_SECONDS = 3600
# Festivos (mes-día) de src/features/calendar.py.
COLOMBIAN_HOLIDAYS = [
    "01-01",
    "01-06",
    "03-25",
    "04-13",
    "04-14",
    "05-01",
    "05-19",
    "06-09",
    "06-16",
    "06-29",
    "07-20",
    "08-07",
    "08-19",
    "10-12",
    "11-02",
    "11-11",
    "12-08",
    "12-25",
]
COLUMNS = [
    "timestamp",
    "temperature_c",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_mean_7d",
    "demand_kwh",
]

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "RAW_PATH", "CURATED_PATH", "DQ_RULESET", "INGESTION_DATE"],
)
glue_context = GlueContext(SparkContext.getOrCreate())
spark = glue_context.spark_session
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
# Horas locales sin zona: UTC evita corrimientos al convertir.
spark.conf.set("spark.sql.session.timeZone", "UTC")
# `ingestion_date` (de los nombres de carpeta) queda como string.
spark.conf.set("spark.sql.sources.partitionColumnTypeInference.enabled",
               "false")
job = Job(glue_context)
job.init(args["JOB_NAME"], args)

# Spark descubre `ingestion_date` de las carpetas estilo Hive.
raw = spark.read.option("header", "true").csv(args["RAW_PATH"])
ingestion_date = args["INGESTION_DATE"]
if ingestion_date == "latest":
    ingestion_date = raw.agg(F.max("ingestion_date")).first()[0]
raw = raw.filter(F.col("ingestion_date") == F.lit(ingestion_date))
print(f"Curating ingestion_date={ingestion_date}")

measurements = raw.select(
    F.to_timestamp(F.trim("timestamp"), TIMESTAMP_FORMAT).alias("moment"),
    F.col("demand_kwh").cast("double").alias("demand_kwh"),
    F.col("temperature_c").cast("double").alias("temperature_c"),
).dropna(subset=["moment"]).dropDuplicates(["moment"])
measurements = measurements.withColumn("epoch", F.col("moment").cast("long"))


def lag(hours: int):
    """Demanda de t − hours, unida por tiempo (no por posición): un hueco en
    la serie deja el lag en null en vez de correrlo."""
    return measurements.select(
        (F.col("epoch") + hours * HOUR_SECONDS).alias("epoch"),
        F.col("demand_kwh").alias(f"demand_lag_{hours}h"),
    )


# Las 168 horas anteriores a t, sin t (sin fuga del target).
last_week = Window.orderBy("epoch").rangeBetween(-168 * HOUR_SECONDS, -1)
day_of_week = (F.dayofweek("moment") + 5) % 7  # Spark: domingo = 1

curated = measurements.join(lag(24), "epoch", "left")
curated = curated.join(lag(168), "epoch", "left")
curated = curated.withColumn(
    "demand_rolling_mean_7d",
    F.round(F.avg("demand_kwh").over(last_week), 2),
)
calendar = {
    "timestamp":
        F.date_format("moment", TIMESTAMP_FORMAT),
    "hour":
        F.hour("moment"),
    "day_of_week":
        day_of_week.cast("int"),
    "month":
        F.month("moment"),
    "is_weekend": (day_of_week >= 5).cast("int"),
    "is_holiday":
        F.date_format("moment", "MM-dd").isin(COLOMBIAN_HOLIDAYS).cast("int"),
}
for name, column in calendar.items():
    curated = curated.withColumn(name, column)
curated = curated.filter(F.col("demand_lag_168h").isNotNull())
curated = curated.select(*COLUMNS).withColumn("ingestion_date",
                                              F.lit(ingestion_date))

# Gate de calidad (reglas definidas una vez en serverless/data/serverless.yml).
dq_results = EvaluateDataQuality.apply(
    frame=DynamicFrame.fromDF(curated, glue_context, "curated"),
    ruleset=args["DQ_RULESET"],
    publishing_options={
        "dataQualityEvaluationContext": "curated_demand",
        "enableDataQualityCloudWatchMetrics": True,
        "enableDataQualityResultsPublishing": True,
    },
).toDF()
dq_results.show(truncate=False)
failed_rules = dq_results.filter(F.col("Outcome") != "Passed").collect()
if failed_rules:
    raise RuntimeError("Data quality rules failed: " +
                       "; ".join(f"{row['Rule']} → {row['FailureReason']}"
                                 for row in failed_rules))

(curated.repartition(1).write.mode("overwrite").partitionBy(
    "ingestion_date").parquet(args["CURATED_PATH"]))
print(f"Wrote {curated.count()} rows to {args['CURATED_PATH']}")

job.commit()
