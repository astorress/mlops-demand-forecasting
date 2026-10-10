-- Crea la base de datos de MLflow (la de Airflow la crea POSTGRES_DB)
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlops;
