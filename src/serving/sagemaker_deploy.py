"""
Deploy del modelo en AWS SageMaker (con soporte para dry_run).

En dry_run=True (default): simula el proceso sin hacer llamadas reales a AWS.
En dry_run=False: requiere boto3 configurado con credenciales AWS válidas.

Uso:
    python src/serving/sagemaker_deploy.py --dry-run
    python src/serving/sagemaker_deploy.py --endpoint-name demand-forecaster --no-dry-run
"""

import argparse
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def deploy_to_sagemaker(
    model_uri: str,
    endpoint_name: str = "demand-forecaster-endpoint",
    instance_type: str = "ml.t2.medium",
    initial_instance_count: int = 1,
    execution_role_arn: Optional[str] = None,
    region: str = "us-east-1",
    dry_run: bool = True,
) -> dict[str, Any]:
    """
    Despliega un modelo MLflow en AWS SageMaker.

    El proceso crea tres recursos de SageMaker:
    1. SageMaker Model: referencia al artefacto en S3
    2. EndpointConfig: configuración de la instancia (tipo, cantidad)
    3. Endpoint: el servicio de inferencia real

    Args:
        model_uri: URI del modelo MLflow (ej: "models:/demand_forecaster/Production")
        endpoint_name: Nombre del endpoint en SageMaker.
        instance_type: Tipo de instancia EC2 (ml.t2.medium para dev, ml.m5.xlarge para prod).
        initial_instance_count: Número inicial de instancias.
        execution_role_arn: ARN del IAM role con permisos SageMaker.
        region: Región AWS.
        dry_run: Si True, solo simula sin hacer llamadas reales.

    Returns:
        Dict con información del deploy (o simulación).
    """
    execution_role_arn = execution_role_arn or os.environ.get(
        "SAGEMAKER_EXECUTION_ROLE_ARN",
        "arn:aws:iam::123456789:role/SageMakerExecutionRole"
    )

    model_name = f"mlops-demand-forecaster-{endpoint_name}"
    endpoint_config_name = f"{endpoint_name}-config"

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Deploy en SageMaker iniciado")
    logger.info(f"  model_uri: {model_uri}")
    logger.info(f"  endpoint_name: {endpoint_name}")
    logger.info(f"  instance_type: {instance_type}")
    logger.info(f"  region: {region}")

    if dry_run:
        return _simulate_deploy(
            model_uri=model_uri,
            model_name=model_name,
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_config_name,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            execution_role_arn=execution_role_arn,
            region=region,
        )

    # --- Deploy real con boto3 ---
    try:
        import boto3
        import mlflow.sagemaker as mfs
    except ImportError as e:
        raise ImportError(f"Instala boto3 y mlflow: pip install boto3 mlflow. Error: {e}")

    # Obtener la imagen Docker de MLflow para SageMaker
    # MLflow provee una imagen pre-construida compatible con SageMaker
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/mlflow-pyfunc:latest"

    sm_client = boto3.client("sagemaker", region_name=region)

    # 1. Crear SageMaker Model
    # Este recurso apunta al artefacto del modelo en S3 (almacenado por MLflow)
    logger.info(f"Creando SageMaker Model: {model_name}")
    sm_client.create_model(
        ModelName=model_name,
        # El container usa la imagen de MLflow que sabe cómo servir modelos pyfunc
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_uri,  # S3 URI del modelo
            "Environment": {
                "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
            },
        },
        # IAM role con permisos para leer S3 y acceder a ECR
        ExecutionRoleArn=execution_role_arn,
    )

    # 2. Crear EndpointConfig
    # Define el tipo y cantidad de instancias para el endpoint
    logger.info(f"Creando EndpointConfig: {endpoint_config_name}")
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": initial_instance_count,
                "InstanceType": instance_type,
                # Traffic routing: 100% al variant por defecto
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    # 3. Crear o actualizar Endpoint
    try:
        logger.info(f"Creando Endpoint: {endpoint_name}")
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "created"
    except sm_client.exceptions.ResourceInUse:
        # Si el endpoint ya existe, actualizarlo con la nueva config
        logger.info(f"Endpoint ya existe, actualizando: {endpoint_name}")
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "updated"

    # Esperar a que el endpoint esté InService (puede tardar 5-10 minutos)
    logger.info(f"Esperando que el endpoint esté InService...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)

    endpoint_info = sm_client.describe_endpoint(EndpointName=endpoint_name)
    logger.info(f"Endpoint {action} exitosamente: {endpoint_name}")

    return {
        "endpoint_name": endpoint_name,
        "status": endpoint_info["EndpointStatus"],
        "endpoint_arn": endpoint_info["EndpointArn"],
        "action": action,
        "dry_run": False,
    }


def _simulate_deploy(
    model_uri: str,
    model_name: str,
    endpoint_name: str,
    endpoint_config_name: str,
    instance_type: str,
    initial_instance_count: int,
    execution_role_arn: str,
    region: str,
) -> dict[str, Any]:
    """Simula el proceso de deploy sin hacer llamadas reales a AWS."""
    print("\n" + "=" * 60)
    print("  [DRY RUN] Simulación de Deploy en AWS SageMaker")
    print("=" * 60)
    print()
    print("PASO 1 — Crear SageMaker Model")
    print(f"""  sm_client.create_model(
    ModelName="{model_name}",
    PrimaryContainer={{
        "Image": "<account>.dkr.ecr.{region}.amazonaws.com/mlflow-pyfunc:latest",
        "ModelDataUrl": "{model_uri}",
        "Environment": {{"MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function"}},
    }},
    ExecutionRoleArn="{execution_role_arn}",
  )""")
    print()
    print("PASO 2 — Crear EndpointConfig")
    print(f"""  sm_client.create_endpoint_config(
    EndpointConfigName="{endpoint_config_name}",
    ProductionVariants=[{{
        "VariantName": "AllTraffic",
        "ModelName": "{model_name}",
        "InitialInstanceCount": {initial_instance_count},
        "InstanceType": "{instance_type}",
        "InitialVariantWeight": 1.0,
    }}],
  )""")
    print()
    print("PASO 3 — Crear Endpoint")
    print(f"""  sm_client.create_endpoint(
    EndpointName="{endpoint_name}",
    EndpointConfigName="{endpoint_config_name}",
  )
  # Tiempo de espera estimado: 5-10 minutos""")
    print()
    print("PASO 4 — Esperar InService")
    print(f'  waiter.wait(EndpointName="{endpoint_name}")')
    print()
    print("=" * 60)
    print(f"  Endpoint URL (después del deploy):")
    print(f"  https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations")
    print("=" * 60 + "\n")

    return {
        "endpoint_name": endpoint_name,
        "endpoint_config_name": endpoint_config_name,
        "model_name": model_name,
        "instance_type": instance_type,
        "region": region,
        "dry_run": True,
        "status": "simulated",
    }


def invoke_sagemaker_endpoint(
    endpoint_name: str,
    features: dict,
    region: str = "us-east-1",
) -> dict[str, Any]:
    """
    Invoca un endpoint SageMaker con los features del modelo.

    El formato de request es el que espera MLflow pyfunc en SageMaker:
    {"dataframe_records": [{"feature1": val1, ...}]}

    Args:
        endpoint_name: Nombre del endpoint en SageMaker.
        features: Dict con los features para la predicción.
        region: Región AWS.

    Returns:
        Dict con las predicciones.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("Instala boto3: pip install boto3")

    runtime_client = boto3.client("sagemaker-runtime", region_name=region)

    # MLflow pyfunc espera el formato dataframe_records o dataframe_split
    payload = json.dumps({"dataframe_records": [features]})

    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read().decode())
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Deploy en AWS SageMaker")
    parser.add_argument("--model-uri", default="models:/demand_forecaster/Production")
    parser.add_argument("--endpoint-name", default="demand-forecaster-endpoint")
    parser.add_argument("--instance-type", default="ml.t2.medium")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = parser.parse_args()

    result = deploy_to_sagemaker(
        model_uri=args.model_uri,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        region=args.region,
        dry_run=args.dry_run,
    )
    print("\nResultado:", json.dumps(result, indent=2))
