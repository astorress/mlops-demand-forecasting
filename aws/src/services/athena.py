import time

import boto3

from src.utilities.errors import AthenaQueryError

CLIENT = boto3.client("athena")
POLL_SECONDS = 2


def run_query(query: str, work_group: str) -> str:
    """Ejecuta una consulta y la espera; devuelve el execution id."""
    execution_id = CLIENT.start_query_execution(
        QueryString=query,
        WorkGroup=work_group,
    )["QueryExecutionId"]
    while True:
        status = CLIENT.get_query_execution(
            QueryExecutionId=execution_id)["QueryExecution"]["Status"]
        if status["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(POLL_SECONDS)
    if status["State"] != "SUCCEEDED":
        raise AthenaQueryError(
            f"Athena query {execution_id} {status['State']}: "
            f"{status.get('StateChangeReason')}")
    return execution_id
