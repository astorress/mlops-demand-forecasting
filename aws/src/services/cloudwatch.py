import boto3

CLIENT = boto3.client("cloudwatch")
MAX_METRICS_PER_CALL = 1000


def put_metrics(namespace: str, metrics: list) -> None:
    """metrics: [{"MetricName", "Value", "Dimensions": [...], "Unit"}]"""
    for start in range(0, len(metrics), MAX_METRICS_PER_CALL):
        CLIENT.put_metric_data(
            Namespace=namespace,
            MetricData=metrics[start:start + MAX_METRICS_PER_CALL],
        )
