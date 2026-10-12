import boto3

CLIENT = boto3.client("sagemaker-runtime")


def invoke_endpoint(
    endpoint_name: str,
    body: bytes,
    content_type: str,
    accept: str,
) -> bytes:
    response = CLIENT.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept=accept,
        Body=body,
    )
    return response["Body"].read()
