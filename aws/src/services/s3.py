import json
from urllib.parse import urlparse

import boto3

CLIENT = boto3.client("s3")


def put_object(bucket: str, key: str, body: bytes, content_type: str) -> None:
    CLIENT.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
    )


def read_json(uri: str) -> dict:
    parsed = urlparse(uri)
    response = CLIENT.get_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
    )
    return json.loads(response["Body"].read())


def list_keys(bucket: str, prefix: str) -> list:
    paginator = CLIENT.get_paginator("list_objects_v2")
    return [
        item["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for item in page.get("Contents", [])
    ]


def read_text(bucket: str, key: str) -> str:
    response = CLIENT.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")
