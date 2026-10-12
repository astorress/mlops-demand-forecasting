import boto3

CLIENT = boto3.client("ssm")


def get_parameter(name: str) -> str:
    return CLIENT.get_parameter(Name=name)["Parameter"]["Value"]


def put_parameter(name: str, value: str) -> None:
    CLIENT.put_parameter(Name=name, Value=value, Type="String", Overwrite=True)
