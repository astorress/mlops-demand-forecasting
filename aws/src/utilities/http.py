import json


def json_response(status_code: int, body: dict) -> dict:
    """Respuesta proxy de API Gateway (HTTP API)."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
        },
        "body": json.dumps(body),
    }
