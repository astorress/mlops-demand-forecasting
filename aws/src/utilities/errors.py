class RequestValidationError(ValueError):
    """Entrada inválida del cliente: el mensaje se le devuelve."""


class NoApprovedModelError(RuntimeError):
    """El contrato de despliegue (SSM) aún no tiene un modelo aprobado."""


class MissingBaselineError(ValueError):
    """El model package aprobado no trae sus baselines de drift."""


class AthenaQueryError(RuntimeError):
    pass
