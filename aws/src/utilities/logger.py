import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Lambda ya configura el handler raíz; solo se fija el nivel."""
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    return logger
