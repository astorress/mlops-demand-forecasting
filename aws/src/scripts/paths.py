"""Rutas locales que usan los scripts (data/ y models/ están en .gitignore)."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
LOCAL_DATA_DIR = PROJECT_DIR / "data" / "local"
LOCAL_MODELS_DIR = PROJECT_DIR / "models" / "local"
RAW_DATASET_PATH = LOCAL_DATA_DIR / "demand.csv"
