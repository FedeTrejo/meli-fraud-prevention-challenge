"""Carga y validaciones básicas del dataset original."""

from pathlib import Path

import pandas as pd


TARGET = "fraude"
DATA_FILE = "MercadoLibre Data Scientist Technical Challenge - Dataset.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / DATA_FILE
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def cargar_datos(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Carga el CSV sin transformar sus columnas."""
    return pd.read_csv(Path(path))


def convertir_fecha(serie: pd.Series) -> pd.Series:
    """Convierte una serie a datetime y falla si algún valor no es válido."""
    fecha = pd.to_datetime(serie, errors="coerce")
    if fecha.isna().any():
        n_invalidas = int(fecha.isna().sum())
        raise ValueError(f"Hay {n_invalidas} fechas que no pudieron convertirse.")
    return fecha
