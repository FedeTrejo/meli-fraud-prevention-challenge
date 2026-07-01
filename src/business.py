"""Función de ganancia usada para comparar políticas y modelos."""

import numpy as np
import pandas as pd


MARGEN_LEGITIMA = 0.25
PERDIDA_FRAUDE = 1.00


def _validar_target(target: pd.Series) -> None:
    target = pd.Series(target)
    if target.isna().any():
        raise ValueError("El target no puede contener valores faltantes.")
    valores = set(target.unique())
    if not valores.issubset({0, 1}):
        raise ValueError("El target debe contener solamente 0 y 1.")


def ganancia_por_operacion(
    target: pd.Series,
    monto: pd.Series,
    margen_legitima: float = MARGEN_LEGITIMA,
    perdida_fraude: float = PERDIDA_FRAUDE,
) -> pd.Series:
    """Ganancia de aprobar cada operación según su resultado real."""
    _validar_target(target)
    target = pd.Series(target)
    monto = pd.Series(monto, index=target.index)
    if monto.isna().any():
        raise ValueError("El monto no puede contener valores faltantes.")

    valores = np.where(
        target.eq(0),
        margen_legitima * monto,
        -perdida_fraude * monto,
    )
    return pd.Series(valores, index=target.index, name="ganancia_si_aprueba")


def ganancia_total(
    target: pd.Series,
    monto: pd.Series,
    aprobar: pd.Series | np.ndarray,
    margen_legitima: float = MARGEN_LEGITIMA,
    perdida_fraude: float = PERDIDA_FRAUDE,
) -> float:
    """Suma la ganancia de las operaciones aprobadas; rechazar aporta cero."""
    ganancia = ganancia_por_operacion(
        target,
        monto,
        margen_legitima=margen_legitima,
        perdida_fraude=perdida_fraude,
    )
    aprobar = pd.Series(aprobar, index=ganancia.index).astype(bool)
    return float(ganancia.loc[aprobar].sum())


def ganancia_aprobar_todo(
    target: pd.Series,
    monto: pd.Series,
    margen_legitima: float = MARGEN_LEGITIMA,
    perdida_fraude: float = PERDIDA_FRAUDE,
) -> float:
    """Ganancia de referencia cuando se aprueban todas las operaciones."""
    aprobar = pd.Series(True, index=pd.Series(target).index)
    return ganancia_total(
        target,
        monto,
        aprobar,
        margen_legitima=margen_legitima,
        perdida_fraude=perdida_fraude,
    )


def umbral_aprobacion(
    margen_legitima: float = MARGEN_LEGITIMA,
    perdida_fraude: float = PERDIDA_FRAUDE,
) -> float:
    """Probabilidad máxima de fraude para que aprobar tenga ganancia esperada."""
    if margen_legitima <= 0 or perdida_fraude <= 0:
        raise ValueError("El margen y la pérdida deben ser positivos.")
    return margen_legitima / (margen_legitima + perdida_fraude)
