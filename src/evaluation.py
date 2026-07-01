"""Métricas comunes para comparar políticas y modelos."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score

from src.business import ganancia_por_operacion, ganancia_total


def metricas_decision(
    target: pd.Series,
    monto: pd.Series,
    aprobar: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Resume ganancia y errores de una decisión aprobar/rechazar."""
    target = pd.Series(target)
    monto = pd.Series(monto, index=target.index)
    aprobar = pd.Series(aprobar, index=target.index).astype(bool)

    legitima = target.eq(0)
    fraude = target.eq(1)
    rechazar = ~aprobar

    return {
        "ganancia": ganancia_total(target, monto, aprobar),
        "tasa_aprobacion": float(aprobar.mean()),
        "fraudes_rechazados": float(rechazar.loc[fraude].mean()),
        "legitimas_rechazadas": float(rechazar.loc[legitima].mean()),
    }


def metricas_probabilidad(
    target: pd.Series,
    probabilidad_fraude: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Mide ordenamiento y calidad de las probabilidades predichas.

    Incluye además versiones normalizadas por prevalencia, que sí se pueden
    promediar entre folds con distinta tasa de fraude:

    - ``pr_auc`` (precision-recall) y ``brier`` dependen de la prevalencia: su
      línea de "sin skill" es la tasa de positivos (PR-AUC) y ``p·(1-p)``
      (Brier). Promediar el valor crudo entre folds con distinta prevalencia
      mezcla escalas distintas y el orden de modelos puede cambiar.
    - ``pr_auc_norm`` reescala PR-AUC sobre su base: ``(pr_auc - p) / (1 - p)``,
      donde 0 es azar y 1 es perfecto.
    - ``brier_skill`` es el Brier Skill Score ``1 - brier / (p·(1-p))``, donde
      0 es azar y 1 es perfecto.
    - ``roc_auc`` es invariante a la prevalencia, así que se promedia tal cual.
    """
    target = pd.Series(target)
    probabilidad = pd.Series(probabilidad_fraude, index=target.index)

    precision, recall, _ = precision_recall_curve(target, probabilidad)
    pr_auc = float(auc(recall, precision))
    brier = float(brier_score_loss(target, probabilidad))
    prevalencia = float(target.mean())
    brier_base = prevalencia * (1.0 - prevalencia)

    if 0.0 < prevalencia < 1.0:
        pr_auc_norm = (pr_auc - prevalencia) / (1.0 - prevalencia)
        brier_skill = 1.0 - brier / brier_base
    else:
        pr_auc_norm = np.nan
        brier_skill = np.nan

    return {
        "roc_auc": float(roc_auc_score(target, probabilidad)),
        "pr_auc": pr_auc,
        "brier": brier,
        "prevalencia": prevalencia,
        "pr_auc_norm": float(pr_auc_norm),
        "brier_skill": float(brier_skill),
    }


def mejor_corte_score(
    score: pd.Series,
    target: pd.Series,
    monto: pd.Series,
) -> int:
    """Elige en train el corte de score con mayor ganancia retrospectiva."""
    mejor_corte = int(score.min())
    mejor_ganancia = -np.inf

    for corte in range(int(score.min()), int(score.max()) + 2):
        ganancia = ganancia_total(target, monto, score < corte)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_corte = corte

    return mejor_corte


def mejor_umbral_ganancia(
    probabilidad_fraude: pd.Series | np.ndarray,
    target: pd.Series,
    monto: pd.Series,
) -> tuple[float, float]:
    """Elige el umbral de aprobación que maximiza ganancia en datos internos.

    Se aprueba cuando ``probabilidad_fraude < umbral``. La búsqueda respeta
    empates de probabilidad: operaciones con el mismo score se deciden juntas.
    """
    probabilidad = pd.Series(probabilidad_fraude, index=pd.Series(target).index)
    ganancia = ganancia_por_operacion(target, monto)

    por_probabilidad = (
        pd.DataFrame({"probabilidad": probabilidad, "ganancia": ganancia})
        .groupby("probabilidad", as_index=False)["ganancia"]
        .sum()
        .sort_values("probabilidad")
    )
    ganancias_acumuladas = por_probabilidad["ganancia"].cumsum()

    # Rechazar todo también es una política candidata y deja ganancia cero.
    candidatas = np.concatenate([[0.0], ganancias_acumuladas.to_numpy()])
    mejor_posicion = int(np.argmax(candidatas))
    mejor_ganancia = float(candidatas[mejor_posicion])

    if mejor_posicion == 0:
        umbral = float(por_probabilidad["probabilidad"].min())
    else:
        ultima_aprobada = float(
            por_probabilidad["probabilidad"].iloc[mejor_posicion - 1]
        )
        umbral = float(np.nextafter(ultima_aprobada, np.inf))

    return umbral, mejor_ganancia


def obtener_params(
    params_guardados: dict,
    modelo: str,
    escenario: str,
    outer_fold: int,
) -> dict:
    """Recupera del JSON los hiperparámetros de un modelo, escenario y fold.

    Centraliza el formato de clave ``"{modelo}|{escenario}|outer{fold}"`` que
    usan los notebooks 04 a 06.
    """
    return params_guardados[f"{modelo}|{escenario}|outer{outer_fold}"]


def registrar_prediccion(
    probabilidad,
    mascara_validacion,
    estrategia,
    escenario,
    modelo,
    outer_fold,
    repeticion,
    *,
    y: pd.Series,
    monto: pd.Series,
    umbral: float,
) -> tuple[dict, pd.DataFrame]:
    """Arma la fila de métricas y el detalle por operación de una predicción.

    ``y``, ``monto`` y ``umbral`` se pasan explícitos (los notebooks los fijan
    una vez con ``functools.partial``). Se aprueba cuando la probabilidad de
    fraude es menor al umbral.
    """
    target_val = y.loc[mascara_validacion]
    monto_val = monto.loc[mascara_validacion]
    probabilidad = pd.Series(probabilidad, index=target_val.index)
    aprobar = probabilidad < umbral

    fila = {
        "estrategia": estrategia,
        "escenario": escenario,
        "modelo": modelo,
        "outer_fold": outer_fold,
        "repeticion": repeticion,
    }
    fila.update(metricas_decision(target_val, monto_val, aprobar))
    fila.update(metricas_probabilidad(target_val, probabilidad))

    ganancia_si_aprueba = ganancia_por_operacion(target_val, monto_val)
    detalle = pd.DataFrame({
        "row_id": target_val.index,
        "estrategia": estrategia,
        "escenario": escenario,
        "modelo": modelo,
        "outer_fold": outer_fold,
        "repeticion": repeticion,
        "target": target_val.to_numpy(),
        "monto": monto_val.to_numpy(),
        "probabilidad": probabilidad.to_numpy(),
        "aprobar": aprobar.to_numpy(),
        "ganancia_realizada": np.where(
            aprobar, ganancia_si_aprueba, 0.0
        ),
    })
    return fila, detalle


def transformar_logit(probabilidad):
    """Lleva probabilidades a la escala logit, recortando los extremos."""
    p = np.clip(np.asarray(probabilidad), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def ajustar_calibrador_sigmoide(probabilidad, target):
    """Ajusta un calibrador de Platt (sigmoide) sobre el logit de la probabilidad."""
    calibrador = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    calibrador.fit(transformar_logit(probabilidad), target)
    return calibrador


def aplicar_calibrador(calibrador, probabilidad):
    """Devuelve la probabilidad recalibrada por un calibrador ya ajustado."""
    return calibrador.predict_proba(transformar_logit(probabilidad))[:, 1]
