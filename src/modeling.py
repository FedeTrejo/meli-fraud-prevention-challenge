"""Constructores comunes para comparar modelos con el mismo protocolo."""

import os
import shutil
import subprocess

import pandas as pd
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.features import (
    CATEGORICA_ALTA_LOGISTICA,
    CATEGORICAS_ARBOL,
    CATEGORICAS_BAJAS_LOGISTICA,
    NUMERICAS_LOGISTICA,
    agregar_features_dirigidas_por_train,
    preparar_categoricas_por_train,
)

# Subgrupos categóricos para el pipeline de Random Forest: las de baja
# cardinalidad van a one-hot y "j" (alta cardinalidad) a target encoding.
CATEGORICAS_BAJAS_ARBOL = ["a", "g", "o", "p"]
CATEGORICA_ALTA_ARBOL = ["j"]


def _hay_gpu_nvidia() -> bool:
    """Detecta una GPU NVIDIA utilizable.

    XGBoost y CatBoost se ajustan en GPU si está disponible y caen a CPU si no,
    de modo que los notebooks corran en cualquier máquina. La variable de entorno
    ``USAR_GPU`` (0/1) permite forzar la decisión.
    """
    forzar = os.environ.get("USAR_GPU")
    if forzar is not None:
        return forzar == "1"
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


USAR_GPU = _hay_gpu_nvidia()


def crear_pipeline_logistica(
    incluir_score: bool = True,
    incluir_k: bool = False,
    semilla: int = 42,
) -> Pipeline:
    """Reproduce la regresión logística elegida en el notebook 03."""
    columnas_numericas = NUMERICAS_LOGISTICA.copy()
    if incluir_score:
        columnas_numericas.append("score")
    if incluir_k:
        columnas_numericas.append("k")

    preprocesamiento = ColumnTransformer(
        [
            (
                "numericas",
                Pipeline(
                    [
                        ("imputacion", SimpleImputer(strategy="median")),
                        ("escalado", StandardScaler()),
                    ]
                ),
                columnas_numericas,
            ),
            (
                "categoricas",
                Pipeline(
                    [
                        (
                            "imputacion",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="faltante",
                            ),
                        ),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAS_BAJAS_LOGISTICA,
            ),
            (
                "j",
                Pipeline(
                    [
                        (
                            "imputacion",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="faltante",
                            ),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=20,
                            ),
                        ),
                    ]
                ),
                CATEGORICA_ALTA_LOGISTICA,
            ),
        ]
    )

    modelo = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        random_state=semilla,
    )
    return Pipeline(
        [
            ("preprocesamiento", preprocesamiento),
            ("modelo", modelo),
        ]
    )


def construir_pipeline_rf(params: dict, x: pd.DataFrame, semilla: int = 42) -> Pipeline:
    """Pipeline de Random Forest con el mismo preprocesamiento usado en el notebook 04.

    Numéricas imputadas por mediana, categóricas de baja cardinalidad en one-hot
    y "j" con target encoding. ``params`` son los hiperparámetros del bosque.
    """
    columnas_numericas = [c for c in x.columns if c not in CATEGORICAS_ARBOL]
    preprocesamiento = ColumnTransformer(
        [
            ("numericas", SimpleImputer(strategy="median"), columnas_numericas),
            (
                "categoricas_bajas",
                Pipeline(
                    [
                        (
                            "imputacion",
                            SimpleImputer(strategy="constant", fill_value="faltante"),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAS_BAJAS_ARBOL,
            ),
            (
                "j_target",
                TargetEncoder(target_type="binary", cv=5, random_state=42),
                CATEGORICA_ALTA_ARBOL,
            ),
        ]
    )
    modelo = RandomForestClassifier(random_state=semilla, n_jobs=-1, **params)
    return Pipeline(
        [
            ("preprocesamiento", preprocesamiento),
            ("modelo", modelo),
        ]
    )


def ajustar_arbol(
    nombre: str,
    params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validacion: pd.DataFrame,
    semilla: int = 42,
    peso_positivo: float = 1.0,
    usar_features_dirigidas: bool = False,
) -> tuple:
    """Ajusta un boosting y devuelve el modelo y los frames ya preparados.

    Las categóricas se aprenden solamente con train (`preparar_categoricas_por_train`).
    Devuelve ``(modelo, train_preparado, validacion_preparada)`` para poder
    reusar el modelo ajustado, por ejemplo para SHAP, además de predecir.
    """
    if usar_features_dirigidas:
        x_train, x_validacion = agregar_features_dirigidas_por_train(
            x_train,
            x_validacion,
        )

    train, validacion = preparar_categoricas_por_train(
        x_train,
        x_validacion,
    )

    if nombre == "xgboost":
        modelo = xgb.XGBClassifier(
            tree_method="hist",
            device="cuda" if USAR_GPU else "cpu",
            enable_categorical=True,
            eval_metric="logloss",
            random_state=semilla,
            scale_pos_weight=peso_positivo,
            **params,
        )
        modelo.fit(train, y_train)
    elif nombre == "lightgbm":
        modelo = lgb.LGBMClassifier(
            random_state=semilla,
            n_jobs=-1,
            verbose=-1,
            bagging_freq=1,
            scale_pos_weight=peso_positivo,
            **params,
        )
        modelo.fit(
            train,
            y_train,
            categorical_feature=CATEGORICAS_ARBOL,
        )
    elif nombre == "catboost":
        modelo = CatBoostClassifier(
            task_type="GPU" if USAR_GPU else "CPU",
            random_seed=semilla,
            verbose=False,
            allow_writing_files=False,
            scale_pos_weight=peso_positivo,
            **params,
        )
        modelo.fit(
            Pool(train, y_train, cat_features=CATEGORICAS_ARBOL)
        )
    elif nombre == "rf":
        modelo = construir_pipeline_rf(params, train, semilla=semilla)
        modelo.fit(train, y_train)
    else:
        raise ValueError(f"Modelo no reconocido: {nombre}")

    return modelo, train, validacion


def predecir_arbol(nombre: str, modelo, validacion: pd.DataFrame) -> pd.Series:
    """Devuelve la probabilidad de fraude sobre un frame ya preparado."""
    if nombre == "catboost":
        probabilidad = modelo.predict_proba(
            Pool(validacion, cat_features=CATEGORICAS_ARBOL)
        )[:, 1]
    else:
        probabilidad = modelo.predict_proba(validacion)[:, 1]
    return pd.Series(probabilidad, index=validacion.index)


def ajustar_y_predecir_arbol(
    nombre: str,
    params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validacion: pd.DataFrame,
    semilla: int = 42,
    peso_positivo: float = 1.0,
    usar_features_dirigidas: bool = False,
) -> pd.Series:
    """Ajusta un boosting con categorías aprendidas solamente en train."""
    modelo, _, validacion = ajustar_arbol(
        nombre,
        params,
        x_train,
        y_train,
        x_validacion,
        semilla=semilla,
        peso_positivo=peso_positivo,
        usar_features_dirigidas=usar_features_dirigidas,
    )
    return predecir_arbol(nombre, modelo, validacion)
