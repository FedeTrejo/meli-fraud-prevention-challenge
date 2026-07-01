"""Creación determinística de features para los modelos del proyecto."""

import numpy as np
import pandas as pd

from src.data import convertir_fecha


NUMERICAS_LOGISTICA = [
    "b",
    "d",
    "h",
    "l",
    "m",
    "n",
    "log_c",
    "log_e",
    "log_f",
    "log_monto",
    "dias_desde_inicio",
    "bc_faltante",
    "dm_faltante",
    "fl_faltante",
    "e_es_cero",
]

CATEGORICAS_BAJAS_LOGISTICA = [
    "a",
    "g",
    "o",
    "p",
    "hora",
    "dia_semana",
]

CATEGORICA_ALTA_LOGISTICA = ["j"]


NUMERICAS_ARBOL = [
    "b",
    "c",
    "d",
    "e",
    "f",
    "h",
    "l",
    "m",
    "n",
    "monto",
    "hora",
    "dia_semana",
    "dias_desde_inicio",
    "bc_faltante",
    "dm_faltante",
    "fl_faltante",
]

CATEGORICAS_ARBOL = [
    "a",
    "g",
    "o",
    "p",
    "j",
]


def crear_features_logistica(
    df: pd.DataFrame,
    incluir_score: bool,
    fecha_origen: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Crea las transformaciones justificadas por el EDA para una logística."""
    fecha = convertir_fecha(df["fecha"])
    if fecha_origen is None:
        fecha_origen = fecha.min().normalize()

    x = pd.DataFrame(index=df.index)

    for col in ["b", "d", "h", "l", "m", "n"]:
        x[col] = df[col]

    x["log_c"] = np.log1p(df["c"])
    x["log_e"] = np.log1p(df["e"])
    x["log_f"] = np.sign(df["f"]) * np.log1p(df["f"].abs())
    x["log_monto"] = np.log1p(df["monto"])
    x["dias_desde_inicio"] = (fecha.dt.normalize() - fecha_origen).dt.days

    x["bc_faltante"] = df["b"].isna().astype("int8")
    x["dm_faltante"] = df["d"].isna().astype("int8")
    x["fl_faltante"] = df["f"].isna().astype("int8")
    x["e_es_cero"] = df["e"].eq(0).astype("int8")

    for col in ["a", "g", "o", "p", "j"]:
        x[col] = df[col].astype("object").where(df[col].notna(), np.nan)

    x["hora"] = fecha.dt.hour.astype("object")
    x["dia_semana"] = fecha.dt.dayofweek.astype("object")

    if incluir_score:
        x["score"] = df["score"]

    return x


def crear_features_arbol(
    df: pd.DataFrame,
    incluir_score: bool,
    fecha_origen: pd.Timestamp | None = None,
    incluir_k: bool = False,
) -> pd.DataFrame:
    """Crea features para modelos de árboles, según las ideas del EDA.

    A diferencia de la logística, los árboles no necesitan log, escalado ni
    recorte de colas, y varios manejan NaN y categóricas de forma nativa. Por
    eso las numéricas quedan crudas (con NaN) y las categóricas conservan sus
    valores con la ausencia como nivel explícito. Sus categorías definitivas
    se aprenden después usando solo el train de cada fold.
    """
    fecha = convertir_fecha(df["fecha"])
    if fecha_origen is None:
        fecha_origen = fecha.min().normalize()

    x = pd.DataFrame(index=df.index)

    for col in ["b", "c", "d", "e", "f", "h", "l", "m", "n", "monto"]:
        x[col] = df[col]

    x["hora"] = fecha.dt.hour.astype("int16")
    x["dia_semana"] = fecha.dt.dayofweek.astype("int16")
    x["dias_desde_inicio"] = (fecha.dt.normalize() - fecha_origen).dt.days.astype("int16")

    x["bc_faltante"] = df["b"].isna().astype("int8")
    x["dm_faltante"] = df["d"].isna().astype("int8")
    x["fl_faltante"] = df["f"].isna().astype("int8")

    for col in CATEGORICAS_ARBOL:
        serie = df[col].astype("object").where(df[col].notna(), "faltante")
        x[col] = serie

    if incluir_score:
        x["score"] = df["score"]

    if incluir_k:
        x["k"] = df["k"]

    return x


def preparar_categoricas_por_train(
    x_train: pd.DataFrame,
    x_validacion: pd.DataFrame,
    columnas: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aprende los niveles categóricos en train y agrupa los nuevos."""
    if columnas is None:
        columnas = CATEGORICAS_ARBOL

    train = x_train.copy()
    validacion = x_validacion.copy()

    for col in columnas:
        valores_train = train[col].astype("object").where(
            train[col].notna(), "faltante"
        ).astype(str)
        valores_validacion = validacion[col].astype("object").where(
            validacion[col].notna(), "faltante"
        ).astype(str)
        conocidos = list(pd.unique(valores_train))
        etiqueta_nueva = "desconocida"
        while etiqueta_nueva in conocidos:
            etiqueta_nueva = f"_{etiqueta_nueva}"

        categorias = conocidos + [etiqueta_nueva]
        valores_validacion = valores_validacion.where(
            valores_validacion.isin(conocidos), etiqueta_nueva
        )

        train[col] = pd.Categorical(valores_train, categories=categorias)
        validacion[col] = pd.Categorical(
            valores_validacion, categories=categorias
        )

    return train, validacion


def agregar_features_dirigidas_por_train(
    x_train: pd.DataFrame,
    x_validacion: pd.DataFrame,
    frecuencia_rara: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega frecuencia de j y un resumen de faltantes sin mirar validación."""
    if frecuencia_rara <= 0:
        raise ValueError("frecuencia_rara debe ser positiva.")

    train = x_train.copy()
    validacion = x_validacion.copy()
    j_train = train["j"].astype("object").where(
        train["j"].notna(), "faltante"
    ).astype(str)
    j_validacion = validacion["j"].astype("object").where(
        validacion["j"].notna(), "faltante"
    ).astype(str)

    conteos = j_train.value_counts()
    for datos, valores_j in [
        (train, j_train),
        (validacion, j_validacion),
    ]:
        frecuencia = valores_j.map(conteos).fillna(0)
        datos["j_frecuencia"] = frecuencia.astype("int32")
        datos["j_frecuencia_relativa"] = (
            frecuencia / len(train)
        ).astype("float32")
        datos["j_es_rara"] = frecuencia.lt(frecuencia_rara).astype("int8")
        datos["cantidad_faltantes"] = datos[
            ["bc_faltante", "dm_faltante", "fl_faltante"]
        ].sum(axis=1).astype("int8")

    return train, validacion
