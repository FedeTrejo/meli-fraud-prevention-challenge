"""Definición y validación del esquema temporal común a todos los modelos."""

import pandas as pd


TEST_START = pd.Timestamp("2020-04-15")
TEST_END = pd.Timestamp("2020-04-22")


def crear_folds_temporales() -> pd.DataFrame:
    """Devuelve los tres folds acumulados fijados para el proyecto."""
    return pd.DataFrame(
        [
            {
                "fold": 1,
                "train_start": pd.Timestamp("2020-03-08"),
                "train_end": pd.Timestamp("2020-03-25"),
                "validation_start": pd.Timestamp("2020-03-25"),
                "validation_end": pd.Timestamp("2020-04-01"),
            },
            {
                "fold": 2,
                "train_start": pd.Timestamp("2020-03-08"),
                "train_end": pd.Timestamp("2020-04-01"),
                "validation_start": pd.Timestamp("2020-04-01"),
                "validation_end": pd.Timestamp("2020-04-08"),
            },
            {
                "fold": 3,
                "train_start": pd.Timestamp("2020-03-08"),
                "train_end": pd.Timestamp("2020-04-08"),
                "validation_start": pd.Timestamp("2020-04-08"),
                "validation_end": TEST_START,
            },
        ]
    )


def crear_folds_ventana(
    folds: pd.DataFrame,
    dias_train: int = 17,
) -> pd.DataFrame:
    """Mantiene las validaciones y limita cada train a la historia reciente."""
    if dias_train <= 0:
        raise ValueError("dias_train debe ser positivo.")

    resultado = folds.copy()
    resultado["train_start"] = (
        resultado["train_end"] - pd.Timedelta(days=dias_train)
    )
    return resultado


def mascaras_desarrollo_test(
    fecha: pd.Series,
    test_start: pd.Timestamp = TEST_START,
    test_end: pd.Timestamp = TEST_END,
) -> tuple[pd.Series, pd.Series]:
    """Separa desarrollo y test final usando límites inclusivo/exclusivo."""
    es_desarrollo = fecha < test_start
    es_test = (fecha >= test_start) & (fecha < test_end)
    return es_desarrollo, es_test


def mascaras_fold(
    fecha: pd.Series,
    fold: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Devuelve las máscaras de train y validación para un fold."""
    train = (fecha >= fold["train_start"]) & (fecha < fold["train_end"])
    validacion = (
        (fecha >= fold["validation_start"])
        & (fecha < fold["validation_end"])
    )
    return train, validacion


def validar_folds(
    fecha: pd.Series,
    folds: pd.DataFrame,
    es_test: pd.Series,
) -> pd.DataFrame:
    """Comprueba orden temporal, separación y tamaños de todos los folds."""
    chequeos = []
    for _, fold in folds.iterrows():
        train, validacion = mascaras_fold(fecha, fold)
        chequeos.append(
            {
                "fold": int(fold["fold"]),
                "sin_superposicion": not (train & validacion).any(),
                "train_antes_de_validacion": (
                    fecha.loc[train].max() < fecha.loc[validacion].min()
                ),
                "sin_filas_de_test": not ((train | validacion) & es_test).any(),
                "n_train": int(train.sum()),
                "n_validacion": int(validacion.sum()),
            }
        )

    resultado = pd.DataFrame(chequeos).set_index("fold")
    columnas_booleanas = [
        "sin_superposicion",
        "train_antes_de_validacion",
        "sin_filas_de_test",
    ]
    if not resultado[columnas_booleanas].all().all():
        raise ValueError("Los folds no respetan la separación temporal.")
    return resultado
