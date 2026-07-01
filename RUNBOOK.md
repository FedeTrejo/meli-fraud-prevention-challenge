# Runbook de reproducción

Este documento contiene los pasos para ejecutar el proyecto desde cero y obtener
resultados comparables con los presentados en el informe.

## 1. Requisitos

- Python 3.13 (el proyecto fue probado con Python 3.13.5).
- Git.
- Espacio suficiente para el dataset, los notebooks ejecutados y los artefactos
  de Optuna.
- GPU NVIDIA opcional. La ejecución completa también funciona en CPU.

Las versiones de las librerías están fijadas en `requirements.txt`.

## 2. Preparar el entorno

Clonar o descargar el repositorio y abrir una terminal en su directorio raíz,
`meli-fraud-prevention-challenge`. Los comandos de las secciones siguientes
asumen esa ubicación.

Crear y activar un entorno virtual:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux o macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Incorporar el dataset

El dataset no está versionado en Git. Copiar el CSV provisto para el challenge
en la siguiente ruta, respetando exactamente el nombre:

```text
data/raw/MercadoLibre Data Scientist Technical Challenge - Dataset.csv
```

No es necesario crear archivos en `data/processed/` ni en `artifacts/`: los
notebooks generan sus dependencias y salidas automáticamente.

## 4. Elegir CPU o GPU

La detección es automática. XGBoost y CatBoost usan una GPU NVIDIA cuando está
disponible y, en caso contrario, utilizan CPU. LightGBM se ejecuta en CPU.

Para forzar CPU:

```powershell
# Windows PowerShell
$env:USAR_GPU="0"
```

```bash
# Linux o macOS
export USAR_GPU=0
```

Para forzar una GPU NVIDIA disponible, usar `USAR_GPU=1`. No se necesita GPU
para reproducir el trabajo.

## 5. Ejecutar los notebooks

Los notebooks deben ejecutarse en orden, del `01` al `06`, porque algunos
consumen artefactos generados por los anteriores. Todos los comandos se lanzan
desde la raíz del repositorio.

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_split_y_validacion.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_modelado_basico.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/04_modelos_arbol.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/05_robustez_y_finalistas.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/06_interpretabilidad_y_test.ipynb
```

También pueden ejecutarse desde Jupyter. En ese caso, abrir cada notebook en
orden y usar **Restart Kernel and Run All Cells**. El contador de la primera
celda debe comenzar en `In [1]`; esto evita que módulos o variables de una
corrida anterior queden cargados en memoria.

El notebook 04 es el paso más costoso porque realiza la búsqueda de
hiperparámetros con Optuna. El parámetro `timeout=-1` evita que Jupyter cancele
las celdas largas.

## 6. Dependencias y artefactos generados

| Notebook | Función principal | Artefactos generados |
|---|---|---|
| `01_eda` | Análisis exploratorio | Ninguno |
| `02_split_y_validacion` | Split y folds temporales | `data/processed/validation_folds.csv` |
| `03_modelado_basico` | Políticas y regresión logística | `data/processed/resultados_modelado_basico.csv` |
| `04_modelos_arbol` | Árboles y búsqueda con Optuna | `data/processed/mejores_params_arbol.json`, `data/processed/optuna_arboles.db` |
| `05_robustez_y_finalistas` | Robustez y elección del finalista | Ninguno |
| `06_interpretabilidad_y_test` | Interpretabilidad y test final | `artifacts/modelo_final/catboost_ventana_17d.cbm`, `artifacts/modelo_final/metadata.json` |

Si falta un artefacto intermedio, el notebook que lo necesita termina con un
mensaje indicando cuál de los anteriores debe ejecutarse.
