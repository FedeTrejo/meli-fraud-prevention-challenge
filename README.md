# Fraud Prevention — Data Scientist Technical Challenge

Modelo de Machine Learning para predecir si una transacción es fraudulenta,
optimizando la **ganancia** de la empresa.

## Contexto económico

- Cada transacción legítima aporta un **25%** de ganancia.
- Cada fraude aprobado pierde el **100%** del monto de la transacción.

Las features del dataset están anonimizadas.

## Estructura del repositorio

```
.
├── data/
│   ├── raw/          # Dataset original (no versionado)
│   └── processed/    # Datos transformados (no versionado)
├── notebooks/        # Exploración y experimentación
├── src/              # Código fuente reutilizable
├── report/           # Informe en PDF
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

El dataset se ubica en `data/raw/` y no se versiona (ver `.gitignore`).

## Estado

> Placeholder — se completa a medida que avanza el análisis: hipótesis,
> transformaciones, modelos, evaluación y conclusión.
