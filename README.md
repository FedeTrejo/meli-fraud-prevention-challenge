# Fraud Prevention - Data Scientist Technical Challenge

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
│   └── processed/    # Artefactos intermedios
├── artifacts/        # Modelo final y metadatos generados
├── notebooks/        # Flujo de análisis y modelado
├── src/              # Código fuente reutilizable importado por los notebooks
├── report/           # Informe en PDF
├── RUNBOOK.md        # Instrucciones de reproducción
├── requirements.txt
└── README.md
```

## Reproducibilidad

Las instrucciones completas para preparar el entorno, ubicar el dataset,
ejecutar los notebooks y verificar los resultados están en
[RUNBOOK.md](RUNBOOK.md).
