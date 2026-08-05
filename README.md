# Proyecto Integrador — Data Science III

## Pre-Entrega N°1: Pipeline de entrenamiento, validación y clasificador base

### Objetivo
Construir la infraestructura técnica inicial del proyecto: un pipeline de entrenamiento y
validación en PyTorch, reproducible y bien organizado, usando un clasificador base sobre
un dataset de referencia.

### Estructura del repositorio
```
proyecto-ds3/
├── data/                # (dataset cargado directamente vía sklearn, no requiere archivos locales)
├── src/
│   ├── config.py        # device (cpu/cuda/mps), seeds, hiperparámetros
│   ├── data.py           # carga y preparación del dataset (DataLoaders)
│   ├── model.py           # arquitectura del clasificador base (MLP nn.Sequential)
│   └── train.py            # training loop, validation loop, métricas
├── notebooks/            # (reservado para exploración futura)
├── requirements.txt
└── README.md
```

### Dataset utilizado
Se usó el dataset **Iris** (clásico, vía `sklearn.datasets.load_iris`) como dataset de
referencia para esta prueba técnica: 150 muestras, 4 features numéricas, 3 clases.
Se aplicó `train_test_split` estratificado (80/20) y estandarización (`StandardScaler`,
ajustado solo con el set de entrenamiento para evitar data leakage).

### Arquitectura
Clasificador base tipo MLP de una capa oculta:
`Linear(4 → 16) → ReLU → Linear(16 → 3)`

### Configuración del entorno
- **Detección automática de dispositivo:** `cuda` → `mps` → `cpu` (fallback).
- **Semilla fija:** `SEED = 42` (aplicada a `random`, `numpy` y `torch`) para reproducibilidad.
- **Versión de PyTorch utilizada:** 2.13.0 (build cu130)
- **Learning rate elegido:** `1e-3` (Adam) — valor estándar de partida que, en las pruebas
  realizadas, logró una convergencia estable sin oscilaciones en la pérdida.
- **Optimizador:** Adam
- **Épocas:** 50
- **Batch size:** 16

### Resultados del experimento inicial
La pérdida de entrenamiento y validación descendió de forma consistente y monotónica a lo
largo de las 50 épocas (de ~1.03 a ~0.25 en train, y de ~1.01 a ~0.30 en validación), sin
señales de overfitting temprano: la curva de validación acompaña a la de entrenamiento sin
separarse significativamente. El accuracy de validación final alcanzado fue de **86.7%**.

Esto confirma que el ciclo de `forward → loss → backward → step` funciona correctamente y
que `torch.autograd` está computando y propagando los gradientes como se espera.

### Errores comunes evitados
- **`optimizer.zero_grad()`** se llama explícitamente en cada iteración del training loop,
  antes del forward pass, para evitar la acumulación de gradientes entre batches.
- **Consistencia de dispositivos:** tanto el modelo (`model.to(device)`) como cada batch de
  datos (`X_batch.to(device)`, `y_batch.to(device)`) se mueven al mismo `device` antes de
  cualquier operación, evitando errores de mezcla CPU/GPU.

### Cómo ejecutar
```bash
pip install -r requirements.txt
cd src
python train.py
```

### Próximos pasos
Este pipeline base servirá como fundación para los siguientes checkpoints del proyecto
(clasificadores de texto, sistemas de traducción y modelos generativos).
