"""
config.py
---------
Configuración del entorno de Deep Learning:
- Detección automática de dispositivo (CUDA / MPS / CPU)
- Fijación de semillas de aleatoriedad para reproducibilidad
"""

import random
import numpy as np
import torch


def get_device() -> torch.device:
    """Detecta automáticamente el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Fija las semillas de todas las fuentes de aleatoriedad relevantes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Hiperparámetros del checkpoint
SEED = 42
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 50
