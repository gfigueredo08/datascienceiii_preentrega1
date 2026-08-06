"""
data.py
-------
Carga del dataset de referencia (Iris) y preparación de los
DataLoaders de entrenamiento y validación.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_dataloaders(batch_size: int = 16, val_size: float = 0.2, seed: int = 42):
    """
    Carga el dataset Iris (clasificación multiclase, 3 clases, 4 features),
    lo estandariza y lo separa en train/val.
    """
    X, y = load_iris(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    # Estandarización (fit solo en train, para evitar data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_features = X_train.shape[1]
    n_classes = len(set(y))

    return train_loader, val_loader, n_features, n_classes
