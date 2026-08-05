"""
train.py
--------
Pipeline de entrenamiento y validación del clasificador base.

Implementa explícitamente:
  1. Configuración del entorno (device + semillas)
  2. Arquitectura base
  3. Ciclo de entrenamiento (forward, loss, backward, step)
  4. Ciclo de validación (evaluación en datos no vistos)
  5. Métricas y tracking (loss + accuracy por época)

Errores comunes evitados:
  - zero_grad() en cada iteración (si no, los gradientes se acumulan)
  - uso consistente de .to(device) en modelo y tensores
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import get_device, set_seed, SEED, LEARNING_RATE, BATCH_SIZE, EPOCHS
from data import get_dataloaders
from model import BaseClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()          # 1. reinicio de gradientes (evita acumulación)
        outputs = model(X_batch)       # 2. forward pass
        loss = criterion(outputs, y_batch)  # 3. cálculo de pérdida
        loss.backward()                # 4. backward pass (autograd)
        optimizer.step()               # 5. actualización de pesos

        running_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        running_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return running_loss / total, correct / total


def main():
    # 1. Configuración del entorno
    set_seed(SEED)
    device = get_device()
    print(f"Dispositivo detectado: {device}")

    # Datos
    train_loader, val_loader, n_features, n_classes = get_dataloaders(
        batch_size=BATCH_SIZE, seed=SEED
    )

    # 2. Arquitectura base
    model = BaseClassifier(n_features, n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 3 y 4. Ciclo de entrenamiento + validación
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    print("\nEntrenamiento finalizado.")
    print(f"Accuracy final de validación: {history['val_acc'][-1]:.4f}")

    return history


if __name__ == "__main__":
    main()
