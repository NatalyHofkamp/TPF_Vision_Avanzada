"""Entrenamiento base para flow matching de proteínas."""

import torch


def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        # TODO: implementar el bucle de entrenamiento
        pass
