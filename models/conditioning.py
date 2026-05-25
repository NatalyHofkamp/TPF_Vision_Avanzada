import torch
import torch.nn as nn


class ConditionEmbedding(nn.Module):
    """
    Embedding simple para conditioning.

    0 -> inactive
    1 -> active
    """

    def __init__(self, num_classes=2, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, labels):
        return self.embedding(labels)
