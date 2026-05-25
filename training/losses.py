import torch.nn as nn


class FlowMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predictions, targets):
        return self.loss_fn(predictions, targets)
