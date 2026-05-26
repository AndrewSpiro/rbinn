from LocalLearning.LocalLearning import KHModel
import torch
from torch import nn

class KHModelWrapper(nn.Module):
    def __init__(self, model: KHModel):
        super().__init__()
        self.model = model

    def to(self, device):
        self.model.to(device)
        return self 

    def forward(self, x):
        logits, hidden = self.model(x)
        return logits 