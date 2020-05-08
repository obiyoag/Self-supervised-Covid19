import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, preds, labels):
        return 1 - (2 * torch.sum(preds * labels) + self.smooth) / (torch.sum(preds) + torch.sum(labels) + self.smooth)
