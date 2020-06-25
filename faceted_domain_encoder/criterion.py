import torch
from torch import nn


class MaskedMSELoss(nn.Module):
    def __init__(self, category_missing_value=1):
        super().__init__()
        self.category_missing_value = category_missing_value

    def forward(self, y_predict, y):
        category_mask = (y != self.category_missing_value).bool()
        mse = (y - y_predict) ** 2
        masked_mse = mse * category_mask
        error = masked_mse.sum(1) / torch.clamp(masked_mse.sum(1), min=1)
        return error.mean()
