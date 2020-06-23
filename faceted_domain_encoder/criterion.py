from torch import nn


class MaskedMSELoss(nn.Module):
    def __init__(self, category_missing_value=1, mask=True):
        super().__init__()
        self.category_missing_value = category_missing_value
        self.mask = mask

    def forward(self, y_predict, y):
        mean_absolute_error = y - y_predict

        if self.mask:
            category_mask = (y != self.category_missing_value).bool()
            mean_absolute_error = mean_absolute_error * category_mask

        return (mean_absolute_error ** 2).mean()
