from enum import Enum

import numpy as np

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    import torch.nn as nn


class NBeatsSMAPE(nn.Module):
    """SMAPE with mask."""

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, mask):
        """Compute metric.

        Parameters
        ----------
        y_true:
            True target.
        y_pred:
            Predicted target.
        mask:
            Binary mask that denotes which points are valid forecasts.

        Returns
        -------
        :
            Metric value.
        """
        ae = torch.abs(y_true - y_pred)
        sape = ae / (torch.abs(y_true) + torch.abs(y_pred))

        sape[sape != sape] = 0.0
        sape[sape == np.inf] = 0.0

        return 200 * torch.mean(sape * mask)


class NBeatsMAPE(nn.Module):
    """MAPE with mask."""

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, mask):
        """Compute metric.

        Parameters
        ----------
        y_true:
            True target.
        y_pred:
            Predicted target.
        mask:
            Binary mask that denotes which points are valid forecasts.

        Returns
        -------
        :
            Metric value.
        """
        ape = torch.abs(y_true - y_pred) / torch.abs(y_true)

        ape[ape != ape] = 0.0
        ape[ape == np.inf] = 0.0

        return torch.mean(ape * mask)


class NBeatsMAE(nn.Module):
    """MAE with mask."""

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, mask):
        """Compute metric.

        Parameters
        ----------
        y_true:
            True target.
        y_pred:
            Predicted target.
        mask:
            Binary mask that denotes which points are valid forecasts.

        Returns
        -------
        :
            Metric value.
        """
        return torch.mean(mask * torch.abs(y_true - y_pred))


class NBeatsMSE(nn.Module):
    """MSE with mask."""

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, mask):
        """Compute metric.

        Parameters
        ----------
        y_true:
            True target.
        y_pred:
            Predicted target.
        mask:
            Binary mask that denotes which points are valid forecasts.

        Returns
        -------
        :
            Metric value.
        """
        return torch.mean(mask * (y_true - y_pred) ** 2)


class NBeatsLoss(Enum):
    """Enum with N-BEATS supported losses."""

    smape = NBeatsSMAPE()
    mape = NBeatsMAPE()
    mae = NBeatsMAE()
    mse = NBeatsMSE()
