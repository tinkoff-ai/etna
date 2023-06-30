import numpy as np
import pytest
import torch

from etna.models.nn.nbeats.metrics import NBeatsMAE
from etna.models.nn.nbeats.metrics import NBeatsMAPE
from etna.models.nn.nbeats.metrics import NBeatsMSE
from etna.models.nn.nbeats.metrics import NBeatsSMAPE


@pytest.fixture
def tru_pred_mask_tensors():
    a = torch.arange(5)
    b = a.flip(dims=(0,))
    c = (a >= 2).float()
    return a, b, c


@pytest.mark.parametrize(
    "metric,expected", ((NBeatsMSE(), 4.0), (NBeatsMAE(), 1.2), (NBeatsSMAPE(), 60.0), (NBeatsMAPE(), 100 / 3))
)
def test_metric(tru_pred_mask_tensors, metric, expected):
    true, pred, mask = tru_pred_mask_tensors
    res = metric(true, pred, mask)
    np.testing.assert_allclose(res.item(), expected)
