from copy import deepcopy
from typing import TypedDict

import torch


class _DeepCopyMixin:
    """Mixin for ``__deepcopy__`` behaviour overriding."""

    def __deepcopy__(self, memo):
        """Drop ``model`` and ``trainer`` attributes while deepcopy."""
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            if k in ["model", "trainer"]:
                v = dict()
            setattr(obj, k, deepcopy(v, memo))
            pass
        return obj


class TrainBatch(TypedDict):
    encoder_real: torch.Tensor
    decoder_real: torch.Tensor
    target: torch.Tensor


class InferenceBatch(TypedDict):
    encoder_real: torch.Tensor
    decoder_real: torch.Tensor
