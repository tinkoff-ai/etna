from typing import TypedDict

import torch


class TrainBatch(TypedDict):
    encoder_real: torch.Tensor
    decoder_real: torch.Tensor
    target: torch.Tensor


class InferenceBatch(TypedDict):
    encoder_real: torch.Tensor
    decoder_real: torch.Tensor
