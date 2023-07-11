import numpy as np
import pytest
import torch

from etna.models.nn.nbeats.utils import _create_or_update
from etna.models.nn.nbeats.utils import prepare_test_batch
from etna.models.nn.nbeats.utils import prepare_train_batch
from etna.models.nn.nbeats.utils import to_tensor


@pytest.fixture
def batched_data_list():
    data = np.random.uniform(0, 1, (100, 10))
    return [
        {"history": row, "history_mask": None, "target": None, "target_mask": None, "segment": i}
        for i, row in enumerate(data)
    ]


@pytest.mark.parametrize(
    "params,name,value,expected",
    ((None, "a", 1, {"a": 1}), ({"b": 3}, "a", "c", {"b": 3, "a": "c"}), ({"a": 1}, "a", 2, {"a": 2})),
)
def test_create_or_update(params, name, value, expected):
    res = _create_or_update(param=params, name=name, value=value)
    assert res == expected


@pytest.mark.parametrize("data", (np.array([1]), np.array([1, 2])))
def test_to_tensor(data):
    res = to_tensor(x=data)
    assert isinstance(res, torch.Tensor)


@pytest.mark.parametrize("window_sampling_limit", (None, 3))
def test_prepare_train_batch_format(
    batched_data_list,
    window_sampling_limit,
    input_size=7,
    output_size=3,
    expected_fields=("history", "history_mask", "target", "target_mask", "segment"),
):
    batch = prepare_train_batch(
        data=batched_data_list,
        input_size=input_size,
        output_size=output_size,
        window_sampling_limit=window_sampling_limit,
    )

    for field in expected_fields:
        assert field in batch

        if field != "segment":
            assert isinstance(batch[field], torch.Tensor)
            assert batch[field].shape[0] == len(batched_data_list)

        else:
            assert batch[field] is None


def test_prepare_test_batch_format(
    batched_data_list, input_size=15, expected_fields=("history", "history_mask", "target", "target_mask", "segment")
):
    batch = prepare_test_batch(data=batched_data_list, input_size=input_size)

    for field in expected_fields:
        assert field in batch

    assert isinstance(batch["history"], torch.Tensor)
    assert tuple(batch["history"].shape) == (len(batched_data_list), input_size)

    assert isinstance(batch["history_mask"], torch.Tensor)
    assert tuple(batch["history_mask"].shape) == (len(batched_data_list), input_size)

    assert batch["target"] is None
    assert batch["target_mask"] is None

    assert isinstance(batch["segment"], list)
    assert len(batch["segment"]) == len(batched_data_list)
