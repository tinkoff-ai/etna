import pytest

from etna.datasets.hierarchical_structure import HierarchicalStructure


@pytest.fixture
def long_hierarchical_structure():
    hs = HierarchicalStructure(
        level_structure={"total": ["X", "Y"], "X": ["a"], "a": ["c"], "Y": ["b"], "b": ["d"]},
        level_names=["l1", "l2", "l3", "l4"],
    )
    return hs


@pytest.fixture
def tailed_hierarchical_structure():
    hs = HierarchicalStructure(
        level_structure={"total": ["X", "Y"], "X": ["a"], "Y": ["c", "d"], "c": ["f"], "d": ["g"], "a": ["e", "h"]},
        level_names=["l1", "l2", "l3", "l4"],
    )
    return hs
