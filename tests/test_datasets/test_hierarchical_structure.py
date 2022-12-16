from typing import Dict
from typing import List

import numpy as np
import pytest

from etna.datasets import HierarchicalStructure


@pytest.fixture
def simple_hierarchical_struct():
    return HierarchicalStructure({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, ["l1", "l2", "l3"])


@pytest.fixture
def tailed_hierarchical_struct():
    return HierarchicalStructure(
        {"total": ["X", "Y"], "X": ["a"], "Y": ["c", "d"], "c": ["f"], "d": ["g"], "a": ["e", "h"]},
        ["l1", "l2", "l3", "l4"],
    )


@pytest.mark.parametrize(
    "target,source,answer",
    (
        ("l1", "l2", np.array([[1, 1]])),
        ("l1", "l4", np.array([[1, 1, 1, 1]])),
        ("l2", "l3", np.array([[1, 0, 0], [0, 1, 1]])),
    ),
)
def test_tailed_struct_matrix(
    tailed_hierarchical_struct: HierarchicalStructure, source: str, target: str, answer: np.ndarray
):
    np.testing.assert_array_almost_equal(
        answer, tailed_hierarchical_struct.get_summing_matrix(target, source).toarray()
    )


@pytest.mark.parametrize(
    "target,source,answer",
    (
        ("l1", "l2", np.array([[1, 1]])),
        ("l1", "l3", np.array([[1, 1, 1, 1]])),
        ("l2", "l3", np.array([[1, 1, 0, 0], [0, 0, 1, 1]])),
    ),
)
def test_simple_struct_matrix(
    simple_hierarchical_struct: HierarchicalStructure, source: str, target: str, answer: np.ndarray
):
    np.testing.assert_array_almost_equal(
        answer, simple_hierarchical_struct.get_summing_matrix(target, source).toarray()
    )


@pytest.mark.parametrize(
    "structure",
    (
        {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d", "total"]},
        {"X": ["a", "b"], "Y": ["c", "d"]},
        dict(),
    ),
)
def test_root_finding_errors(structure: Dict[str, List[str]]):
    with pytest.raises(ValueError, match="Invalid tree definition: unable to find root!"):
        HierarchicalStructure(structure)


@pytest.mark.parametrize(
    "structure",
    (
        {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"], "a": ["X"]},
        {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d", "Y"]},
    ),
)
def test_tree_structure_errors(structure: Dict[str, List[str]]):
    with pytest.raises(ValueError, match="Invalid tree definition: invalid number of nodes and edges!"):
        HierarchicalStructure(structure)


@pytest.mark.parametrize(
    "structure,names,answer",
    (
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, None, ["level_0", "level_1", "level_2"]),
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, ["l1", "l2", "l3"], ["l1", "l2", "l3"]),
    ),
)
def test_level_names(structure: Dict[str, List[str]], names: List[str], answer: List[str]):
    h = HierarchicalStructure(structure, names)
    for name, correct in zip(h.level_names, answer):
        assert name == correct


@pytest.mark.parametrize(
    "level,answer",
    (
        ("l1", ["total"]),
        ("l2", ["X", "Y"]),
        ("l3", ["a", "b", "c", "d"])
    ),
)
def test_level_segments(simple_hierarchical_struct: HierarchicalStructure, level: str, answer: List[str]):
    for name, correct in zip(simple_hierarchical_struct.get_level_segments(level), answer):
        assert name == correct


@pytest.mark.parametrize(
    "segment,answer",
    (
        ("total", "l1"),
        ("Y", "l2"),
        ("c", "l3")
    ),
)
def test_level_of_segments(simple_hierarchical_struct: HierarchicalStructure, segment: str, answer: str):
    assert simple_hierarchical_struct.get_level_of_segment(segment) == answer


@pytest.mark.parametrize(
    "structure,names",
    (
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, ["l1"]),
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, ["l1", "l2", "l3", "l4"]),
    ),
)
def test_level_names_errors(structure: Dict[str, List[str]], names: List[str]):
    with pytest.raises(ValueError, match="Length of `level_names` must be equal to hierarchy tree depth!"):
        HierarchicalStructure(structure, names)


@pytest.mark.parametrize(
    "structure",
    (
        {"total": ["X", "Y"], "X": ["a"], "Y": ["c", "d"], "c": ["e", "f"]},
        {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"], "a": ["e"]},
    ),
)
def test_leaves_level_errors(structure: Dict[str, List[str]]):
    with pytest.raises(ValueError, match="All hierarchy tree leaves must be on the same level!"):
        HierarchicalStructure(structure)


@pytest.mark.parametrize(
    "target,source,error",
    (
        ("l0", "l2", "Invalid level name: l0"),
        ("l1", "l0", "Invalid level name: l0"),
        ("l2", "l1", "Target level must be higher in hierarchy than source level!"),
    ),
)
def test_level_transition_errors(
    simple_hierarchical_struct: HierarchicalStructure,
    target: str,
    source: str,
    error: str,
):
    with pytest.raises(ValueError, match=error):
        simple_hierarchical_struct.get_summing_matrix(target, source)
