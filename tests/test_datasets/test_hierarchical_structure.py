from typing import Dict
from typing import List

import numpy as np
import pytest
import scipy.sparse

from etna.datasets import HierarchicalStructure


@pytest.fixture
def simple_hierarchical_struct():
    return HierarchicalStructure(
        level_structure={"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, level_names=["l1", "l2", "l3"]
    )


@pytest.fixture
def tailed_hierarchical_struct():
    return HierarchicalStructure(
        level_structure={"total": ["X", "Y"], "X": ["a"], "Y": ["c", "d"], "c": ["f"], "d": ["g"], "a": ["e", "h"]},
        level_names=["l1", "l2", "l3", "l4"],
    )


@pytest.fixture
def long_hierarchical_struct():
    return HierarchicalStructure(
        level_structure={"total": ["X", "Y"], "X": ["a"], "Y": ["b"], "a": ["c"], "b": ["d"]},
        level_names=["l1", "l2", "l3", "l4"],
    )


@pytest.mark.parametrize(
    "target,source",
    (
        ("l1", "l2"),
        ("l2", "l3"),
    ),
)
def test_get_summing_matrix_format(simple_hierarchical_struct: HierarchicalStructure, source: str, target: str):
    output = simple_hierarchical_struct.get_summing_matrix(target_level=target, source_level=source)
    assert isinstance(output, scipy.sparse.base.spmatrix)


@pytest.mark.parametrize(
    "struct, target,source,answer",
    (
        ("tailed_hierarchical_struct", "l1", "l2", np.array([[1, 1]])),
        ("tailed_hierarchical_struct", "l1", "l3", np.array([[1, 1, 1]])),
        ("tailed_hierarchical_struct", "l1", "l4", np.array([[1, 1, 1, 1]])),
        ("tailed_hierarchical_struct", "l2", "l3", np.array([[1, 0, 0], [0, 1, 1]])),
        ("tailed_hierarchical_struct", "l2", "l4", np.array([[1, 1, 0, 0], [0, 0, 1, 1]])),
        ("tailed_hierarchical_struct", "l3", "l4", np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
        ("simple_hierarchical_struct", "l1", "l2", np.array([[1, 1]])),
        ("simple_hierarchical_struct", "l1", "l3", np.array([[1, 1, 1, 1]])),
        ("simple_hierarchical_struct", "l2", "l3", np.array([[1, 1, 0, 0], [0, 0, 1, 1]])),
        ("long_hierarchical_struct", "l1", "l2", np.array([[1, 1]])),
        ("long_hierarchical_struct", "l1", "l3", np.array([[1, 1]])),
        ("long_hierarchical_struct", "l1", "l4", np.array([[1, 1]])),
        ("long_hierarchical_struct", "l2", "l3", np.array([[1, 0], [0, 1]])),
        ("long_hierarchical_struct", "l2", "l4", np.array([[1, 0], [0, 1]])),
        ("long_hierarchical_struct", "l3", "l4", np.array([[1, 0], [0, 1]])),
    ),
)
def test_summing_matrix(struct: str, source: str, target: str, answer: np.ndarray, request: pytest.FixtureRequest):
    np.testing.assert_array_almost_equal(
        answer, request.getfixturevalue(struct).get_summing_matrix(target_level=target, source_level=source).toarray()
    )


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
        simple_hierarchical_struct.get_summing_matrix(target_level=target, source_level=source)


@pytest.mark.parametrize(
    "structure,answer",
    (
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, "total"),
        ({"X": ["a", "b"]}, "X"),
    ),
)
def test_root_finding(structure: Dict[str, List[str]], answer: str):
    h = HierarchicalStructure(level_structure=structure)
    assert h._hierarchy_root == answer


@pytest.mark.parametrize(
    "structure",
    (
        {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d", "total"]},  # loop to root
        {"X": ["a", "b"], "Y": ["c", "d"]},  # 2 trees
        dict(),  # empty list
    ),
)
def test_root_finding_errors(structure: Dict[str, List[str]]):
    with pytest.raises(ValueError, match="Invalid tree definition: unable to find root!"):
        HierarchicalStructure(level_structure=structure)


@pytest.mark.parametrize(
    "structure",
    (
        {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"], "a": ["X"]},  # loop
        {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d", "Y"]},  # self loop
    ),
)
def test_invalid_tree_structure_initialization_fails(structure: Dict[str, List[str]]):
    with pytest.raises(ValueError, match="Invalid tree definition: invalid number of nodes and edges!"):
        HierarchicalStructure(level_structure=structure)


@pytest.mark.parametrize(
    "structure,names,answer",
    (
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, None, ["level_0", "level_1", "level_2"]),
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, ["l1", "l2", "l3"], ["l1", "l2", "l3"]),
    ),
)
def test_level_names(structure: Dict[str, List[str]], names: List[str], answer: List[str]):
    h = HierarchicalStructure(level_structure=structure, level_names=names)
    assert h.level_names == answer


@pytest.mark.parametrize(
    "structure,names",
    (
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, ["l1"]),
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, ["l1", "l2", "l3", "l4"]),
    ),
)
def test_level_names_length_error(structure: Dict[str, List[str]], names: List[str]):
    with pytest.raises(ValueError, match="Length of `level_names` must be equal to hierarchy tree depth!"):
        HierarchicalStructure(level_structure=structure, level_names=names)


@pytest.mark.parametrize(
    "level,answer",
    (("l1", ["total"]), ("l2", ["X", "Y"]), ("l3", ["a", "b", "c", "d"])),
)
def test_level_segments(simple_hierarchical_struct: HierarchicalStructure, level: str, answer: List[str]):
    assert simple_hierarchical_struct.get_level_segments(level) == answer


@pytest.mark.parametrize(
    "segment,answer",
    (("total", "l1"), ("Y", "l2"), ("c", "l3")),
)
def test_segments_level(simple_hierarchical_struct: HierarchicalStructure, segment: str, answer: str):
    assert simple_hierarchical_struct.get_segment_level(segment) == answer


@pytest.mark.parametrize(
    "structure",
    (
        {"total": ["X", "Y"], "X": ["a"], "Y": ["c", "d"], "c": ["e", "f"]},  # e f leaves have lower level
        {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"], "a": ["e"]},  # e has lower level
    ),
)
def test_leaves_level_errors(structure: Dict[str, List[str]]):
    with pytest.raises(ValueError, match="All hierarchy tree leaves must be on the same level!"):
        HierarchicalStructure(level_structure=structure)
