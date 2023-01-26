from typing import Dict
from typing import List

import numpy as np
import pytest

from etna.datasets import HierarchicalStructure


@pytest.fixture
def simple_hierarchical_structure():
    return HierarchicalStructure(
        level_structure={"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, level_names=["l1", "l2", "l3"]
    )


@pytest.mark.parametrize(
    "struct, target,source,answer",
    (
        ("simple_hierarchical_structure", "l1", "l1", np.array([[1]])),
        ("simple_hierarchical_structure", "l2", "l2", np.array([[1, 0], [0, 1]])),
        ("simple_hierarchical_structure", "l1", "l2", np.array([[1, 1]])),
        ("simple_hierarchical_structure", "l1", "l3", np.array([[1, 1, 1, 1]])),
        ("simple_hierarchical_structure", "l2", "l3", np.array([[1, 1, 0, 0], [0, 0, 1, 1]])),
        ("tailed_hierarchical_structure", "l3", "l3", np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        ("tailed_hierarchical_structure", "l1", "l2", np.array([[1, 1]])),
        ("tailed_hierarchical_structure", "l1", "l3", np.array([[1, 1, 1]])),
        ("tailed_hierarchical_structure", "l1", "l4", np.array([[1, 1, 1, 1]])),
        ("tailed_hierarchical_structure", "l2", "l3", np.array([[1, 0, 0], [0, 1, 1]])),
        ("tailed_hierarchical_structure", "l2", "l4", np.array([[1, 1, 0, 0], [0, 0, 1, 1]])),
        ("tailed_hierarchical_structure", "l3", "l4", np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
        ("long_hierarchical_structure", "l1", "l2", np.array([[1, 1]])),
        ("long_hierarchical_structure", "l1", "l3", np.array([[1, 1]])),
        ("long_hierarchical_structure", "l1", "l4", np.array([[1, 1]])),
        ("long_hierarchical_structure", "l2", "l3", np.array([[1, 0], [0, 1]])),
        ("long_hierarchical_structure", "l2", "l4", np.array([[1, 0], [0, 1]])),
        ("long_hierarchical_structure", "l3", "l4", np.array([[1, 0], [0, 1]])),
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
        ("l2", "l1", "Target level must be higher or equal in hierarchy than source level!"),
    ),
)
def test_level_transition_errors(
    simple_hierarchical_structure: HierarchicalStructure,
    target: str,
    source: str,
    error: str,
):
    with pytest.raises(ValueError, match=error):
        simple_hierarchical_structure.get_summing_matrix(target_level=target, source_level=source)


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


@pytest.mark.parametrize(
    "structure,answer",
    (
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, "total"),
        ({"X": ["a", "b"]}, "X"),
    ),
)
def test_root_finding(structure: Dict[str, List[str]], answer: str):
    assert HierarchicalStructure._find_tree_root(structure) == answer


@pytest.mark.parametrize(
    "structure,answer",
    (
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, 7),
        ({"X": ["a", "b"]}, 3),
    ),
)
def test_num_nodes(structure: Dict[str, List[str]], answer: int):
    assert HierarchicalStructure._find_num_nodes(structure) == answer


@pytest.mark.parametrize(
    "level_names,tree_depth,answer",
    (
        (None, 3, ["level_0", "level_1", "level_2"]),
        (["l1", "l2", "l3", "l4"], 4, ["l1", "l2", "l3", "l4"]),
    ),
)
def test_get_level_names(level_names: List[str], tree_depth: int, answer: List[str]):
    assert HierarchicalStructure._get_level_names(level_names, tree_depth) == answer


@pytest.mark.parametrize(
    "structure,answer",
    (
        ({"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]}, [["total"], ["X", "Y"], ["a", "b", "c", "d"]]),
        ({"X": ["a", "b"]}, [["X"], ["a", "b"]]),
    ),
)
def test_find_hierarchy_levels(structure: Dict[str, List[str]], answer: List[List[str]]):
    h = HierarchicalStructure(level_structure=structure)
    hierarchy_levels = h._find_hierarchy_levels()
    for i, level_segments in enumerate(answer):
        assert hierarchy_levels[i] == level_segments


@pytest.mark.parametrize(
    "structure,answer",
    (
        (
            {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]},
            {"total": 4, "X": 2, "Y": 2, "a": 1, "b": 1, "c": 1, "d": 1},
        ),
        ({"total": ["X", "Y"], "X": ["a"], "Y": ["c", "d"]}, {"total": 3, "X": 1, "Y": 2, "a": 1, "c": 1, "d": 1}),
        ({"X": ["a", "b"]}, {"X": 2, "a": 1, "b": 1}),
    ),
)
def test_get_num_reachable_leafs(structure: Dict[str, List[str]], answer: Dict[str, int]):
    h = HierarchicalStructure(level_structure=structure)
    hierarchy_levels = h._find_hierarchy_levels()
    reachable_leafs = h._get_num_reachable_leafs(hierarchy_levels)
    assert len(reachable_leafs) == len(answer)
    for segment in answer:
        assert reachable_leafs[segment] == answer[segment]


@pytest.mark.parametrize(
    "structure,level_names,answer",
    (
        (
            {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]},
            None,
            {"level_0": 0, "level_1": 1, "level_2": 2},
        ),
        (
            {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]},
            ["l1", "l2", "l3"],
            {"l1": 0, "l2": 1, "l3": 2},
        ),
        (
            {"X": ["a"]},
            None,
            {"level_0": 0, "level_1": 1},
        ),
    ),
)
def test_level_to_index(structure: Dict[str, List[str]], level_names: List[str], answer: Dict[str, int]):
    h = HierarchicalStructure(level_structure=structure, level_names=level_names)
    assert len(h._level_to_index) == len(answer)
    for level in answer:
        assert h._level_to_index[level] == answer[level]


@pytest.mark.parametrize(
    "structure,answer",
    (
        (
            {"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]},
            {
                "total": "level_0",
                "X": "level_1",
                "Y": "level_1",
                "a": "level_2",
                "b": "level_2",
                "c": "level_2",
                "d": "level_2",
            },
        ),
        ({"X": ["a"]}, {"X": "level_0", "a": "level_1"}),
    ),
)
def test_segment_to_level(structure: Dict[str, List[str]], answer: Dict[str, str]):
    h = HierarchicalStructure(level_structure=structure)
    assert len(h._segment_to_level) == len(answer)
    for segment in answer:
        assert h._segment_to_level[segment] == answer[segment]


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
    (
        ("l1", ["total"]),
        ("l2", ["X", "Y"]),
        ("l3", ["a", "b", "c", "d"]),
    ),
)
def test_level_segments(simple_hierarchical_structure: HierarchicalStructure, level: str, answer: List[str]):
    assert simple_hierarchical_structure.get_level_segments(level) == answer


@pytest.mark.parametrize(
    "segment,answer",
    (("total", "l1"), ("Y", "l2"), ("c", "l3")),
)
def test_segments_level(simple_hierarchical_structure: HierarchicalStructure, segment: str, answer: str):
    assert simple_hierarchical_structure.get_segment_level(segment) == answer


@pytest.mark.parametrize(
    "target_level,answer",
    (("l2", 1), ("l3", 2), ("l1", 0)),
)
def test_get_level_depth(simple_hierarchical_structure, target_level, answer):
    assert simple_hierarchical_structure.get_level_depth(level_name=target_level) == answer


@pytest.mark.parametrize(
    "target_level",
    ("", "abcd"),
)
def test_get_level_depth_invalid_name_error(simple_hierarchical_structure, target_level):
    with pytest.raises(ValueError, match=f"Invalid level name: {target_level}"):
        simple_hierarchical_structure.get_level_depth(level_name=target_level)
