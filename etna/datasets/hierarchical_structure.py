from collections import defaultdict
from itertools import chain
from queue import Queue
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import scipy
from scipy.sparse import lil_matrix

from etna.core import BaseMixin


class HierarchicalStructure(BaseMixin):
    """Represents hierarchical structure for provided hierarchical tree."""

    def __init__(self, level_structure: Dict[str, List[str]], level_names: Optional[List[str]] = None):
        """Init HierarchicalStructure.

        Parameters
        ----------
        level_structure:
            Adjacency list describing the structure of the hierarchy tree (i.e. {"total":["X", "Y"], "X":["a", "b"], "Y":["c", "d"]}).
        level_names:
            Names of levels in the hierarchy in the order from top to bottom (i.e. ["total", "category", "product"]).
            If None is passed, level names are generated automatically with structure "level_<level_index>".
        """
        self._hierarchy_root: Union[str, None] = None
        self._hierarchy_interm_nodes: Set[str] = set()
        self._hierarchy_leaves: Set[str] = set()

        self.level_structure: Dict[str, List[str]] = level_structure

        self._find_graph_structure(level_structure)
        hierarchy_levels = self._find_hierarchy_levels(level_structure)
        tree_depth = len(hierarchy_levels)

        if level_names is None:
            level_names = [f"level_{i}" for i in range(tree_depth)]

        if len(level_names) != tree_depth:
            raise ValueError("Length of `level_names` must be equal to hierarchy tree depth!")

        self.level_names: List[str] = level_names
        self._level_series: Dict[str, List[str]] = {level_names[i]: hierarchy_levels[i] for i in range(tree_depth)}
        self._level_to_index: Dict[str, int] = {level_names[i]: i for i in range(tree_depth)}

        self._sub_segment_size_map: Dict[str, int] = {k: len(v) for k, v in level_structure.items()}

        self._segment_to_level: Dict[str, str] = {
            segment: level for level in self._level_series for segment in self._level_series[level]
        }

    def _find_graph_structure(self, adj_list: Dict[str, List[str]]):
        """Find hierarchy top level (root of tree)."""
        children = set(chain(*adj_list.values()))
        parents = set(adj_list.keys())

        tree_roots = parents.difference(children)
        if len(tree_roots) != 1:
            raise ValueError("Invalid tree definition: unable to find root!")

        self._hierarchy_interm_nodes = parents & children
        self._hierarchy_leaves = children.difference(parents)

        tree_root = tree_roots.pop()
        self._hierarchy_root = tree_root

    def _find_hierarchy_levels(self, hierarchy_structure: Dict[str, List[str]]):
        """Traverse hierarchy tree to group segments into levels."""
        nodes: Set[str] = self._hierarchy_interm_nodes | self._hierarchy_leaves
        nodes.add(str(self._hierarchy_root))

        num_edges = sum(map(len, hierarchy_structure.values()))

        num_nodes = len(nodes)
        if num_edges != num_nodes - 1:
            raise ValueError("Invalid tree definition: invalid number of nodes and edges!")

        leaves_level = None
        node_levels = []
        seen_nodes = {self._hierarchy_root}
        queue: Queue = Queue()
        queue.put((self._hierarchy_root, 0))
        while not queue.empty():
            node, level = queue.get()
            node_levels.append((level, node))
            child_nodes = hierarchy_structure.get(node, [])

            if len(child_nodes) == 0:
                if leaves_level is not None and level != leaves_level:
                    raise ValueError("All hierarchy tree leaves must be on the same level!")
                else:
                    leaves_level = level

            for adj_node in child_nodes:
                if adj_node not in seen_nodes:
                    queue.put((adj_node, level + 1))
                    seen_nodes.add(adj_node)

        if len(seen_nodes) != num_nodes:
            raise ValueError("Invalid tree definition: disconnected graph!")

        levels = defaultdict(list)
        for level, node in node_levels:
            levels[level].append(node)

        return levels

    def get_summing_matrix(self, target_level: str, source_level: str) -> scipy.sparse.base.spmatrix:
        """
        Get summing matrix for transition from source level to target level.

        Parameters
        ----------
        target_level:
            Name of target level.
        source_level:
            Name of source level.

        Returns
        -------
        :
            transition matrix from source level to target level

        """
        try:
            target_idx = self._level_to_index[target_level]
            source_idx = self._level_to_index[source_level]
        except KeyError as e:
            raise ValueError("Invalid level name: " + e.args[0])

        if target_idx >= source_idx:
            raise ValueError("Target level must be higher in hierarchy than source level!")

        level_names = self.level_names
        summing_matrix = None
        for i in range(target_idx, source_idx):
            top_level = level_names[i]
            bottom_level = level_names[i + 1]

            matrix = lil_matrix((len(self.get_level_segments(top_level)), len(self.get_level_segments(bottom_level))))

            offset = 0
            for i, segment in enumerate(self.get_level_segments(top_level)):
                sub_segment_size = self._sub_segment_size_map.get(segment, 0)
                for j in range(sub_segment_size):
                    matrix[i, offset + j] = 1
                offset += sub_segment_size

            matrix.tocsr()
            if summing_matrix is None:
                summing_matrix = matrix
            else:
                summing_matrix = summing_matrix @ matrix

        return summing_matrix

    def get_level_segments(self, level_name: str) -> List[str]:
        """Get all segments from particular level."""
        try:
            return self._level_series[level_name]
        except KeyError as e:
            raise ValueError("Invalid level name: " + e.args[0])

    def get_segment_level(self, segment: str) -> Union[str, None]:
        """Get level name for provided segment."""
        try:
            return self._segment_to_level[segment]
        except KeyError:
            return None
