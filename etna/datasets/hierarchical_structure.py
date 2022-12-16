from collections import defaultdict
from queue import Queue
from typing import Dict
from typing import List
from typing import Optional
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
            Adjacency list describing the structure of the hierarchy tree (i.e. {"total":["X", "Y"], "X":["a", "b"], "Y":["c", "d"]})
        level_names:
            Names of levels in the hierarchy in the order from top to bottom (i.e. ["total", "category", "product"]), if None is passed, generate level names
        """
        hierarchy_root = self._find_tree_root(level_structure)
        hierarchy_levels = self._find_hierarchy_levels(hierarchy_root, level_structure)
        tree_depth = len(hierarchy_levels)

        if level_names is None:
            level_names = [f"level_{i}" for i in range(tree_depth)]

        if len(level_names) != tree_depth:
            raise ValueError("Length of `level_names` must be equal to hierarchy tree depth!")

        self._level_series = dict()
        self._level_index_map = dict()
        for i in range(tree_depth):
            self._level_index_map[level_names[i]] = i
            self._level_series[level_names[i]] = hierarchy_levels[i]

        self._sub_segment_size_map = {k: len(v) for k, v in level_structure.items()}

        self._segment_levels_map = dict()
        for level in self._level_series:
            for segment in self._level_series[level]:
                self._segment_levels_map[segment] = level

    @staticmethod
    def _find_tree_root(adj_list: Dict[str, List[str]]):
        """Find hierarchy top level (root of tree)."""
        children = set()
        parents = set(adj_list.keys())
        for adj_nodes in adj_list.values():
            children |= set(adj_nodes)

        top_nodes = parents.difference(children)
        if len(top_nodes) != 1:
            raise ValueError("Invalid tree definition: unable to find root!")

        tree_root = top_nodes.pop()
        return tree_root

    @staticmethod
    def _find_hierarchy_levels(hierarchy_root: str, hierarchy_structure: Dict[str, List[str]]):
        """Traverse hierarchy tree to group segments into levels."""
        num_edges = 0
        nodes = set(hierarchy_structure.keys())
        for node_list in hierarchy_structure.values():
            nodes |= set(node_list)
            num_edges += len(node_list)

        leaves = {n for n in nodes if n not in hierarchy_structure}

        num_nodes = len(nodes)
        if num_edges != num_nodes - 1:
            raise ValueError("Invalid tree definition: invalid number of nodes and edges!")

        node_levels = []
        seen_nodes = {hierarchy_root}
        queue: Queue = Queue()
        queue.put((hierarchy_root, 0))
        while not queue.empty():
            node, level = queue.get()
            node_levels.append((level, node))
            for adj_node in hierarchy_structure.get(node, []):
                if adj_node not in seen_nodes:
                    queue.put((adj_node, level + 1))
                    seen_nodes.add(adj_node)

        if len(seen_nodes) != num_nodes:
            raise ValueError("Invalid tree definition: disconnected graph!")

        leaves_levels = set()
        levels = defaultdict(list)
        for level, node in node_levels:
            levels[level].append(node)
            if node in leaves:
                leaves_levels.add(level)

        if len(leaves_levels) != 1:
            raise ValueError("All hierarchy tree leaves must be on the same level!")

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
            target_idx = self._level_index_map[target_level]
            source_idx = self._level_index_map[source_level]
        except KeyError as e:
            raise ValueError("Invalid level name: " + e.args[0])

        if target_idx >= source_idx:
            raise ValueError("Target level must be higher in hierarchy than source level!")

        level_names = self.level_names
        transition_matrix = None
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
            if transition_matrix is None:
                transition_matrix = matrix
            else:
                transition_matrix = transition_matrix @ matrix

        return transition_matrix

    @property
    def level_names(self) -> List[str]:
        """Get all levels names."""
        return sorted(self._level_index_map.keys(), key=lambda l: self._level_index_map[l])

    def get_level_segments(self, level_name: str) -> List[str]:
        """Get all segments from particular level."""
        try:
            return self._level_series[level_name]
        except KeyError as e:
            raise ValueError("Invalid level name: " + e.args[0])

    def get_segment_level(self, segment: str) -> Union[str, None]:
        """Get level name for provided segment."""
        try:
            return self._segment_levels_map[segment]
        except KeyError:
            return None
