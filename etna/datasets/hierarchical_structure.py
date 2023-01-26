from collections import defaultdict
from itertools import chain
from queue import Queue
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

from etna.core import BaseMixin


class HierarchicalStructure(BaseMixin):
    """Represents hierarchical structure of TSDataset."""

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
        self.level_structure = level_structure
        self._hierarchy_root = self._find_tree_root(self.level_structure)
        self._num_nodes = self._find_num_nodes(self.level_structure)

        hierarchy_levels = self._find_hierarchy_levels()
        tree_depth = len(hierarchy_levels)

        self.level_names = self._get_level_names(level_names, tree_depth)
        self._level_series: Dict[str, Tuple[str, ...]] = {
            self.level_names[i]: tuple(hierarchy_levels[i]) for i in range(tree_depth)
        }
        self._level_to_index: Dict[str, int] = {self.level_names[i]: i for i in range(tree_depth)}

        self._segment_num_reachable_leafs: Dict[str, int] = self._get_num_reachable_leafs(hierarchy_levels)

        self._segment_to_level: Dict[str, str] = {
            segment: level for level in self._level_series for segment in self._level_series[level]
        }

    @staticmethod
    def _get_level_names(level_names: Optional[List[str]], tree_depth: int) -> List[str]:
        """Assign level names if not provided."""
        if level_names is None:
            level_names = [f"level_{i}" for i in range(tree_depth)]

        if len(level_names) != tree_depth:
            raise ValueError("Length of `level_names` must be equal to hierarchy tree depth!")

        return level_names

    @staticmethod
    def _find_tree_root(hierarchy_structure: Dict[str, List[str]]) -> str:
        """Find hierarchy top level (root of tree)."""
        children = set(chain(*hierarchy_structure.values()))
        parents = set(hierarchy_structure.keys())

        tree_roots = parents.difference(children)
        if len(tree_roots) != 1:
            raise ValueError("Invalid tree definition: unable to find root!")

        return tree_roots.pop()

    @staticmethod
    def _find_num_nodes(hierarchy_structure: Dict[str, List[str]]) -> int:
        """Count number of nodes in tree."""
        children = set(chain(*hierarchy_structure.values()))
        parents = set(hierarchy_structure.keys())

        num_nodes = len(children | parents)

        num_edges = sum(map(len, hierarchy_structure.values()))
        if num_edges != num_nodes - 1:
            raise ValueError("Invalid tree definition: invalid number of nodes and edges!")

        return num_nodes

    def _find_hierarchy_levels(self) -> Dict[int, List[str]]:
        """Traverse hierarchy tree to group segments into levels."""
        leaves_levels = set()
        levels = defaultdict(list)
        seen_nodes = {self._hierarchy_root}
        queue: Queue = Queue()
        queue.put((self._hierarchy_root, 0))
        while not queue.empty():
            node, level = queue.get()
            levels[level].append(node)
            child_nodes = self.level_structure.get(node, [])

            if len(child_nodes) == 0:
                leaves_levels.add(level)

            for adj_node in child_nodes:
                queue.put((adj_node, level + 1))
                seen_nodes.add(adj_node)

        if len(seen_nodes) != self._num_nodes:
            raise ValueError("Invalid tree definition: disconnected graph!")

        if len(leaves_levels) != 1:
            raise ValueError("All hierarchy tree leaves must be on the same level!")

        return levels

    def _get_num_reachable_leafs(self, hierarchy_levels: Dict[int, List[str]]) -> Dict[str, int]:
        """Compute subtree size for each node."""
        num_reachable_leafs: Dict[str, int] = dict()
        for level in sorted(hierarchy_levels.keys(), reverse=True):
            for node in hierarchy_levels[level]:
                if node in self.level_structure:
                    num_reachable_leafs[node] = sum(
                        num_reachable_leafs[child_node] for child_node in self.level_structure[node]
                    )

                else:
                    num_reachable_leafs[node] = 1

        return num_reachable_leafs

    def get_summing_matrix(self, target_level: str, source_level: str) -> csr_matrix:
        """Get summing matrix for transition from source level to target level.

        Generation algorithm is based on summing matrix structure. Number of 1 in such matrices equals to
        number of nodes on the source level. Each row of summing matrices has ones only for source level nodes that
        belongs to subtree rooted from corresponding target level node. BFS order of nodes on levels view simplifies
        algorithm to calculation necessary offsets for each row.

        Parameters
        ----------
        target_level:
            Name of target level.
        source_level:
            Name of source level.

        Returns
        -------
        :
            Summing matrix from source level to target level

        """
        try:
            target_idx = self._level_to_index[target_level]
            source_idx = self._level_to_index[source_level]
        except KeyError as e:
            raise ValueError(f"Invalid level name: {e.args[0]}")

        if target_idx > source_idx:
            raise ValueError("Target level must be higher or equal in hierarchy than source level!")

        target_level_segment = self.get_level_segments(target_level)
        source_level_segment = self.get_level_segments(source_level)
        summing_matrix = lil_matrix((len(target_level_segment), len(source_level_segment)), dtype="int32")

        current_source_segment_id = 0
        for current_target_segment_id, segment in enumerate(target_level_segment):
            num_reachable_leafs_left = self._segment_num_reachable_leafs[segment]

            while num_reachable_leafs_left > 0:
                source_segment = source_level_segment[current_source_segment_id]
                num_reachable_leafs_left -= self._segment_num_reachable_leafs[source_segment]
                summing_matrix[current_target_segment_id, current_source_segment_id] = 1
                current_source_segment_id += 1

        return summing_matrix.tocsr()

    def get_level_segments(self, level_name: str) -> List[str]:
        """Get all segments from particular level."""
        try:
            return list(self._level_series[level_name])
        except KeyError:
            raise ValueError(f"Invalid level name: {level_name}")

    def get_segment_level(self, segment: str) -> str:
        """Get level name for provided segment."""
        try:
            return self._segment_to_level[segment]
        except KeyError:
            raise ValueError(f"Segment {segment} is out of the hierarchy")

    def get_level_depth(self, level_name: str) -> int:
        """Get level depth in a hierarchy tree."""
        try:
            return self._level_to_index[level_name]
        except KeyError:
            raise ValueError(f"Invalid level name: {level_name}")
