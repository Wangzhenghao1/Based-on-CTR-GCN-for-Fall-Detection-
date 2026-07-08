import sys

sys.path.extend(['../'])
from graph import tools

num_node = 17


class Graph:
    def __init__(self, CoM=12, labeling_mode='spatial'):
        self.num_node = num_node
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_hierarchical_graph(
                num_node,
                tools.get_edgeset(dataset='COCO', CoM=self.CoM)
            )
        else:
            raise ValueError()
        return A, self.CoM, 'COCO'
