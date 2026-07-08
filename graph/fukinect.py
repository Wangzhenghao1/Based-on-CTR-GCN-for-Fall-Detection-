import sys

sys.path.extend(["../"])
from graph import tools


num_node = 20
self_link = [(i, i) for i in range(num_node)]

# Kinect v1 20-joint layout used by FUKinect_Fall:
# 1 hip_center, 2 spine, 3 shoulder_center, 4 head,
# 5-8 left arm, 9-12 right arm, 13-16 left leg, 17-20 right leg.
inward_ori_index = [
    (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7), (7, 8),
    (3, 9), (9, 10), (10, 11), (11, 12),
    (1, 13), (13, 14), (14, 15), (15, 16),
    (1, 17), (17, 18), (18, 19), (19, 20),
]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode="spatial"):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            return tools.get_spatial_graph(num_node, self_link, inward, outward)
        raise ValueError()
