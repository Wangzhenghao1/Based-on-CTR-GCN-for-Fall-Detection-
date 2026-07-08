import sys
sys.path.extend(['../'])
from graph import tools

num_node = 17
self_link = [(i, i) for i in range(num_node)]

# COCO17 skeleton
# 0 nose
# 1 left_eye
# 2 right_eye
# 3 left_ear
# 4 right_ear
# 5 left_shoulder
# 6 right_shoulder
# 7 left_elbow
# 8 right_elbow
# 9 left_wrist
# 10 right_wrist
# 11 left_hip
# 12 right_hip
# 13 left_knee
# 14 right_knee
# 15 left_ankle
# 16 right_ankle

inward_ori_index = [
    (1, 2), (1, 3),
    (2, 4), (3, 5),
    (1, 6), (1, 7),
    (6, 8), (8, 10),
    (7, 9), (9, 11),
    (6, 12), (7, 13),
    (12, 14), (14, 16),
    (13, 15), (15, 17)
]

# 转成 0-based
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A