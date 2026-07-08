import numpy as np
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        label_path=None,
        p_interval=1,
        split="train",
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,
        window_size=-1,
        normalization=False,
        debug=False,
        use_mmap=False,
        bone=False,
        vel=False,
    ):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        npz_data = np.load(self.data_path, allow_pickle=True)
        data_key = "x_{}".format(self.split)
        label_key = "y_{}".format(self.split)
        if data_key not in npz_data or label_key not in npz_data:
            raise ValueError("missing {} or {} in {}".format(data_key, label_key, self.data_path))

        data = np.asarray(npz_data[data_key], dtype=np.float32)
        label = npz_data[label_key]
        if label.ndim > 1:
            label = np.where(label > 0)[1]
        label = np.asarray(label, dtype=np.int64)

        if data.ndim == 3 and data.shape[2] == 60:
            n, t, _ = data.shape
            data = data.reshape(n, t, 20, 3).transpose(0, 3, 1, 2)[:, :, :, :, None]
        if data.ndim != 5:
            raise ValueError("expected N,C,T,V,M data, got {}".format(data.shape))

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        valid_index = []
        for index, sample in enumerate(data):
            valid_frame_num = np.count_nonzero(np.any(np.abs(sample) > 1e-6, axis=(0, 2, 3)))
            if valid_frame_num > 0:
                valid_index.append(index)

        self.data = data[valid_index]
        self.label = label[valid_index]
        self.sample_name = ["{}_{}".format(self.split, index) for index in valid_index]

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            self.sample_name = self.sample_name[:100]

    def get_mean_map(self):
        data = self.data
        n, c, t, v, m = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((n * t * m, c * v)).std(axis=0).reshape((c, 1, v, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index], dtype=np.float32)
        label = int(self.label[index])

        valid_frame_num = np.count_nonzero(np.any(np.abs(data_numpy) > 1e-6, axis=(0, 2, 3)))
        if valid_frame_num <= 0:
            valid_frame_num = 1

        data_numpy = tools.valid_crop_resize(
            data_numpy,
            valid_frame_num,
            self.p_interval,
            self.window_size,
        )
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import fukinect_pairs

            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in fukinect_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [label in rank[index, -top_k:] for index, label in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
