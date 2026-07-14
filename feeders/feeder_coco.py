import numpy as np
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self,
                 data_path,
                 label_path=None,
                 p_interval=1,
                 split='train',
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
                 retained_source_ids=None,
                 subset_mode='retain',
                 coord_jitter_sigma=0.0,
                 joint_dropout_prob=0.0,
                 score_scale_range=(1.0, 1.0)):

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
        self.retained_source_ids = [int(source_id) for source_id in (retained_source_ids or [])]
        self.subset_mode = subset_mode
        self.coord_jitter_sigma = float(coord_jitter_sigma)
        self.joint_dropout_prob = float(joint_dropout_prob)
        self.score_scale_range = tuple(score_scale_range)
        self.shadow_ood = self.subset_mode == 'shadow_ood'

        self.load_data()

        if normalization:
            self.get_mean_map()

    def load_data(self):
        npz_data = np.load(self.data_path, allow_pickle=True)

        if self.split == 'train':
            data = npz_data['x_train']
            label = npz_data['y_train']
        elif self.split == 'test':
            data = npz_data['x_test']
            label = npz_data['y_test']
        else:
            raise NotImplementedError('data split only supports train/test')

        raw_total = len(data)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        valid_index = []
        for index, sample in enumerate(data):
            valid_frame_num = np.count_nonzero(
                np.any(np.abs(sample) > 1e-6, axis=(0, 2, 3))
            )
            if valid_frame_num > 0:
                valid_index.append(index)

        data = data[valid_index]
        source_label = np.asarray(label[valid_index], dtype=np.int64)
        valid_total = len(data)

        if self.retained_source_ids:
            retained_ids = np.asarray(self.retained_source_ids, dtype=np.int64)
            retained_mask = np.isin(source_label, retained_ids)
            if self.subset_mode == 'retain':
                subset_mask = retained_mask
            elif self.subset_mode == 'shadow_ood':
                subset_mask = ~retained_mask
            elif self.subset_mode == 'all':
                subset_mask = np.ones_like(source_label, dtype=bool)
            else:
                raise ValueError('unsupported subset_mode: {}'.format(self.subset_mode))
            data = data[subset_mask]
            source_label = source_label[subset_mask]

        self.data = data
        self.source_label = np.asarray(source_label, dtype=np.int64)

        if self.retained_source_ids and self.subset_mode == 'retain':
            source_to_compact = {
                int(source_id): index for index, source_id in enumerate(self.retained_source_ids)
            }
            self.label = np.asarray(
                [source_to_compact[int(source_id)] for source_id in self.source_label],
                dtype=np.int64
            )
        elif self.shadow_ood:
            self.label = np.zeros(len(self.source_label), dtype=np.int64)
        else:
            self.label = np.asarray(self.source_label, dtype=np.int64)

        self.sample_name = [f'{self.split}_{self.subset_mode}_{index}' for index in range(len(self.data))]
        print(
            f'{self.split}/{self.subset_mode}: raw_total={raw_total}, '
            f'valid_total={valid_total}, subset_total={len(self.data)}, '
            f'filtered_empty={raw_total - valid_total}, filtered_subset={valid_total - len(self.data)}'
        )

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            self.source_label = self.source_label[:100]
            self.sample_name = self.sample_name[:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def apply_deploy_noise(self, data_numpy):
        if self.split != 'train':
            return data_numpy

        channel_count = data_numpy.shape[0]
        if channel_count == 5:
            coordinate_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
            score_index = 4
        elif channel_count == 3:
            coordinate_indices = np.asarray([0, 1], dtype=np.int64)
            score_index = 2
        elif channel_count == 2:
            coordinate_indices = np.asarray([0, 1], dtype=np.int64)
            score_index = None
        else:
            raise ValueError(
                'deploy noise supports C=2 (x,y), C=3 (x,y,score), or '
                'C=5 (absolute_x,absolute_y,relative_x,relative_y,score); '
                'got C={}'.format(channel_count)
            )

        if score_index is not None:
            valid_mask = (data_numpy[score_index:score_index + 1] > 0).astype(np.float32)
        else:
            valid_mask = (
                np.any(np.abs(data_numpy[0:2]) > 1e-6, axis=0, keepdims=True)
            ).astype(np.float32)

        if self.coord_jitter_sigma > 0:
            coordinates = data_numpy[coordinate_indices]
            coord_noise = np.random.normal(
                loc=0.0,
                scale=self.coord_jitter_sigma,
                size=coordinates.shape
            ).astype(np.float32)
            data_numpy[coordinate_indices] = coordinates + coord_noise * valid_mask

        if self.joint_dropout_prob > 0:
            keep_mask = (
                np.random.rand(*data_numpy.shape[1:]) >= self.joint_dropout_prob
            ).astype(np.float32)
            data_numpy *= keep_mask[None, ...]

        if score_index is not None:
            score_low, score_high = self.score_scale_range
            if score_low != 1.0 or score_high != 1.0:
                score_scale = np.random.uniform(
                    low=score_low,
                    high=score_high,
                    size=data_numpy[score_index:score_index + 1].shape
                ).astype(np.float32)
                data_numpy[score_index:score_index + 1] *= score_scale
            data_numpy[score_index:score_index + 1] = np.clip(
                data_numpy[score_index:score_index + 1], 0.0, 1.0
            )

        return data_numpy

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]

        data_numpy = np.array(data_numpy, dtype=np.float32)
        data_numpy = np.nan_to_num(data_numpy, nan=0.0, posinf=0.0, neginf=0.0)

        valid_frame_num = np.count_nonzero(
            np.any(np.abs(data_numpy) > 1e-6, axis=(0, 2, 3))
        )
        if valid_frame_num <= 0:
            valid_frame_num = 1

        data_numpy = tools.valid_crop_resize(
            data_numpy,
            valid_frame_num,
            self.p_interval,
            self.window_size
        )
        data_numpy = self.apply_deploy_noise(data_numpy)

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        if self.bone:
            from .bone_pairs import coco_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in coco_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        if self.shadow_ood:
            return 0.0
        rank = score.argsort()
        hit_top_k = [label in rank[index, -top_k:] for index, label in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
