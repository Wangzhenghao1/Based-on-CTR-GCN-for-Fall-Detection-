import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.lib import format as npy_format


COCO17_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
]


TARGET_CLASSES = {
    32: "running",
    36: "walking",
    37: "jogging",
    79: "squat_down",
    42: "falling_down",
}


class NpzArrayReader:
    def __init__(self, npz_path, array_name):
        self.zip_file = zipfile.ZipFile(npz_path)
        self.member_name = array_name if array_name.endswith(".npy") else array_name + ".npy"
        self.file_obj = self.zip_file.open(self.member_name)
        version = npy_format.read_magic(self.file_obj)
        if version == (1, 0):
            self.shape, self.fortran_order, self.dtype = npy_format.read_array_header_1_0(self.file_obj)
        elif version == (2, 0):
            self.shape, self.fortran_order, self.dtype = npy_format.read_array_header_2_0(self.file_obj)
        else:
            raise ValueError("unsupported npy version: {}".format(version))
        if self.fortran_order:
            raise ValueError("Fortran-order arrays are not supported")
        self.data_offset = self.file_obj.tell()
        self.sample_shape = self.shape[1:]
        self.sample_nbytes = int(np.prod(self.sample_shape) * np.dtype(self.dtype).itemsize)

    def read_sample(self, index):
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)
        self.file_obj.seek(self.data_offset + index * self.sample_nbytes)
        raw = self.file_obj.read(self.sample_nbytes)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.sample_shape).copy()

    def close(self):
        self.file_obj.close()
        self.zip_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize selected COCO17 npz samples as GIF files.")
    parser.add_argument(
        "--npz",
        default=r"D:\Server_download\ntu120_coco17_1file\fall_coco_walk_replace.npz",
        help="Path to fall_coco-style npz.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/skeleton_gifs",
        help="Directory for GIF outputs.",
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--duration-ms", type=int, default=80)
    return parser.parse_args()


def load_labels(npz_path, split):
    with zipfile.ZipFile(npz_path) as zip_file:
        with zip_file.open("y_{}.npy".format(split)) as file_obj:
            return np.load(file_obj)


def visible_mask(sample, score_thr=0.05):
    x = sample[0]
    y = sample[1]
    score = sample[2]
    return (score > score_thr) & ((np.abs(x) + np.abs(y)) > 1e-6)


def motion_score(sample):
    mask = visible_mask(sample)
    centers = []
    heights = []
    for frame_index in range(sample.shape[1]):
        best_person = None
        best_visible = 0
        for person_index in range(sample.shape[3]):
            person_mask = mask[frame_index, :, person_index]
            count = int(person_mask.sum())
            if count > best_visible:
                best_visible = count
                best_person = person_index
        if best_person is None or best_visible < 5:
            continue
        person_mask = mask[frame_index, :, best_person]
        xy = np.stack([
            sample[0, frame_index, person_mask, best_person],
            sample[1, frame_index, person_mask, best_person],
        ], axis=1)
        centers.append(xy.mean(axis=0))
        heights.append(float(xy[:, 1].max() - xy[:, 1].min()))
    if len(centers) < 2:
        return -1.0
    centers = np.asarray(centers)
    heights = np.asarray(heights)
    bbox_w = max(float(sample[0][mask].max() - sample[0][mask].min()), 1.0)
    bbox_h = max(float(sample[1][mask].max() - sample[1][mask].min()), 1.0)
    diag = max(float(np.hypot(bbox_w, bbox_h)), 1.0)
    center_motion = np.abs(np.diff(centers, axis=0)).sum() / diag
    height_change = (heights.max() - heights.min()) / max(bbox_h, 1.0)
    visibility = float(mask.sum()) / float(np.prod(mask.shape))
    return float(center_motion + 2.0 * height_change + visibility)


def person_activity(sample, person_index):
    mask = visible_mask(sample)[:, :, person_index]
    visible_frames = 0
    centers = []
    for frame_index in range(sample.shape[1]):
        frame_mask = mask[frame_index]
        if frame_mask.sum() < 5:
            continue
        xy = np.stack([
            sample[0, frame_index, frame_mask, person_index],
            sample[1, frame_index, frame_mask, person_index],
        ], axis=1)
        centers.append(xy.mean(axis=0))
        visible_frames += 1

    if visible_frames < 2:
        return 0, 999.0, 0

    centers = np.asarray(centers, dtype=np.float32)
    activity = float(np.sqrt((np.diff(centers, axis=0) ** 2).sum(axis=1)).sum())
    limb_lengths = []
    for start, end in COCO17_EDGES:
        ok = mask[:, start] & mask[:, end]
        if ok.sum() < 8:
            continue
        dx = sample[0, ok, start, person_index] - sample[0, ok, end, person_index]
        dy = sample[1, ok, start, person_index] - sample[1, ok, end, person_index]
        dist = np.sqrt(dx * dx + dy * dy)
        mean = float(dist.mean())
        if mean > 1:
            limb_lengths.append(float(dist.std() / mean))
    limb_cv = float(np.mean(limb_lengths)) if limb_lengths else 999.0
    return activity, limb_cv, visible_frames


def sample_selection_score(sample):
    person_metrics = [person_activity(sample, person_index) for person_index in range(sample.shape[3])]
    active = [metric for metric in person_metrics if metric[2] >= 8]
    if not active:
        return -1e9
    best_activity, best_limb_cv, best_visible_frames = max(active, key=lambda item: item[0])
    active_penalty = max(0, len(active) - 1) * 1000.0
    visibility_bonus = best_visible_frames / float(sample.shape[1])
    return best_activity + visibility_bonus * 100.0 - best_limb_cv * 250.0 - active_penalty


def choose_sample(reader, labels, source_id, max_candidates):
    indices = np.flatnonzero(labels == source_id)
    if len(indices) == 0:
        raise ValueError("source_id {} not found in selected split".format(source_id))
    if len(indices) > max_candidates:
        candidate_positions = np.linspace(0, len(indices) - 1, max_candidates, dtype=int)
        indices = indices[candidate_positions]

    best_index = int(indices[0])
    best_score = -1e18
    for index in indices:
        sample = reader.read_sample(int(index))
        score = sample_selection_score(sample)
        if score > best_score:
            best_index = int(index)
            best_score = score
    return best_index, best_score


def compute_transform(sample, width, height, pad=40):
    mask = visible_mask(sample)
    xs = sample[0][mask]
    ys = sample[1][mask]
    if len(xs) == 0:
        return lambda x, y: (x, y)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    x_span = max(xmax - xmin, 1.0)
    y_span = max(ymax - ymin, 1.0)
    scale = min((width - 2 * pad) / x_span, (height - 2 * pad) / y_span)
    x_offset = (width - x_span * scale) * 0.5
    y_offset = (height - y_span * scale) * 0.5

    def transform(x, y):
        return x_offset + (x - xmin) * scale, y_offset + (y - ymin) * scale

    return transform


def draw_sample_gif(sample, output_path, title, source_id, sample_index, width, height, duration_ms):
    transform = compute_transform(sample, width, height)
    font = ImageFont.load_default()
    person_colors = [(230, 45, 45), (45, 110, 230)]
    frames = []

    for frame_index in range(sample.shape[1]):
        image = Image.new("RGB", (width, height), (248, 248, 244))
        draw = ImageDraw.Draw(image)
        draw.text(
            (12, 10),
            "{} | A{:03d} | sample {} | frame {:02d}".format(title, source_id + 1, sample_index, frame_index + 1),
            fill=(20, 20, 20),
            font=font,
        )

        for person_index in range(sample.shape[3]):
            color = person_colors[person_index % len(person_colors)]
            points = []
            visible = []
            for joint_index in range(sample.shape[2]):
                x = float(sample[0, frame_index, joint_index, person_index])
                y = float(sample[1, frame_index, joint_index, person_index])
                score = float(sample[2, frame_index, joint_index, person_index])
                is_visible = score > 0.05 and abs(x) + abs(y) > 1e-6
                visible.append(is_visible)
                points.append(transform(x, y) if is_visible else None)

            if sum(visible) < 3:
                continue
            for start, end in COCO17_EDGES:
                if visible[start] and visible[end]:
                    draw.line([points[start], points[end]], fill=color, width=4)
            for joint_index, point in enumerate(points):
                if not visible[joint_index]:
                    continue
                radius = 5 if joint_index not in (0, 1, 2, 3, 4) else 4
                x, y = point
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))

        frames.append(image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    output_dir = Path(args.output_dir)
    labels = load_labels(npz_path, args.split)
    reader = NpzArrayReader(npz_path, "x_{}".format(args.split))
    try:
        for source_id, name in TARGET_CLASSES.items():
            sample_index, score = choose_sample(reader, labels, source_id, args.max_candidates)
            sample = reader.read_sample(sample_index)
            output_path = output_dir / "A{:03d}_{}.gif".format(source_id + 1, name)
            draw_sample_gif(
                sample=sample,
                output_path=output_path,
                title=name,
                source_id=source_id,
                sample_index=sample_index,
                width=args.width,
                height=args.height,
                duration_ms=args.duration_ms,
            )
            print("{} A{:03d}: sample={} motion_score={:.3f} -> {}".format(
                name,
                source_id + 1,
                sample_index,
                score,
                output_path,
            ))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
