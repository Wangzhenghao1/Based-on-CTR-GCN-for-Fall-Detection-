import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont


FUKINECT20_EDGES = (
    (0, 1), (1, 2), (2, 3),
    (2, 4), (4, 5), (5, 6), (6, 7),
    (2, 8), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
)

JOINT_NAMES = (
    "hip_center",
    "spine",
    "shoulder_center",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize FUKinect 20-joint skeleton as a GIF.")
    parser.add_argument(
        "--input",
        required=True,
        help="Raw .mat file or converted npz file.",
    )
    parser.add_argument("--output", default="outputs/fukinect_skeleton.gif")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--view",
        choices=("xy", "zy", "xz"),
        default="xy",
        help="Projection plane. zy is useful for side-view fall motion.",
    )
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--duration-ms", type=int, default=80)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--show-joint-ids", action="store_true")
    return parser.parse_args()


def load_mat(path):
    mat = sio.loadmat(path)
    if "iskelet" not in mat:
        raise ValueError("missing 'iskelet' key in {}".format(path))
    data = np.asarray(mat["iskelet"], dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 60:
        raise ValueError("expected (T, 60), got {}".format(data.shape))
    return data.reshape(data.shape[0], 20, 3)


def load_npz(path, split, index):
    data = np.load(path, allow_pickle=True)
    key = "x_{}".format(split)
    if key not in data:
        raise ValueError("{} not found in {}".format(key, path))
    samples = data[key]
    if index < 0 or index >= len(samples):
        raise IndexError("index {} outside {} samples".format(index, len(samples)))
    sample = np.asarray(samples[index], dtype=np.float32)
    if sample.ndim != 4 or sample.shape[0] != 3 or sample.shape[2] != 20:
        raise ValueError("expected C,T,20,M sample, got {}".format(sample.shape))
    sample = sample[:, :, :, 0].transpose(1, 2, 0)
    label = None
    label_key = "y_{}".format(split)
    if label_key in data:
        label = int(data[label_key][index])
    return sample, label


def load_sequence(input_path, split, index):
    path = Path(input_path)
    if path.suffix.lower() == ".mat":
        return load_mat(path), None
    if path.suffix.lower() == ".npz":
        return load_npz(path, split, index)
    raise ValueError("unsupported input suffix: {}".format(path.suffix))


def project_points(sequence_tvc, view):
    if view == "xy":
        x_axis, y_axis = 0, 1
    elif view == "zy":
        x_axis, y_axis = 2, 1
    else:
        x_axis, y_axis = 0, 2

    projected = sequence_tvc[:, :, [x_axis, y_axis]].copy()
    if view in ("xy", "zy"):
        projected[:, :, 1] *= -1.0
    return projected


def compute_transform(points_tvd, width, height, pad=50):
    valid = np.isfinite(points_tvd).all(axis=2) & (np.abs(points_tvd).sum(axis=2) > 1e-6)
    points = points_tvd[valid]
    if len(points) == 0:
        return lambda x, y: (x, y)

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    offset = np.array([
        (width - span[0] * scale) * 0.5,
        (height - span[1] * scale) * 0.5,
    ], dtype=np.float32)

    def transform(x, y):
        point = np.array([x, y], dtype=np.float32)
        mapped = offset + (point - min_xy) * scale
        return float(mapped[0]), float(mapped[1])

    return transform


def draw_gif(sequence_tvc, output_path, title, view, width, height, duration_ms, show_joint_ids):
    projected = project_points(sequence_tvc, view)
    transform = compute_transform(projected, width, height)
    font = ImageFont.load_default()
    frames = []

    edge_color = (45, 105, 185)
    left_color = (30, 145, 90)
    right_color = (210, 95, 35)
    center_color = (40, 40, 40)

    for frame_index in range(projected.shape[0]):
        image = Image.new("RGB", (width, height), (248, 247, 240))
        draw = ImageDraw.Draw(image)
        draw.text((12, 10), "{} | {} | frame {}".format(title, view, frame_index + 1), fill=(25, 25, 25), font=font)

        points = []
        visible = []
        for joint_index in range(projected.shape[1]):
            x, y = projected[frame_index, joint_index]
            ok = np.isfinite(x) and np.isfinite(y) and abs(float(x)) + abs(float(y)) > 1e-6
            visible.append(ok)
            points.append(transform(float(x), float(y)) if ok else None)

        for start, end in FUKINECT20_EDGES:
            if visible[start] and visible[end]:
                draw.line([points[start], points[end]], fill=edge_color, width=4)

        for joint_index, point in enumerate(points):
            if not visible[joint_index]:
                continue
            if joint_index in (4, 5, 6, 7, 12, 13, 14, 15):
                color = left_color
            elif joint_index in (8, 9, 10, 11, 16, 17, 18, 19):
                color = right_color
            else:
                color = center_color
            x, y = point
            radius = 5
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))
            if show_joint_ids:
                draw.text((x + 6, y - 6), str(joint_index + 1), fill=(10, 10, 10), font=font)

        frames.append(image)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    args = parse_args()
    sequence, label = load_sequence(args.input, args.split, args.index)
    sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
    if args.stride > 1:
        sequence = sequence[::args.stride]
    if args.max_frames > 0:
        sequence = sequence[:args.max_frames]
    if len(sequence) == 0:
        raise ValueError("no frames to visualize")

    title = Path(args.input).stem
    if label is not None:
        title = "{} | label {}".format(title, label)
    draw_gif(
        sequence,
        args.output,
        title,
        args.view,
        args.width,
        args.height,
        args.duration_ms,
        args.show_joint_ids,
    )
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
