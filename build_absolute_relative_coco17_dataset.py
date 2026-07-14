import argparse
import json
from pathlib import Path

import numpy as np

from rel_coord_utils import (
    DEFAULT_RELATIVE_COORDINATE_RULE,
    SCALE_BONES,
    build_relative_coordinate_args,
    convert_person_xy_to_relative,
)


DEFAULT_INPUT_NPZ = "/home/wzh/projects/data/ntu120_coco17_1file/ntu120t60_2d_xsub_benchmark.npz"
DEFAULT_OUTPUT_NPZ = "/home/wzh/projects/data/ntu120_coco17_1file/ntu120t60_2d_xsub_benchmark_absrel5.npz"
DEFAULT_SUMMARY_JSON = "/home/wzh/projects/data/ntu120_coco17_1file/ntu120t60_2d_xsub_benchmark_absrel5_summary.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a COCO17 CTR-GCN npz from x,y,score to "
            "absolute_x,absolute_y,relative_x,relative_y,score."
        )
    )
    parser.add_argument("--input-npz", default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--output-npz", default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--keypoint-thr",
        type=float,
        default=DEFAULT_RELATIVE_COORDINATE_RULE["keypoint_thr"],
        help="A joint is valid when score > this threshold.",
    )
    parser.add_argument(
        "--origin-smooth-window",
        type=int,
        default=DEFAULT_RELATIVE_COORDINATE_RULE["origin_smooth_window"],
    )
    parser.add_argument(
        "--scale-smooth-window",
        type=int,
        default=DEFAULT_RELATIVE_COORDINATE_RULE["scale_smooth_window"],
    )
    parser.add_argument(
        "--min-scale-bones",
        type=int,
        default=DEFAULT_RELATIVE_COORDINATE_RULE["min_scale_bones"],
    )
    parser.add_argument(
        "--scale-ratio-min",
        type=float,
        default=DEFAULT_RELATIVE_COORDINATE_RULE["scale_ratio_min"],
    )
    parser.add_argument(
        "--scale-ratio-max",
        type=float,
        default=DEFAULT_RELATIVE_COORDINATE_RULE["scale_ratio_max"],
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_RELATIVE_COORDINATE_RULE["smooth_window"],
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed. Smaller file, slower write.",
    )
    return parser.parse_args()


def validate_args(args):
    normalized = build_relative_coordinate_args(vars(args))
    for key in DEFAULT_RELATIVE_COORDINATE_RULE:
        setattr(args, key, getattr(normalized, key))


def is_ctvm(data):
    return data.ndim == 5 and data.shape[1] in (2, 3, 5) and data.shape[3] == 17


def ensure_ctvm(data, key):
    if is_ctvm(data):
        return np.asarray(data, dtype=np.float32), False
    if data.ndim == 5 and data.shape[2] == 17 and data.shape[3] in (2, 3, 5):
        return np.transpose(data, (0, 3, 1, 2, 4)).astype(np.float32), True
    raise ValueError(
        "{} must have shape N,C,T,17,M or N,T,17,C,M, got {}".format(
            key,
            data.shape,
        )
    )


def restore_original_layout(data, was_ntvcm):
    if was_ntvcm:
        return np.transpose(data, (0, 2, 3, 1, 4))
    return data


def convert_split(data, split_name, args):
    data, was_ntvcm = ensure_ctvm(data, split_name)
    if data.shape[1] < 3:
        raise ValueError("{} must contain x,y,score channels.".format(split_name))

    n, _c, t, v, m = data.shape
    if v != 17:
        raise ValueError("{} expected 17 joints, got {}".format(split_name, v))

    output = np.zeros((n, 5, t, v, m), dtype=np.float32)
    output[:, 0:2] = data[:, 0:2]
    output[:, 4:5] = data[:, 2:3]

    valid_person_tracks = 0
    total_valid_frames = 0
    total_reference_bones = 0

    for sample_index in range(n):
        if sample_index == 0 or (sample_index + 1) % 1000 == 0 or sample_index + 1 == n:
            print("[{}] {}/{}".format(split_name, sample_index + 1, n), flush=True)

        for person_index in range(m):
            keypoints = np.transpose(
                data[sample_index, 0:2, :, :, person_index],
                (1, 2, 0),
            )
            scores = data[sample_index, 2, :, :, person_index]
            relative, stats = convert_person_xy_to_relative(keypoints, scores, args)
            if stats["valid_frames"] > 0:
                valid_person_tracks += 1
                total_valid_frames += stats["valid_frames"]
                total_reference_bones += stats["reference_bones"]
            output[sample_index, 2:4, :, :, person_index] = np.transpose(
                relative,
                (2, 0, 1),
            )

    summary = {
        "shape": list(output.shape),
        "input_layout": "N,T,V,C,M" if was_ntvcm else "N,C,T,V,M",
        "output_layout": "N,T,V,C,M" if was_ntvcm else "N,C,T,V,M",
        "valid_person_tracks": int(valid_person_tracks),
        "total_valid_frames": int(total_valid_frames),
        "avg_reference_bones_per_valid_track": (
            float(total_reference_bones / valid_person_tracks)
            if valid_person_tracks
            else 0.0
        ),
    }
    return restore_original_layout(output, was_ntvcm), summary


def copy_npz_items(npz_data):
    return {key: npz_data[key] for key in npz_data.files}


def main():
    args = parse_args()
    validate_args(args)

    input_path = Path(args.input_npz)
    output_path = Path(args.output_npz)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print("Input:", input_path)
    print("Output:", output_path)
    npz_data = np.load(str(input_path), allow_pickle=True)
    output_items = copy_npz_items(npz_data)

    rule = {
        key: getattr(args, key)
        for key in DEFAULT_RELATIVE_COORDINATE_RULE
    }
    summary = {
        "input_npz": str(input_path),
        "output_npz": str(output_path),
        "channels": [
            "absolute_x",
            "absolute_y",
            "relative_x",
            "relative_y",
            "score",
        ],
        "relative_coordinate_rule": {
            "origin_x": "mean hip x if hips valid, otherwise mean valid joint x",
            "origin_y": "max ankle y if ankles valid, otherwise frame skeleton bottom",
            "scale": "median body-bone length with temporal smoothed ratio",
            "scale_bones": [list(pair) for pair in SCALE_BONES],
            **rule,
        },
        "splits": {},
    }

    for key in ("x_train", "x_test"):
        if key not in npz_data.files:
            print("Skip missing key:", key)
            continue
        converted, split_summary = convert_split(npz_data[key], key, args)
        output_items[key] = converted.astype(np.float32)
        summary["splits"][key] = split_summary

    output_items["absrel5_metadata_json"] = np.array(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    save_func = np.savez_compressed if args.compressed else np.savez
    save_func(str(output_path), **output_items)

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved npz:", output_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
