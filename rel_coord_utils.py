from types import SimpleNamespace

import numpy as np


SCALE_BONES = (
    (5, 6), (11, 12),
    (5, 11), (6, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
)

DEFAULT_RELATIVE_COORDINATE_RULE = {
    "keypoint_thr": 0.0,
    "origin_smooth_window": 5,
    "scale_smooth_window": 7,
    "min_scale_bones": 4,
    "scale_ratio_min": 0.5,
    "scale_ratio_max": 2.0,
    "smooth_window": 5,
}


def build_relative_coordinate_args(overrides=None):
    rule = dict(DEFAULT_RELATIVE_COORDINATE_RULE)
    if overrides:
        rule.update(overrides)

    rule["keypoint_thr"] = float(rule["keypoint_thr"])
    rule["origin_smooth_window"] = max(1, int(rule["origin_smooth_window"]))
    rule["scale_smooth_window"] = max(1, int(rule["scale_smooth_window"]))
    rule["min_scale_bones"] = max(1, int(rule["min_scale_bones"]))
    rule["scale_ratio_min"] = float(rule["scale_ratio_min"])
    rule["scale_ratio_max"] = float(rule["scale_ratio_max"])
    rule["smooth_window"] = max(1, int(rule["smooth_window"]))
    if (
        rule["scale_ratio_min"] <= 0
        or rule["scale_ratio_max"] < rule["scale_ratio_min"]
    ):
        raise ValueError("Invalid scale ratio range.")
    return SimpleNamespace(**rule)


def valid_joint_mask(keypoints, scores, keypoint_thr):
    finite_xy = np.isfinite(keypoints[:, 0]) & np.isfinite(keypoints[:, 1])
    finite_score = np.isfinite(scores)
    nonzero_xy = np.any(np.abs(keypoints) > 1e-8, axis=1)
    return finite_xy & finite_score & nonzero_xy & (scores > keypoint_thr)


def frame_box_from_valid(keypoints, valid):
    if np.any(valid):
        points = keypoints[valid]
        return np.array([
            np.min(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 0]),
            np.max(points[:, 1]),
        ], dtype=np.float32)
    return np.zeros(4, dtype=np.float32)


def root_motion_origin(keypoints, valid, frame_box):
    valid_hips = [index for index in (11, 12) if valid[index]]
    if valid_hips:
        origin_x = float(np.mean(keypoints[valid_hips, 0]))
    elif np.any(valid):
        origin_x = float(np.mean(keypoints[valid, 0]))
    else:
        origin_x = 0.0

    valid_ankles = [index for index in (15, 16) if valid[index]]
    if valid_ankles:
        origin_y = float(np.max(keypoints[valid_ankles, 1]))
    elif np.any(valid):
        origin_y = float(frame_box[3])
    else:
        origin_y = 0.0

    return np.array([origin_x, origin_y], dtype=np.float32)


def median_smooth_series(values, frame_mask, window_size):
    if window_size <= 1 or len(values) == 0:
        return values.astype(np.float32)

    radius = max(0, window_size // 2)
    smoothed = values.copy().astype(np.float32)
    frame_ids = np.arange(len(values), dtype=np.int32)

    for index, frame_id in enumerate(frame_ids):
        if not frame_mask[index]:
            continue
        nearby = (np.abs(frame_ids - frame_id) <= radius) & frame_mask
        if np.any(nearby):
            smoothed[index] = np.median(values[nearby], axis=0)
    return smoothed


def build_reference_bone_lengths(keypoints, valid_masks):
    lengths_by_bone = {bone: [] for bone in SCALE_BONES}
    for frame_index in range(keypoints.shape[0]):
        valid = valid_masks[frame_index]
        frame_keypoints = keypoints[frame_index]
        for bone in SCALE_BONES:
            first, second = bone
            if not valid[first] or not valid[second]:
                continue
            length = float(np.linalg.norm(frame_keypoints[second] - frame_keypoints[first]))
            if np.isfinite(length) and length > 1e-8:
                lengths_by_bone[bone].append(length)

    return {
        bone: float(np.median(lengths))
        for bone, lengths in lengths_by_bone.items()
        if lengths
    }


def estimate_raw_scale_ratios(keypoints, valid_masks, reference_lengths, min_bones):
    raw_ratios = np.full((keypoints.shape[0],), np.nan, dtype=np.float32)
    bone_counts = np.zeros((keypoints.shape[0],), dtype=np.int32)

    for frame_index in range(keypoints.shape[0]):
        ratios = []
        valid = valid_masks[frame_index]
        frame_keypoints = keypoints[frame_index]
        for bone, reference_length in reference_lengths.items():
            first, second = bone
            if not valid[first] or not valid[second] or reference_length <= 0:
                continue
            current_length = float(np.linalg.norm(frame_keypoints[second] - frame_keypoints[first]))
            if np.isfinite(current_length) and current_length > 1e-8:
                ratios.append(current_length / reference_length)
        bone_counts[frame_index] = len(ratios)
        if len(ratios) >= min_bones:
            raw_ratios[frame_index] = float(np.median(ratios))

    return raw_ratios, bone_counts


def smooth_scale_ratios(raw_ratios, frame_mask, window_size, ratio_min, ratio_max):
    finite = np.isfinite(raw_ratios)
    finite_ratios = raw_ratios[finite]
    global_ratio = float(np.median(finite_ratios)) if len(finite_ratios) else 1.0
    ratios = np.full_like(raw_ratios, global_ratio, dtype=np.float32)
    radius = max(0, window_size // 2)
    frame_ids = np.arange(len(raw_ratios), dtype=np.int32)

    for index, frame_id in enumerate(frame_ids):
        if not frame_mask[index]:
            continue
        nearby = np.abs(frame_ids - frame_id) <= radius
        nearby_ratios = raw_ratios[nearby & finite]
        if len(nearby_ratios):
            value = float(np.median(nearby_ratios))
        elif np.isfinite(raw_ratios[index]):
            value = float(raw_ratios[index])
        else:
            value = global_ratio
        ratios[index] = float(np.clip(value, ratio_min, ratio_max))

    return ratios


def smooth_relative(relative, valid_masks, frame_mask, window_size):
    if window_size <= 1 or relative.shape[0] < 3:
        return relative

    radius = max(1, window_size // 2)
    original = relative.copy()
    smoothed = relative.copy()
    frame_ids = np.arange(relative.shape[0], dtype=np.int32)

    for frame_index, frame_id in enumerate(frame_ids):
        if not frame_mask[frame_index]:
            continue
        nearby = np.flatnonzero((np.abs(frame_ids - frame_id) <= radius) & frame_mask)
        for joint_index in range(17):
            available = [
                sample_index
                for sample_index in nearby
                if valid_masks[sample_index, joint_index]
            ]
            if available:
                smoothed[frame_index, joint_index] = np.mean(
                    original[available, joint_index],
                    axis=0,
                )
    return smoothed


def convert_person_xy_to_relative(keypoints, scores, args=None):
    args = args or build_relative_coordinate_args()
    if isinstance(args, dict):
        args = build_relative_coordinate_args(args)

    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 2):
        raise ValueError("keypoints must have shape T,17,2; got {}".format(keypoints.shape))
    if scores.shape != keypoints.shape[:2]:
        raise ValueError("scores must have shape T,17; got {}".format(scores.shape))

    valid_masks = np.asarray([
        valid_joint_mask(keypoints[index], scores[index], args.keypoint_thr)
        for index in range(keypoints.shape[0])
    ], dtype=bool)
    frame_mask = np.any(valid_masks, axis=1)
    if not np.any(frame_mask):
        return np.zeros_like(keypoints, dtype=np.float32), {
            "valid_frames": 0,
            "reference_bones": 0,
            "mean_scale_bone_count": 0.0,
        }

    frame_boxes = np.asarray([
        frame_box_from_valid(keypoints[index], valid_masks[index])
        for index in range(keypoints.shape[0])
    ], dtype=np.float32)
    raw_origins = np.asarray([
        root_motion_origin(keypoints[index], valid_masks[index], frame_boxes[index])
        for index in range(keypoints.shape[0])
    ], dtype=np.float32)
    frame_origins = median_smooth_series(
        raw_origins,
        frame_mask,
        args.origin_smooth_window,
    )

    reference_lengths = build_reference_bone_lengths(keypoints, valid_masks)
    if reference_lengths:
        canonical_bone_scale = float(np.median(list(reference_lengths.values())))
    else:
        frame_heights = frame_boxes[frame_mask, 3] - frame_boxes[frame_mask, 1]
        valid_heights = frame_heights[frame_heights > 1e-8]
        canonical_bone_scale = float(np.median(valid_heights)) if len(valid_heights) else 1.0
    canonical_bone_scale = max(canonical_bone_scale, 1e-6)

    raw_ratios, bone_counts = estimate_raw_scale_ratios(
        keypoints,
        valid_masks,
        reference_lengths,
        args.min_scale_bones,
    )
    scale_ratios = smooth_scale_ratios(
        raw_ratios,
        frame_mask,
        args.scale_smooth_window,
        args.scale_ratio_min,
        args.scale_ratio_max,
    )
    normalization_scales = np.maximum(canonical_bone_scale * scale_ratios, 1e-6)

    relative = np.zeros_like(keypoints, dtype=np.float32)
    for frame_index in np.flatnonzero(frame_mask):
        valid = valid_masks[frame_index]
        relative[frame_index, valid] = (
            keypoints[frame_index, valid] - frame_origins[frame_index]
        ) / normalization_scales[frame_index]

    relative = smooth_relative(
        relative,
        valid_masks,
        frame_mask,
        args.smooth_window,
    )
    relative[~valid_masks] = 0.0

    stats = {
        "valid_frames": int(np.count_nonzero(frame_mask)),
        "reference_bones": int(len(reference_lengths)),
        "mean_scale_bone_count": float(np.mean(bone_counts[frame_mask])),
    }
    return relative.astype(np.float32), stats
