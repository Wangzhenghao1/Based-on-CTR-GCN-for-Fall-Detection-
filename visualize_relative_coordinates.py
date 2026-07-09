import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tracking_utils import simple_tracking


PROJECT_ROOT = Path(__file__).resolve().parent

# Edit these defaults directly if you do not want to pass command-line arguments.
# DEFAULT_VIDEO_PATH = r"D:\PyCharm\data\MPFDD-main\Scene_4\S4-P4-F2-FALL-1.mp4"
DEFAULT_VIDEO_PATH = r"D:\PyCharm\data\Le2i_Fall_Detection\Coffee_room_01\Coffee_room_01\Videos\video (1).avi"
DEFAULT_OUTPUT_PATH = str(PROJECT_ROOT / "outputs" / "relative_coordinates1.mp4")
DEFAULT_CSV_PATH = str(PROJECT_ROOT / "outputs" / "relative_coordinates.csv")
DEFAULT_YOLO_POSE_WEIGHTS = str(PROJECT_ROOT / "yolo26x-pose.pt")

COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
)

# Face edges are excluded because they are short and noisy. These body bones
# provide redundant scale estimates, so individual occlusions have less impact.
SCALE_BONES = (
    (5, 6), (11, 12),
    (5, 11), (6, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
)

# Approximate body-segment mass fractions used to estimate the center of mass.
BODY_SEGMENTS = (
    ("head", 0.081, (0, 1, 2, 3, 4)),
    ("torso", 0.497, (5, 6, 11, 12)),
    ("left_upper_arm", 0.028, (5, 7)),
    ("right_upper_arm", 0.028, (6, 8)),
    ("left_forearm_hand", 0.022, (7, 9)),
    ("right_forearm_hand", 0.022, (8, 10)),
    ("left_thigh", 0.100, (11, 13)),
    ("right_thigh", 0.100, (12, 14)),
    ("left_shank_foot", 0.061, (13, 15)),
    ("right_shank_foot", 0.061, (14, 16)),
)


def default_device():
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def yolo_pose_to_persons(result, max_people=None):
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    boxes_conf = result.boxes.conf.detach().cpu().numpy()
    keypoints = result.keypoints.xy.detach().cpu().numpy()
    keypoint_scores = result.keypoints.conf.detach().cpu().numpy()

    order = np.argsort(-boxes_conf)
    if max_people is not None and max_people > 0:
        order = order[:max_people]

    persons = []
    for index in order:
        persons.append({
            "keypoints": keypoints[index].astype(np.float32),
            "keypoint_scores": keypoint_scores[index].astype(np.float32),
            "bbox": np.array([
                boxes_xyxy[index][0],
                boxes_xyxy[index][1],
                boxes_xyxy[index][2],
                boxes_xyxy[index][3],
                boxes_conf[index],
            ], dtype=np.float32),
        })
    return persons


def collect_pose_results(args):
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError("Cannot open video: {}".format(args.video))

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(args.weights)
    pose_results = []
    frame_index = 0
    start_time = time.time()

    while True:
        success, frame = capture.read()
        if not success:
            break
        if args.max_frames > 0 and frame_index >= args.max_frames:
            break

        result = model.predict(
            source=frame,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]
        pose_results.append(yolo_pose_to_persons(result, args.max_people))
        frame_index += 1

        if frame_index == 1 or frame_index % args.log_interval == 0:
            elapsed = time.time() - start_time
            print("[pose] frame={} elapsed={:.1f}s".format(frame_index, elapsed))

    capture.release()
    if not pose_results:
        raise RuntimeError("No frames were processed.")
    return pose_results, fps, width, height


def valid_joint_mask(keypoints, scores, threshold):
    return (
        (scores >= threshold)
        & np.isfinite(keypoints[:, 0])
        & np.isfinite(keypoints[:, 1])
        & (keypoints[:, 0] > 1.0)
        & (keypoints[:, 1] > 1.0)
    )


def root_motion_origin(keypoints, valid, bbox):
    valid_hips = [index for index in (11, 12) if valid[index]]
    if valid_hips:
        origin_x = float(np.mean(keypoints[valid_hips, 0]))
        x_source = "hips"
    else:
        valid_torso = [index for index in (5, 6, 11, 12) if valid[index]]
        if valid_torso:
            origin_x = float(np.mean(keypoints[valid_torso, 0]))
            x_source = "torso"
        else:
            origin_x = float((bbox[0] + bbox[2]) * 0.5)
            x_source = "bbox_center"

    valid_ankles = [index for index in (15, 16) if valid[index]]
    if valid_ankles:
        origin_y = float(np.max(keypoints[valid_ankles, 1]))
        y_source = "ankles"
    else:
        origin_y = float(bbox[3])
        y_source = "bbox_bottom"

    return np.array([origin_x, origin_y], dtype=np.float32), "{}+{}".format(
        x_source,
        y_source,
    )


def extract_frame_pose(person, keypoint_threshold):
    keypoints = person["keypoints"]
    scores = person["keypoint_scores"]
    bbox = person["bbox"]
    valid = valid_joint_mask(keypoints, scores, keypoint_threshold)

    if np.any(valid):
        points = keypoints[valid]
        frame_box = np.array([
            np.min(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 0]),
            np.max(points[:, 1]),
        ], dtype=np.float32)
        coordinate_source = "frame_skeleton"
    else:
        frame_box = np.asarray(bbox[:4], dtype=np.float32)
        coordinate_source = "frame_bbox_fallback"

    origin, root_source = root_motion_origin(keypoints, valid, bbox)
    return {
        "keypoints": keypoints.copy(),
        "valid": valid,
        "raw_origin": origin,
        "frame_box": frame_box,
        "coordinate_source": "{}:{}".format(coordinate_source, root_source),
    }


def smooth_track_origins(observations, window_size):
    if not observations:
        return

    raw_origins = np.asarray(
        [item["raw_origin"] for item in observations],
        dtype=np.float32,
    )
    frame_ids = np.asarray([item["frame"] for item in observations], dtype=np.int32)
    radius = max(0, window_size // 2)

    for index, frame_id in enumerate(frame_ids):
        if radius == 0:
            origin = raw_origins[index]
        else:
            nearby = np.abs(frame_ids - frame_id) <= radius
            origin = np.median(raw_origins[nearby], axis=0)
        observations[index]["frame_origin"] = origin.astype(np.float32)


def build_reference_bone_lengths(observations):
    lengths_by_bone = {bone: [] for bone in SCALE_BONES}
    for observation in observations:
        keypoints = observation["keypoints"]
        valid = observation["valid"]
        for bone in SCALE_BONES:
            first, second = bone
            if not valid[first] or not valid[second]:
                continue
            length = float(np.linalg.norm(keypoints[second] - keypoints[first]))
            if np.isfinite(length) and length > 2.0:
                lengths_by_bone[bone].append(length)

    reference_lengths = {
        bone: float(np.median(lengths))
        for bone, lengths in lengths_by_bone.items()
        if lengths
    }
    return reference_lengths


def estimate_raw_bone_scale(observation, reference_lengths, min_bones):
    keypoints = observation["keypoints"]
    valid = observation["valid"]
    ratios = []

    for bone, reference_length in reference_lengths.items():
        first, second = bone
        if not valid[first] or not valid[second] or reference_length <= 0:
            continue
        current_length = float(np.linalg.norm(keypoints[second] - keypoints[first]))
        if np.isfinite(current_length) and current_length > 2.0:
            ratios.append(current_length / reference_length)

    if len(ratios) < min_bones:
        return np.nan, len(ratios)
    return float(np.median(ratios)), len(ratios)


def smooth_bone_scales(
    observations,
    reference_lengths,
    window_size,
    min_bones,
    ratio_min,
    ratio_max,
):
    if reference_lengths:
        canonical_bone_scale = float(np.median(list(reference_lengths.values())))
    else:
        box_heights = [
            float(item["frame_box"][3] - item["frame_box"][1])
            for item in observations
            if item["frame_box"][3] > item["frame_box"][1]
        ]
        canonical_bone_scale = float(np.median(box_heights)) if box_heights else 1.0
    canonical_bone_scale = max(canonical_bone_scale, 1.0)

    raw_ratios = []
    for observation in observations:
        ratio, bone_count = estimate_raw_bone_scale(
            observation,
            reference_lengths,
            min_bones,
        )
        raw_ratios.append(ratio)
        observation["scale_bone_count"] = bone_count
        observation["raw_scale_ratio"] = ratio

    raw_ratios = np.asarray(raw_ratios, dtype=np.float32)
    frame_ids = np.asarray([item["frame"] for item in observations], dtype=np.int32)
    radius = max(0, window_size // 2)
    finite_ratios = raw_ratios[np.isfinite(raw_ratios)]
    global_ratio = float(np.median(finite_ratios)) if len(finite_ratios) else 1.0

    for index, frame_id in enumerate(frame_ids):
        nearby = np.abs(frame_ids - frame_id) <= radius
        nearby_ratios = raw_ratios[nearby & np.isfinite(raw_ratios)]
        if len(nearby_ratios):
            scale_ratio = float(np.median(nearby_ratios))
        elif np.isfinite(raw_ratios[index]):
            scale_ratio = float(raw_ratios[index])
        else:
            scale_ratio = global_ratio

        scale_ratio = float(np.clip(scale_ratio, ratio_min, ratio_max))
        observations[index]["scale_ratio"] = scale_ratio
        observations[index]["normalization_scale"] = (
            canonical_bone_scale * scale_ratio
        )
        observations[index]["canonical_bone_scale"] = canonical_bone_scale


def convert_to_relative_coordinates(observations):
    for observation in observations:
        origin = observation["frame_origin"]
        keypoints = observation["keypoints"]
        valid = observation["valid"]
        frame_box = observation["frame_box"]
        normalization_scale = observation["normalization_scale"]

        relative = np.zeros_like(keypoints, dtype=np.float32)
        relative[valid] = (keypoints[valid] - origin) / normalization_scale
        observation["relative"] = relative
        observation["relative_box"] = np.array([
            (frame_box[0] - origin[0]) / normalization_scale,
            (frame_box[1] - origin[1]) / normalization_scale,
            (frame_box[2] - origin[0]) / normalization_scale,
            (frame_box[3] - origin[1]) / normalization_scale,
        ], dtype=np.float32)


def smooth_relative_track(observations, window_size):
    if window_size <= 1 or len(observations) < 3:
        return

    radius = max(1, window_size // 2)
    frame_ids = np.asarray([item["frame"] for item in observations], dtype=np.int32)
    original = [item["relative"].copy() for item in observations]
    masks = [item["valid"].copy() for item in observations]

    for index, frame_id in enumerate(frame_ids):
        nearby = np.flatnonzero(np.abs(frame_ids - frame_id) <= radius)
        for joint_index in range(17):
            available = [
                sample_index
                for sample_index in nearby
                if masks[sample_index][joint_index]
            ]
            if available:
                observations[index]["relative"][joint_index] = np.mean(
                    [original[sample_index][joint_index] for sample_index in available],
                    axis=0,
                )


def local_segment_center(name, indices, relative, valid):
    if name == "torso":
        shoulders = [index for index in (5, 6) if valid[index]]
        hips = [index for index in (11, 12) if valid[index]]
        if shoulders and hips:
            shoulder_center = np.mean(relative[shoulders], axis=0)
            hip_center = np.mean(relative[hips], axis=0)
            return (shoulder_center + hip_center) * 0.5

    available = [index for index in indices if valid[index]]
    if not available:
        return None
    if len(indices) == 2 and len(available) < 2:
        return None
    return np.mean(relative[available], axis=0)


def compute_local_centers_of_mass(observations):
    for observation in observations:
        relative = observation["relative"]
        valid = observation["valid"]
        weighted_center = np.zeros(2, dtype=np.float64)
        total_mass = 0.0

        for name, mass_fraction, indices in BODY_SEGMENTS:
            center = local_segment_center(name, indices, relative, valid)
            if center is None:
                continue
            weighted_center += center * mass_fraction
            total_mass += mass_fraction

        if total_mass > 0:
            center_of_mass = weighted_center / total_mass
            source = "segment_mass"
        elif np.any(valid):
            center_of_mass = np.mean(relative[valid], axis=0)
            source = "joint_mean"
        else:
            center_of_mass = np.zeros(2, dtype=np.float64)
            source = "root_fallback"

        observation["local_com"] = center_of_mass.astype(np.float32)
        observation["local_com_source"] = source


def calculate_local_com_velocities(observations, fps, max_gap):
    for observation in observations:
        observation["local_com_velocity"] = None
        observation["local_com_speed"] = None

    for index in range(1, len(observations) - 1):
        previous_item = observations[index - 1]
        current_item = observations[index]
        next_item = observations[index + 1]
        previous_gap = current_item["frame"] - previous_item["frame"]
        next_gap = next_item["frame"] - current_item["frame"]
        if previous_gap > max_gap or next_gap > max_gap:
            continue

        frame_interval = next_item["frame"] - previous_item["frame"]
        if frame_interval <= 0:
            continue

        delta_time = frame_interval / fps
        velocity = (
            next_item["local_com"] - previous_item["local_com"]
        ) / delta_time
        current_item["local_com_velocity"] = velocity.astype(np.float32)
        current_item["local_com_speed"] = float(np.linalg.norm(velocity))


def build_relative_data(tracks, args, fps):
    data_by_frame = {}
    all_observations = {}

    for track_id, track_data in tracks.items():
        observations = []
        for item in track_data:
            pose = extract_frame_pose(
                item["data"],
                args.keypoint_thr,
            )
            observations.append({
                "frame": int(item["frame"]),
                "track_id": int(track_id),
                "keypoints": pose["keypoints"],
                "valid": pose["valid"],
                "raw_origin": pose["raw_origin"],
                "frame_box": pose["frame_box"],
                "coordinate_source": pose["coordinate_source"],
            })

        smooth_track_origins(observations, args.origin_smooth_window)
        reference_lengths = build_reference_bone_lengths(observations)
        smooth_bone_scales(
            observations,
            reference_lengths,
            args.scale_smooth_window,
            args.min_scale_bones,
            args.scale_ratio_min,
            args.scale_ratio_max,
        )
        convert_to_relative_coordinates(observations)
        smooth_relative_track(observations, args.smooth_window)
        compute_local_centers_of_mass(observations)
        calculate_local_com_velocities(
            observations,
            fps,
            args.max_velocity_gap,
        )
        all_observations[int(track_id)] = observations
        for observation in observations:
            data_by_frame.setdefault(observation["frame"], []).append(observation)

    return data_by_frame, all_observations


def track_color(track_id):
    hue = int((track_id * 47) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def build_track_layout(all_observations, width, height):
    track_ids = sorted(all_observations)
    count = max(1, len(track_ids))
    columns = max(1, int(math.ceil(math.sqrt(count * width / max(height, 1)))))
    rows = int(math.ceil(count / columns))
    cell_width = width / columns
    cell_height = height / rows

    layout = {}
    for slot, track_id in enumerate(track_ids):
        row = slot // columns
        column = slot % columns
        relative_boxes = np.asarray(
            [item["relative_box"] for item in all_observations[track_id]],
            dtype=np.float32,
        )
        relative_box = np.array([
            np.min(relative_boxes[:, 0]),
            np.min(relative_boxes[:, 1]),
            np.max(relative_boxes[:, 2]),
            np.max(relative_boxes[:, 3]),
        ], dtype=np.float32)
        box_width = max(float(relative_box[2] - relative_box[0]), 1e-6)
        box_height = max(float(relative_box[3] - relative_box[1]), 1e-6)
        display_scale = min(
            0.82 * cell_width / box_width,
            0.72 * cell_height / box_height,
        )
        box_center_relative = np.array([
            (relative_box[0] + relative_box[2]) * 0.5,
            (relative_box[1] + relative_box[3]) * 0.5,
        ], dtype=np.float32)
        cell_center = np.array([
            (column + 0.5) * cell_width,
            (row + 0.56) * cell_height,
        ], dtype=np.float32)
        layout[track_id] = {
            "origin": cell_center - box_center_relative * display_scale,
            "scale": display_scale,
            "relative_box": relative_box,
            "bounds": (
                int(column * cell_width),
                int(row * cell_height),
                int((column + 1) * cell_width),
                int((row + 1) * cell_height),
            ),
        }
    return layout


def relative_to_canvas(relative_point, origin, scale):
    point = origin + relative_point * scale
    return int(round(point[0])), int(round(point[1]))


def local_velocity_arrow_points(
    local_com,
    velocity,
    layout_item,
    arrow_time,
    min_length,
    max_length,
):
    origin = layout_item["origin"]
    display_scale = layout_item["scale"]
    start_float = origin + local_com * display_scale
    displacement = velocity.astype(np.float64) * arrow_time * display_scale
    length = float(np.linalg.norm(displacement))

    if 0 < length < min_length:
        displacement *= min_length / length
        length = min_length
    if length > max_length:
        displacement *= max_length / length

    end_float = start_float + displacement
    start = (int(round(start_float[0])), int(round(start_float[1])))
    end = (int(round(end_float[0])), int(round(end_float[1])))
    return start, end


def draw_frame_box(canvas, layout_item, relative_box):
    origin = layout_item["origin"]
    scale = layout_item["scale"]
    center = (int(round(origin[0])), int(round(origin[1])))
    axis_color = (75, 75, 75)
    box_top_left = relative_to_canvas(
        relative_box[:2],
        origin,
        scale,
    )
    box_bottom_right = relative_to_canvas(
        relative_box[2:],
        origin,
        scale,
    )

    cv2.rectangle(
        canvas,
        box_top_left,
        box_bottom_right,
        axis_color,
        1,
        cv2.LINE_AA,
    )

    cv2.line(
        canvas,
        center,
        (box_bottom_right[0], center[1]),
        axis_color,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        center,
        (center[0], box_top_left[1]),
        axis_color,
        1,
        cv2.LINE_AA,
    )
    cv2.circle(canvas, center, 4, (255, 255, 255), -1, cv2.LINE_AA)


def render_video(args, data_by_frame, all_observations, fps, width, height, frame_count):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot create output video: {}".format(output_path))

    track_ids = sorted(all_observations)
    layout = build_track_layout(all_observations, width, height)

    for frame_index in range(frame_count):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for observation in data_by_frame.get(frame_index, []):
            track_id = observation["track_id"]
            layout_item = layout[track_id]
            relative = observation["relative"]
            valid = observation["valid"]
            color = track_color(track_id)

            draw_frame_box(canvas, layout_item, observation["relative_box"])

            for first, second in COCO_SKELETON:
                if not valid[first] or not valid[second]:
                    continue
                point_a = relative_to_canvas(
                    relative[first],
                    layout_item["origin"],
                    layout_item["scale"],
                )
                point_b = relative_to_canvas(
                    relative[second],
                    layout_item["origin"],
                    layout_item["scale"],
                )
                cv2.line(
                    canvas,
                    point_a,
                    point_b,
                    color,
                    args.line_thickness,
                    cv2.LINE_AA,
                )

            for joint_index in range(17):
                if not valid[joint_index]:
                    continue
                point = relative_to_canvas(
                    relative[joint_index],
                    layout_item["origin"],
                    layout_item["scale"],
                )
                cv2.circle(
                    canvas,
                    point,
                    args.joint_radius,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

            local_com = observation["local_com"]
            com_point = relative_to_canvas(
                local_com,
                layout_item["origin"],
                layout_item["scale"],
            )
            com_color = (0, 165, 255)
            cv2.circle(
                canvas,
                com_point,
                args.com_radius + 2,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                com_point,
                args.com_radius,
                com_color,
                -1,
                cv2.LINE_AA,
            )

            velocity = observation["local_com_velocity"]
            speed = observation["local_com_speed"]
            if (
                velocity is not None
                and speed is not None
                and speed >= args.velocity_threshold
            ):
                arrow_start, arrow_end = local_velocity_arrow_points(
                    local_com,
                    velocity,
                    layout_item,
                    args.velocity_arrow_time,
                    args.min_velocity_arrow_length,
                    args.max_velocity_arrow_length,
                )
                cv2.arrowedLine(
                    canvas,
                    arrow_start,
                    arrow_end,
                    com_color,
                    args.velocity_arrow_thickness,
                    cv2.LINE_AA,
                    tipLength=0.22,
                )

            if args.show_labels:
                x1, y1, _, _ = layout_item["bounds"]
                speed_text = "--" if speed is None else "{:.2f}".format(speed)
                cv2.putText(
                    canvas,
                    "ID:{}  local COM speed={}/s".format(track_id, speed_text),
                    (x1 + 12, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        writer.write(canvas)

    writer.release()
    print("Video saved:", output_path)


def write_csv(csv_path, all_observations):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([
            "frame",
            "track_id",
            "joint_id",
            "relative_x",
            "relative_y",
            "valid",
            "frame_origin_x",
            "frame_origin_y",
            "raw_origin_x",
            "raw_origin_y",
            "raw_scale_ratio",
            "scale_ratio",
            "scale_bone_count",
            "canonical_bone_scale",
            "normalization_scale",
            "local_com_x",
            "local_com_y",
            "local_com_velocity_x_per_s",
            "local_com_velocity_y_per_s",
            "local_com_speed_per_s",
            "local_com_source",
            "frame_box_x1",
            "frame_box_y1",
            "frame_box_x2",
            "frame_box_y2",
            "coordinate_source",
        ])
        for track_id in sorted(all_observations):
            for observation in all_observations[track_id]:
                for joint_id in range(17):
                    writer.writerow([
                        observation["frame"],
                        track_id,
                        joint_id,
                        "{:.6f}".format(float(observation["relative"][joint_id, 0])),
                        "{:.6f}".format(float(observation["relative"][joint_id, 1])),
                        int(observation["valid"][joint_id]),
                        "{:.4f}".format(float(observation["frame_origin"][0])),
                        "{:.4f}".format(float(observation["frame_origin"][1])),
                        "{:.4f}".format(float(observation["raw_origin"][0])),
                        "{:.4f}".format(float(observation["raw_origin"][1])),
                        (
                            ""
                            if not np.isfinite(observation["raw_scale_ratio"])
                            else "{:.6f}".format(float(observation["raw_scale_ratio"]))
                        ),
                        "{:.6f}".format(float(observation["scale_ratio"])),
                        int(observation["scale_bone_count"]),
                        "{:.4f}".format(float(observation["canonical_bone_scale"])),
                        "{:.4f}".format(float(observation["normalization_scale"])),
                        "{:.6f}".format(float(observation["local_com"][0])),
                        "{:.6f}".format(float(observation["local_com"][1])),
                        (
                            ""
                            if observation["local_com_velocity"] is None
                            else "{:.6f}".format(float(observation["local_com_velocity"][0]))
                        ),
                        (
                            ""
                            if observation["local_com_velocity"] is None
                            else "{:.6f}".format(float(observation["local_com_velocity"][1]))
                        ),
                        (
                            ""
                            if observation["local_com_speed"] is None
                            else "{:.6f}".format(float(observation["local_com_speed"]))
                        ),
                        observation["local_com_source"],
                        "{:.4f}".format(float(observation["frame_box"][0])),
                        "{:.4f}".format(float(observation["frame_box"][1])),
                        "{:.4f}".format(float(observation["frame_box"][2])),
                        "{:.4f}".format(float(observation["frame_box"][3])),
                        observation["coordinate_source"],
                    ])
    print("CSV saved:", path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize COCO17 joints in a per-frame root-motion coordinate "
            "system with robust multi-bone scale normalization."
        )
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output MP4 path.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Output relative-coordinate CSV path.")
    parser.add_argument("--weights", default=DEFAULT_YOLO_POSE_WEIGHTS, help="YOLO26 Pose weights.")
    parser.add_argument("--device", default=default_device(), help="cuda, cpu, or CUDA index.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-people", type=int, default=0, help="Maximum people per frame; 0 means unlimited.")
    parser.add_argument("--score-thr", type=float, default=0.4, help="Tracking bbox confidence threshold.")
    parser.add_argument("--keypoint-thr", type=float, default=0.3, help="Minimum keypoint confidence.")
    parser.add_argument("--dist-thr", type=float, default=100.0, help="Tracking center-distance threshold.")
    parser.add_argument("--max-missed", type=int, default=20, help="Frames kept for an occluded track.")
    parser.add_argument("--min-hits", type=int, default=3, help="Detections required to confirm a track.")
    parser.add_argument(
        "--origin-smooth-window",
        type=int,
        default=5,
        help="Odd temporal median window for the root-motion origin.",
    )
    parser.add_argument(
        "--scale-smooth-window",
        type=int,
        default=7,
        help="Odd temporal median window for the multi-bone scale ratio.",
    )
    parser.add_argument(
        "--min-scale-bones",
        type=int,
        default=4,
        help="Minimum valid bones required for a frame scale estimate.",
    )
    parser.add_argument(
        "--scale-ratio-min",
        type=float,
        default=0.5,
        help="Minimum allowed per-frame scale ratio.",
    )
    parser.add_argument(
        "--scale-ratio-max",
        type=float,
        default=2.0,
        help="Maximum allowed per-frame scale ratio.",
    )
    parser.add_argument("--smooth-window", type=int, default=5, help="Temporal smoothing window.")
    parser.add_argument("--line-thickness", type=int, default=3, help="Skeleton line thickness.")
    parser.add_argument("--joint-radius", type=int, default=4, help="Joint point radius.")
    parser.add_argument("--com-radius", type=int, default=7, help="Local center-of-mass point radius.")
    parser.add_argument(
        "--max-velocity-gap",
        type=int,
        default=3,
        help="Do not calculate local COM velocity across a larger frame gap.",
    )
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=0.05,
        help="Minimum local COM speed required to draw an arrow.",
    )
    parser.add_argument(
        "--velocity-arrow-time",
        type=float,
        default=0.20,
        help="Seconds of local COM motion represented by the arrow.",
    )
    parser.add_argument(
        "--min-velocity-arrow-length",
        type=float,
        default=8.0,
        help="Minimum visible velocity arrow length in pixels.",
    )
    parser.add_argument(
        "--max-velocity-arrow-length",
        type=float,
        default=120.0,
        help="Maximum velocity arrow length in pixels.",
    )
    parser.add_argument(
        "--velocity-arrow-thickness",
        type=int,
        default=3,
        help="Local COM velocity arrow thickness.",
    )
    parser.add_argument("--show-labels", action="store_true", help="Show track IDs.")
    parser.add_argument("--max-frames", type=int, default=0, help="Process only the first N frames; 0 means all.")
    parser.add_argument("--log-interval", type=int, default=30, help="Pose progress interval.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.max_people = None if args.max_people <= 0 else args.max_people
    args.origin_smooth_window = max(1, args.origin_smooth_window)
    args.scale_smooth_window = max(1, args.scale_smooth_window)
    args.min_scale_bones = max(1, args.min_scale_bones)
    if args.scale_ratio_min <= 0 or args.scale_ratio_max < args.scale_ratio_min:
        raise ValueError("Invalid scale ratio range.")
    args.smooth_window = max(1, args.smooth_window)
    args.max_velocity_gap = max(1, args.max_velocity_gap)
    args.log_interval = max(1, args.log_interval)

    print("Video:", args.video)
    print("Weights:", args.weights)
    print("Device:", args.device)

    pose_results, fps, width, height = collect_pose_results(args)
    clean_pose_results, tracks = simple_tracking(
        pose_results,
        dist_thr=args.dist_thr,
        max_missed=args.max_missed,
        min_hits=args.min_hits,
        score_thr=args.score_thr,
        keypoint_score_thr=args.keypoint_thr,
        log_prefix="relative_coordinates",
    )
    del clean_pose_results

    data_by_frame, all_observations = build_relative_data(tracks, args, fps)
    print("Frames:", len(pose_results))
    print("Tracks:", len(all_observations))

    render_video(
        args,
        data_by_frame,
        all_observations,
        fps,
        width,
        height,
        len(pose_results),
    )
    write_csv(args.csv, all_observations)


if __name__ == "__main__":
    main()
