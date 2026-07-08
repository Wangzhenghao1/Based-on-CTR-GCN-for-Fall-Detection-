import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RETAINED_SOURCE_IDS = (
    list(range(49)) + [60, 61, 79, 86, 87, 88, 89, 90, 91, 100, 113]
)

# Keep num_class=60 by reusing three weak normal slots.
# Source IDs are 0-based:
#   A038 salute -> 37 -> jogging
#   A037 wipe face -> 36 -> walking
#   A033 check time -> 32 -> running
DEFAULT_REPLACEMENT_LABELS = {
    "jogging": 37,
    "walking": 36,
    "running": 32,
}

TRAIN_SUBJECT_IDS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
    31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
    58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
    93, 94, 95, 97, 98, 100, 103,
}

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}

# COCO17 order:
# nose, left_eye, right_eye, left_ear, right_ear,
# left_shoulder, right_shoulder, left_elbow, right_elbow,
# left_wrist, right_wrist, left_hip, right_hip,
# left_knee, right_knee, left_ankle, right_ankle
NTU_TO_COCO = {
    5: 4,    # left_shoulder
    6: 8,    # right_shoulder
    7: 5,    # left_elbow
    8: 9,    # right_elbow
    9: 6,    # left_wrist
    10: 10,  # right_wrist
    11: 12,  # left_hip
    12: 16,  # right_hip
    13: 13,  # left_knee
    14: 17,  # right_knee
    15: 14,  # left_ankle
    16: 18,  # right_ankle
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a COCO17 2D fall_coco-style npz from NTU .skeleton files, "
            "replacing A038/A037/A033 with jogging/walking/running videos."
        )
    )
    parser.add_argument(
        "--ntu-skeleton-root",
        default="/home/wzh/projects/data/nturgb+d_skeletons/",
        help="Directory containing NTU .skeleton files.",
    )
    parser.add_argument(
        "--normal-video-root",
        default="/home/wzh/projects/data/normal/",
        help="Directory with jogging, walking, running subdirectories.",
    )
    parser.add_argument(
        "--output-npz",
        default="/home/wzh/projects/data/ntu120_coco17_1file/fall_coco_walk_replace.npz",
        help="Output npz path.",
    )
    parser.add_argument(
        "--label-template",
        default="/home/wzh/projects/CTR-GCN-main/label.txt",
        help="Original 120-class label file.",
    )
    parser.add_argument(
        "--label-output",
        default="/home/wzh/projects/data/ntu120_coco17_1file/label_walk_replace.txt",
        help="Output label file with A033/A037/A038 names replaced.",
    )
    parser.add_argument(
        "--summary-output",
        default="/home/wzh/projects/data/ntu120_coco17_1file/fall_coco_walk_replace_summary.json",
        help="Output summary json path.",
    )
    parser.add_argument(
        "--counts-output",
        default="/home/wzh/projects/data/ntu120_coco17_1file/fall_coco_walk_replace_counts.csv",
        help="Output per-class count csv path.",
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--max-person", type=int, default=2)
    parser.add_argument(
        "--ntu-person-mode",
        choices=["primary", "multi"],
        default="primary",
        help="Use only the main moving body for NTU single-person classes, or keep up to max-person bodies.",
    )
    parser.add_argument("--min-valid-frames", type=int, default=8)
    parser.add_argument("--min-valid-joints", type=int, default=8)
    parser.add_argument(
        "--max-interpolate-gap",
        type=int,
        default=8,
        help="Only interpolate missing pose gaps up to this many frames; never extrapolate before entry or after exit.",
    )
    parser.add_argument("--video-test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--yolo-weights", default="yolo26x-pose.pt")
    parser.add_argument("--yolo-device", default="cuda")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--video-frame-step", type=int, default=1)
    parser.add_argument(
        "--video-window-stride",
        type=int,
        default=16,
        help="Sliding-window stride for external videos after pose extraction.",
    )
    parser.add_argument("--max-videos-per-class", type=int, default=0)
    parser.add_argument("--max-ntu-per-class", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Only print planned files/counts.")
    return parser.parse_args()


def parse_ntu_filename(path):
    match = re.search(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})", path.stem)
    if not match:
        return None
    setup, camera, performer, replication, action = [int(value) for value in match.groups()]
    return {
        "setup": setup,
        "camera": camera,
        "performer": performer,
        "replication": replication,
        "source_id": action - 1,
    }


def is_train_ntu_sample(meta):
    return int(meta["performer"]) in TRAIN_SUBJECT_IDS


def read_ntu_skeleton(path):
    frames = []
    with open(path, "r", encoding="utf-8") as file_obj:
        num_frames = int(file_obj.readline().strip())
        for _ in range(num_frames):
            bodies = []
            num_bodies = int(file_obj.readline().strip())
            for _ in range(num_bodies):
                body_info = file_obj.readline().strip().split()
                body_id = int(float(body_info[0])) if body_info else -1
                num_joints = int(file_obj.readline().strip())
                joints = []
                for _ in range(num_joints):
                    joints.append([float(value) for value in file_obj.readline().strip().split()])
                bodies.append({"body_id": body_id, "joints": np.asarray(joints, dtype=np.float32)})
            frames.append(bodies)
    return frames


def body_score(joints):
    xy = joints[:, [5, 6]]
    tracking = joints[:, 11] if joints.shape[1] > 11 else np.ones(len(joints), dtype=np.float32)
    valid = (xy[:, 0] > 1.0) & (xy[:, 1] > 1.0) & (tracking > 0)
    if not np.any(valid):
        return 0.0
    area = bbox_area(xy[valid])
    return float(np.count_nonzero(valid) * 1000.0 + area)


def bbox_area(points):
    if len(points) == 0:
        return 0.0
    width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    return max(0.0, width) * max(0.0, height)


def body_center(joints):
    xy = joints[:, [5, 6]]
    tracking = joints[:, 11] if joints.shape[1] > 11 else np.ones(len(joints), dtype=np.float32)
    torso_ids = [4, 8, 12, 16]
    valid = []
    for index in torso_ids:
        if xy[index, 0] > 1.0 and xy[index, 1] > 1.0 and tracking[index] > 0:
            valid.append(xy[index])
    if valid:
        return np.mean(valid, axis=0)
    valid_mask = (xy[:, 0] > 1.0) & (xy[:, 1] > 1.0)
    if np.any(valid_mask):
        return np.mean(xy[valid_mask], axis=0)
    return None


def select_bodies_with_continuity(frames, max_person=2):
    selected = [[] for _ in range(max_person)]
    last_centers = [None for _ in range(max_person)]
    last_body_ids = [None for _ in range(max_person)]

    for bodies in frames:
        candidates = []
        for body in bodies:
            joints = body["joints"]
            center = body_center(joints)
            if center is None:
                continue
            candidates.append({
                "body": body,
                "center": center,
                "score": body_score(joints),
            })

        assigned = set()
        frame_selected = [None for _ in range(max_person)]

        for person_index in range(max_person):
            best_index = None
            best_cost = None
            for cand_index, cand in enumerate(candidates):
                if cand_index in assigned:
                    continue
                cost = -cand["score"] * 1e-4
                if last_body_ids[person_index] is not None and cand["body"]["body_id"] == last_body_ids[person_index]:
                    cost -= 100.0
                if last_centers[person_index] is not None:
                    cost += float(np.linalg.norm(cand["center"] - last_centers[person_index]))
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_index = cand_index
            if best_index is not None:
                assigned.add(best_index)
                cand = candidates[best_index]
                frame_selected[person_index] = cand["body"]["joints"]
                last_centers[person_index] = cand["center"]
                last_body_ids[person_index] = cand["body"]["body_id"]

        for person_index in range(max_person):
            selected[person_index].append(frame_selected[person_index])

    return selected


def raw_body_valid_joint_count(raw_joints):
    if raw_joints is None or raw_joints.shape[0] < 25:
        return 0
    xy = raw_joints[:, [5, 6]]
    tracking = raw_joints[:, 11] if raw_joints.shape[1] > 11 else np.ones(len(raw_joints), dtype=np.float32)
    valid = (xy[:, 0] > 1.0) & (xy[:, 1] > 1.0) & (tracking > 0)
    return int(np.count_nonzero(valid))


def body_motion_energy(person_frames):
    coords = []
    for raw_joints in person_frames:
        if raw_joints is None or raw_joints.shape[0] < 25:
            continue
        valid_count = raw_body_valid_joint_count(raw_joints)
        if valid_count < 8:
            continue
        coords.append(raw_joints[:, :3])
    if len(coords) < 2:
        return 0.0
    arr = np.asarray(coords, dtype=np.float32)
    # Standard NTU preprocessing ranks bodies by temporal 3D pose variation.
    return float(np.sum(np.std(arr, axis=0)))


def select_primary_body_by_motion(frames):
    track_frames = {}
    track_valid_counts = Counter()
    anonymous_index = 0

    for frame_index, bodies in enumerate(frames):
        for body in bodies:
            raw_joints = body["joints"]
            if body_center(raw_joints) is None:
                continue
            body_id = body["body_id"]
            if body_id < 0:
                body_id = "anonymous_{}".format(anonymous_index)
                anonymous_index += 1
            if body_id not in track_frames:
                track_frames[body_id] = [None for _ in frames]
            track_frames[body_id][frame_index] = raw_joints
            track_valid_counts[body_id] += raw_body_valid_joint_count(raw_joints)

    if not track_frames:
        return [None for _ in frames]

    best_id = None
    best_rank = None
    total_frames = max(1, len(frames))
    for body_id, person_frames in track_frames.items():
        valid_frames = sum(1 for raw_joints in person_frames if raw_body_valid_joint_count(raw_joints) >= 8)
        coverage = valid_frames / float(total_frames)
        motion = body_motion_energy(person_frames)
        avg_valid_joints = track_valid_counts[body_id] / float(max(1, valid_frames))
        rank = (motion, coverage, avg_valid_joints)
        if best_rank is None or rank > best_rank:
            best_id = body_id
            best_rank = rank

    return track_frames[best_id]


def synthesize_face_points(coco_xy, coco_score, raw_joints):
    head = raw_joints[3, [5, 6]]
    neck = raw_joints[2, [5, 6]]
    left_shoulder = raw_joints[4, [5, 6]]
    right_shoulder = raw_joints[8, [5, 6]]
    tracking = raw_joints[:, 11] if raw_joints.shape[1] > 11 else np.ones(len(raw_joints), dtype=np.float32)

    if head[0] <= 1 or head[1] <= 1 or tracking[3] <= 0:
        return

    face_score = 1.0 if tracking[3] >= 2 else 0.65
    coco_xy[0] = head
    coco_score[0] = face_score

    if (
        neck[0] > 1 and neck[1] > 1
        and left_shoulder[0] > 1 and left_shoulder[1] > 1
        and right_shoulder[0] > 1 and right_shoulder[1] > 1
    ):
        shoulder_vec = left_shoulder - right_shoulder
        up_vec = head - neck
        norm = np.linalg.norm(shoulder_vec)
        if norm > 1:
            unit_side = shoulder_vec / norm
        else:
            unit_side = np.array([1.0, 0.0], dtype=np.float32)
        eye_offset = unit_side * norm * 0.10
        ear_offset = unit_side * norm * 0.20
        up_offset = up_vec * 0.08
        coco_xy[1] = head + eye_offset + up_offset
        coco_xy[2] = head - eye_offset + up_offset
        coco_xy[3] = head + ear_offset
        coco_xy[4] = head - ear_offset
        coco_score[1:5] = face_score * 0.75
    else:
        coco_xy[1:5] = head
        coco_score[1:5] = face_score * 0.5


def ntu_body_to_coco(raw_joints):
    coco_xy = np.zeros((17, 2), dtype=np.float32)
    coco_score = np.zeros(17, dtype=np.float32)
    if raw_joints is None or raw_joints.shape[0] < 25:
        return coco_xy, coco_score

    tracking = raw_joints[:, 11] if raw_joints.shape[1] > 11 else np.ones(len(raw_joints), dtype=np.float32)
    for coco_index, ntu_index in NTU_TO_COCO.items():
        point = raw_joints[ntu_index, [5, 6]]
        if point[0] > 1 and point[1] > 1 and tracking[ntu_index] > 0:
            coco_xy[coco_index] = point
            coco_score[coco_index] = 1.0 if tracking[ntu_index] >= 2 else 0.65

    synthesize_face_points(coco_xy, coco_score, raw_joints)
    return coco_xy, coco_score


def short_internal_gap_mask(valid, max_gap):
    fill = np.zeros_like(valid, dtype=bool)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < 2:
        return fill

    first_valid = int(valid_idx[0])
    last_valid = int(valid_idx[-1])
    index = first_valid
    while index <= last_valid:
        if valid[index]:
            index += 1
            continue
        start = index
        while index <= last_valid and not valid[index]:
            index += 1
        end = index
        if start > first_valid and end <= last_valid and (end - start) <= max_gap:
            fill[start:end] = True
    return fill


def smooth_active_segments(arr, active_mask, weights):
    pad = len(weights) // 2
    t_count = len(arr)
    index = 0
    while index < t_count:
        if not active_mask[index]:
            index += 1
            continue
        start = index
        while index < t_count and active_mask[index]:
            index += 1
        end = index
        if end - start >= len(weights):
            segment = arr[start:end]
            padded = np.pad(segment, (pad, pad), mode="edge")
            arr[start:end] = np.convolve(padded, weights, mode="valid")


def interpolate_and_smooth(person_tvc, min_score=0.05, max_gap=8):
    arr = np.asarray(person_tvc, dtype=np.float32).copy()
    if len(arr) == 0:
        return arr
    t_count, joint_count, _ = arr.shape
    time_idx = np.arange(t_count)
    weights = np.array([1, 2, 4, 2, 1], dtype=np.float32)
    weights = weights / weights.sum()
    pad = len(weights) // 2

    for joint_index in range(joint_count):
        score = arr[:, joint_index, 2]
        valid = (
            (score > min_score)
            & (arr[:, joint_index, 0] > 1.0)
            & (arr[:, joint_index, 1] > 1.0)
        )
        if not np.any(valid):
            continue

        valid_idx = time_idx[valid]
        fill_gap = short_internal_gap_mask(valid, max_gap=max(0, int(max_gap)))
        active = valid | fill_gap
        for coord_index in (0, 1):
            signal = arr[:, joint_index, coord_index]
            filled = np.interp(time_idx, valid_idx, signal[valid])
            signal[active] = filled[active]
            signal[~active] = 0.0
            smooth_active_segments(signal, active, weights)
            arr[:, joint_index, coord_index] = signal

        filled_score = np.interp(time_idx, valid_idx, score[valid])
        new_score = np.zeros_like(score)
        new_score[valid] = score[valid]
        new_score[fill_gap] = filled_score[fill_gap] * 0.5
        smooth_active_segments(new_score, active, weights)
        arr[:, joint_index, 2] = np.clip(new_score, 0.0, 1.0)

    return arr


def resample_sequence(person_tvc, target_t):
    arr = np.asarray(person_tvc, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((target_t, 17, 3), dtype=np.float32)
    if len(arr) == target_t:
        return arr.astype(np.float32)
    src_idx = np.arange(len(arr), dtype=np.float32)
    dst_idx = np.linspace(0, len(arr) - 1, target_t, dtype=np.float32)
    out = np.zeros((target_t, arr.shape[1], arr.shape[2]), dtype=np.float32)
    for joint_index in range(arr.shape[1]):
        for coord_index in range(arr.shape[2]):
            out[:, joint_index, coord_index] = np.interp(
                dst_idx,
                src_idx,
                arr[:, joint_index, coord_index],
            )
    out[:, :, 2] = np.clip(out[:, :, 2], 0.0, 1.0)
    return out


def persons_to_ctvm(person_sequences, target_t=64, max_person=2, max_interpolate_gap=8):
    out = np.zeros((3, target_t, 17, max_person), dtype=np.float32)
    for person_index, seq in enumerate(person_sequences[:max_person]):
        smoothed = interpolate_and_smooth(seq, max_gap=max_interpolate_gap)
        fixed = resample_sequence(smoothed, target_t)
        out[:, :, :, person_index] = fixed.transpose(2, 0, 1)
    return out


def convert_ntu_file(path, target_t=64, max_person=2, person_mode="primary", max_interpolate_gap=8):
    frames = read_ntu_skeleton(path)
    if person_mode == "primary":
        selected = [select_primary_body_by_motion(frames)]
    else:
        selected = select_bodies_with_continuity(frames, max_person=max_person)
    person_sequences = []
    for person_frames in selected:
        seq = []
        for raw_joints in person_frames:
            xy, score = ntu_body_to_coco(raw_joints)
            seq.append(np.concatenate([xy, score[:, None]], axis=1))
        person_sequences.append(np.asarray(seq, dtype=np.float32))
    return persons_to_ctvm(
        person_sequences,
        target_t=target_t,
        max_person=max_person,
        max_interpolate_gap=max_interpolate_gap,
    )


def is_valid_ctvm_sample(data, min_valid_frames=8, min_valid_joints=8):
    score = data[2]
    xy_sum = np.abs(data[0]) + np.abs(data[1])
    visible = (score > 0.05) & (xy_sum > 1e-6)
    per_frame_visible = visible.sum(axis=(1, 2))
    return int(np.count_nonzero(per_frame_visible >= min_valid_joints)) >= min_valid_frames


def load_yolo_model(weights):
    from ultralytics import YOLO
    return YOLO(weights)


def video_frame_to_person(result, previous_center=None, conf_threshold=0.0):
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None, previous_center

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    boxes_conf = result.boxes.conf.detach().cpu().numpy()
    kpts_xy = result.keypoints.xy.detach().cpu().numpy()
    kpts_conf = result.keypoints.conf.detach().cpu().numpy()

    best_index = None
    best_cost = None
    for index in range(len(boxes_xyxy)):
        if boxes_conf[index] < conf_threshold:
            continue
        bbox = boxes_xyxy[index]
        center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)
        cost = -float(boxes_conf[index]) * 100.0
        if previous_center is not None:
            cost += float(np.linalg.norm(center - previous_center))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_index = index

    if best_index is None:
        return None, previous_center

    bbox = boxes_xyxy[best_index]
    center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)
    person = np.zeros((17, 3), dtype=np.float32)
    person[:, 0:2] = kpts_xy[best_index].astype(np.float32)
    person[:, 2] = kpts_conf[best_index].astype(np.float32)
    return person, center


def convert_video_file(path, yolo_model, args):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(path))

    frames = []
    previous_center = None
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if (frame_index - 1) % max(1, args.video_frame_step) != 0:
            continue
        result = yolo_model.predict(
            source=frame,
            device=args.yolo_device,
            imgsz=args.yolo_imgsz,
            conf=args.yolo_conf,
            verbose=False,
        )[0]
        person, previous_center = video_frame_to_person(
            result,
            previous_center=previous_center,
            conf_threshold=args.yolo_conf,
        )
        if person is None:
            person = np.zeros((17, 3), dtype=np.float32)
        frames.append(person)
    cap.release()

    if not frames:
        return []

    frame_array = np.asarray(frames, dtype=np.float32)
    windows = []
    valid = valid_pose_frame_mask(frame_array, min_valid_joints=args.min_valid_joints)
    segments = [frame_array[valid]]

    for segment in segments:
        if len(segment) < args.min_valid_frames:
            continue
        for start, end in iter_window_ranges(len(segment), args.window_size, args.video_window_stride):
            windows.append(
                persons_to_ctvm(
                    [segment[start:end]],
                    target_t=args.window_size,
                    max_person=args.max_person,
                    max_interpolate_gap=args.max_interpolate_gap,
                )
            )
    return windows


def valid_pose_frame_mask(frame_array, min_valid_joints=8):
    score = frame_array[:, :, 2]
    xy_sum = np.abs(frame_array[:, :, 0]) + np.abs(frame_array[:, :, 1])
    return ((score > 0.05) & (xy_sum > 1e-6)).sum(axis=1) >= min_valid_joints


def iter_pose_segments(frame_array, min_valid_joints=8, max_gap=8):
    valid_frames = np.flatnonzero(valid_pose_frame_mask(frame_array, min_valid_joints=min_valid_joints))
    if len(valid_frames) == 0:
        return

    group_start = 0
    max_distance = max(0, int(max_gap)) + 1
    for index in range(1, len(valid_frames)):
        if int(valid_frames[index] - valid_frames[index - 1]) > max_distance:
            group = valid_frames[group_start:index]
            yield int(group[0]), int(group[-1]) + 1
            group_start = index
    group = valid_frames[group_start:]
    yield int(group[0]), int(group[-1]) + 1


def iter_window_ranges(num_frames, window_size, stride):
    if num_frames <= 0:
        return
    window_size = max(1, int(window_size))
    stride = max(1, int(stride))
    if num_frames <= window_size:
        yield 0, num_frames
        return

    starts = list(range(0, num_frames - window_size + 1, stride))
    last_start = num_frames - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        yield start, start + window_size


def estimate_video_window_count(video_files, window_size, stride, frame_step):
    try:
        import cv2
    except ImportError:
        return None

    counts = Counter()
    for path, _source_id, class_name in video_files:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        processed_frames = int(np.ceil(frame_count / float(max(1, frame_step))))
        counts[class_name] += sum(1 for _ in iter_window_ranges(processed_frames, window_size, stride))
    return counts


def collect_ntu_files(root, retained_source_ids, replaced_source_ids, max_per_class=0):
    root = Path(root)
    by_class_count = Counter()
    files = []
    for path in sorted(root.glob("*.skeleton")):
        meta = parse_ntu_filename(path)
        if meta is None:
            continue
        source_id = meta["source_id"]
        if source_id not in retained_source_ids:
            continue
        if source_id in replaced_source_ids:
            continue
        if max_per_class and by_class_count[source_id] >= max_per_class:
            continue
        by_class_count[source_id] += 1
        files.append((path, source_id, "train" if is_train_ntu_sample(meta) else "test"))
    return files


def collect_video_files(root, replacement_labels, max_per_class=0):
    root = Path(root)
    rng_files = []
    for folder_name, source_id in replacement_labels.items():
        class_dir = root / folder_name
        class_files = []
        if class_dir.exists():
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    class_files.append(path)
        if max_per_class:
            class_files = class_files[:max_per_class]
        for path in class_files:
            rng_files.append((path, source_id, folder_name))
    return rng_files


def split_video_files(video_files, test_ratio, seed):
    grouped = defaultdict(list)
    for item in video_files:
        grouped[item[2]].append(item)

    rng = np.random.default_rng(seed)
    split_items = []
    for class_name, items in grouped.items():
        order = np.arange(len(items))
        rng.shuffle(order)
        test_count = max(1, int(round(len(items) * test_ratio))) if len(items) > 1 else 0
        test_indices = set(order[:test_count].tolist())
        for index, item in enumerate(items):
            split = "test" if index in test_indices else "train"
            split_items.append((item[0], item[1], split, class_name))
    return split_items


def append_sample(split_data, split_labels, split, data, label):
    if split == "train":
        split_data["train"].append(data)
        split_labels["train"].append(label)
    elif split == "test":
        split_data["test"].append(data)
        split_labels["test"].append(label)
    else:
        raise ValueError("unsupported split: {}".format(split))


def load_label_lines(path):
    if not path or not Path(path).exists():
        return ["A{:03d} class_{}".format(index + 1, index) for index in range(120)]
    with open(path, "r", encoding="utf-8") as file_obj:
        return [line.rstrip("\n") for line in file_obj]


def write_replaced_labels(template_path, output_path, replacement_labels):
    labels = load_label_lines(template_path)
    while len(labels) < 120:
        labels.append("A{:03d} class_{}".format(len(labels) + 1, len(labels)))
    for class_name, source_id in replacement_labels.items():
        labels[source_id] = "A{:03d} {}".format(source_id + 1, class_name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(labels) + "\n")


def write_counts_csv(path, labels, counters):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=[
            "source_id_0based",
            "ntu_id",
            "label",
            "train_count",
            "test_count",
            "total_count",
        ])
        writer.writeheader()
        for source_id in RETAINED_SOURCE_IDS:
            writer.writerow({
                "source_id_0based": source_id,
                "ntu_id": "A{:03d}".format(source_id + 1),
                "label": labels[source_id] if source_id < len(labels) else "class_{}".format(source_id),
                "train_count": counters["train"][source_id],
                "test_count": counters["test"][source_id],
                "total_count": counters["train"][source_id] + counters["test"][source_id],
            })


def main():
    args = parse_args()
    retained_source_ids = set(RETAINED_SOURCE_IDS)
    replacement_labels = dict(DEFAULT_REPLACEMENT_LABELS)
    replaced_source_ids = set(replacement_labels.values())

    ntu_files = collect_ntu_files(
        args.ntu_skeleton_root,
        retained_source_ids=retained_source_ids,
        replaced_source_ids=replaced_source_ids,
        max_per_class=args.max_ntu_per_class,
    )
    raw_video_files = collect_video_files(
        args.normal_video_root,
        replacement_labels,
        max_per_class=args.max_videos_per_class,
    )
    video_files = split_video_files(
        raw_video_files,
        test_ratio=args.video_test_ratio,
        seed=args.seed,
    )
    estimated_windows = estimate_video_window_count(
        raw_video_files,
        window_size=args.window_size,
        stride=args.video_window_stride,
        frame_step=args.video_frame_step,
    )

    print("NTU skeleton files:", len(ntu_files))
    print("External normal videos:", len(video_files))
    if estimated_windows is not None:
        print("Estimated external video windows:", dict(sorted(estimated_windows.items())))
    print("Replacement labels:", replacement_labels)
    if args.dry_run:
        print("Dry run only.")
        return

    split_data = {"train": [], "test": []}
    split_labels = {"train": [], "test": []}
    counters = {"train": Counter(), "test": Counter()}

    skipped_invalid = Counter()

    for index, (path, source_id, split) in enumerate(ntu_files, start=1):
        data = convert_ntu_file(
            path,
            target_t=args.window_size,
            max_person=args.max_person,
            person_mode=args.ntu_person_mode,
            max_interpolate_gap=args.max_interpolate_gap,
        )
        if not is_valid_ctvm_sample(
            data,
            min_valid_frames=args.min_valid_frames,
            min_valid_joints=args.min_valid_joints,
        ):
            skipped_invalid[source_id] += 1
            print("[NTU] skip invalid A{:03d}: {}".format(source_id + 1, path.name))
            continue
        append_sample(split_data, split_labels, split, data, source_id)
        counters[split][source_id] += 1
        if index == 1 or index % 1000 == 0:
            print("[NTU] {}/{} {}".format(index, len(ntu_files), path.name))

    yolo_model = None
    if video_files:
        yolo_model = load_yolo_model(args.yolo_weights)

    for index, (path, source_id, split, class_name) in enumerate(video_files, start=1):
        windows = convert_video_file(path, yolo_model, args)
        if not windows:
            print("[VIDEO] skip empty:", path)
            continue
        for data in windows:
            if not is_valid_ctvm_sample(
                data,
                min_valid_frames=args.min_valid_frames,
                min_valid_joints=args.min_valid_joints,
            ):
                skipped_invalid[source_id] += 1
                continue
            append_sample(split_data, split_labels, split, data, source_id)
            counters[split][source_id] += 1
        print("[VIDEO] {}/{} {} -> A{:03d} {} windows={}".format(
            index,
            len(video_files),
            path.name,
            source_id + 1,
            class_name,
            len(windows),
        ))

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    x_train = np.stack(split_data["train"], axis=0).astype(np.float32) if split_data["train"] else np.zeros((0, 3, args.window_size, 17, args.max_person), dtype=np.float32)
    y_train = np.asarray(split_labels["train"], dtype=np.int64)
    x_test = np.stack(split_data["test"], axis=0).astype(np.float32) if split_data["test"] else np.zeros((0, 3, args.window_size, 17, args.max_person), dtype=np.float32)
    y_test = np.asarray(split_labels["test"], dtype=np.int64)

    np.savez(output_npz, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    write_replaced_labels(args.label_template, args.label_output, replacement_labels)
    labels = load_label_lines(args.label_output)
    write_counts_csv(args.counts_output, labels, counters)

    summary = {
        "output_npz": str(output_npz),
        "shape": {
            "x_train": list(x_train.shape),
            "y_train": list(y_train.shape),
            "x_test": list(x_test.shape),
            "y_test": list(y_test.shape),
        },
        "retained_source_ids": list(RETAINED_SOURCE_IDS),
        "replacement_labels": replacement_labels,
        "skipped_invalid": {str(key): int(value) for key, value in sorted(skipped_invalid.items())},
        "counts": {
            "train": {str(key): int(value) for key, value in sorted(counters["train"].items())},
            "test": {str(key): int(value) for key, value in sorted(counters["test"].items())},
        },
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("Saved:", output_npz)
    print("Labels:", args.label_output)
    print("Counts:", args.counts_output)
    print("Summary:", args.summary_output)


if __name__ == "__main__":
    main()
