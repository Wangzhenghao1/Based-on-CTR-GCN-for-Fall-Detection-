import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO

import fall_detection

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.HDGCN import Model


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}
ANNOTATION_EXTENSIONS = {".txt", ".csv", ".ann"}

NUM_POINT = 17
NUM_PERSON = 2
IN_CHANNELS = 3
INFER_CLIP_LEN = 64


def init_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HD-GCN fall detection on Le2i Annotation_benchmark labels."
    )
    parser.add_argument(
        "--dataset-root",
        default=r"D:\PyCharm\data\Le2i_Fall_Detection",
        help="Le2i root. The script auto-discovers Annotation_benchmark and sibling Videos folders."
    )
    parser.add_argument("--videos-root", help="Root directory containing videos.")
    parser.add_argument("--annotations-root", help="Root directory containing Annotation_benchmark files.")
    parser.add_argument(
        "--checkpoint",
        default=r"D:\Server_download\HD-GCN_best_stage1.pt",
        help="Path to HD-GCN checkpoint."
    )
    parser.add_argument(
        "--config",
        default=os.path.join("config", "fall-coco", "default.yaml"),
        help="Path to HD-GCN fall-coco config yaml."
    )
    parser.add_argument(
        "--yolo-weights",
        default=os.path.join(os.path.dirname(PROJECT_ROOT), "yolo26x-pose.pt"),
        help="YOLO pose checkpoint path."
    )
    parser.add_argument(
        "--output-dir",
        default=r"le2i_hdgcn_eval",
        help="Directory to save evaluation outputs."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--yolo-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--window-frames", type=int, default=64)
    parser.add_argument("--stride-frames", type=int, default=15)
    parser.add_argument("--dist-thr", type=float, default=100)
    parser.add_argument("--max-missed", type=int, default=20)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--score-thr", type=float, default=0.4)
    parser.add_argument("--min-window-len", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rescue-min-consecutive-fall-like",
        type=int,
        default=2,
        help="Minimum consecutive fall-like windows for rescue fall relabeling."
    )
    parser.add_argument(
        "--rescue-fall-prob-threshold",
        type=float,
        default=0.10,
        help="Minimum max p(fall) for rescue fall relabeling."
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def iter_files(root_dir, extensions):
    for root, _dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in extensions:
                yield os.path.join(root, filename)


def build_video_index(videos_root):
    by_relative = {}
    by_stem = defaultdict(list)
    for video_path in iter_files(videos_root, VIDEO_EXTENSIONS):
        relative_no_ext = os.path.splitext(os.path.relpath(video_path, videos_root))[0].replace("\\", "/")
        stem = os.path.splitext(os.path.basename(video_path))[0]
        by_relative[relative_no_ext.lower()] = video_path
        by_stem[stem.lower()].append(video_path)
    return by_relative, by_stem


def build_scene_entries(annotations_dir, videos_dir):
    video_by_relative, video_by_stem = build_video_index(videos_dir)
    entries = []
    missing_videos = []

    for annotation_path in sorted(iter_files(annotations_dir, ANNOTATION_EXTENSIONS)):
        relative_no_ext = os.path.splitext(os.path.relpath(annotation_path, annotations_dir))[0].replace("\\", "/")
        stem = os.path.splitext(os.path.basename(annotation_path))[0]

        video_path = video_by_relative.get(relative_no_ext.lower())
        if video_path is None:
            candidates = video_by_stem.get(stem.lower(), [])
            if len(candidates) == 1:
                video_path = candidates[0]
            elif len(candidates) > 1:
                for candidate in candidates:
                    candidate_rel = os.path.splitext(os.path.relpath(candidate, videos_dir))[0].replace("\\", "/")
                    if candidate_rel.lower().endswith(relative_no_ext.lower()):
                        video_path = candidate
                        break

        if video_path is None:
            missing_videos.append(annotation_path)
            continue

        entries.append({
            "annotation_path": annotation_path,
            "video_path": video_path,
            "relative_key": relative_no_ext,
            "stem": stem,
            "scene_root": os.path.dirname(annotations_dir),
        })

    return entries, missing_videos


def build_annotation_entries(annotations_root, videos_root):
    return build_scene_entries(annotations_root, videos_root)


def build_dataset_entries(dataset_root):
    entries = []
    missing_videos = []
    discovered_scenes = []

    for root, dirs, _files in os.walk(dataset_root):
        if "Annotation_benchmark" not in dirs:
            continue
        annotations_dir = os.path.join(root, "Annotation_benchmark")
        videos_dir = os.path.join(root, "Videos")
        if not os.path.isdir(videos_dir):
            continue

        scene_entries, scene_missing = build_scene_entries(annotations_dir, videos_dir)
        entries.extend(scene_entries)
        missing_videos.extend(scene_missing)
        discovered_scenes.append(root)

    return entries, missing_videos, discovered_scenes


def parse_annotation_label(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as file_obj:
        lines = [line.strip() for line in file_obj if line.strip()]

    if not lines:
        raise ValueError("Empty annotation file: {}".format(annotation_path))

    def parse_single_value(text):
        try:
            return int(text)
        except ValueError:
            return None

    if len(lines) == 1:
        value = parse_single_value(lines[0])
        if value in (0, 1):
            return value

    first_value = parse_single_value(lines[0])
    if first_value in (0, 1) and "," not in lines[0]:
        return first_value

    frame_labels = []
    for line in lines:
        if "," not in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            label_value = int(parts[1])
        except ValueError:
            continue
        if label_value in (0, 1):
            frame_labels.append(label_value)

    if frame_labels:
        return 0 if any(label == 0 for label in frame_labels) else 1

    raise ValueError("Unsupported annotation format: {}".format(annotation_path))


def load_runtime_settings(config_path):
    raw_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as file_obj:
            raw_config = yaml.safe_load(file_obj) or {}

    model_num_class = int(raw_config.get("model_args", {}).get("num_class", 60))
    runtime_config = fall_detection.build_config(raw_config, num_classes=model_num_class)
    source_label_names = fall_detection.load_label_names(os.path.join(PROJECT_ROOT, "label.txt"))
    compact_label_names = fall_detection.build_compact_label_names(source_label_names, runtime_config)
    return runtime_config, compact_label_names, source_label_names


def load_checkpoint_weights(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: {}".format(path))
    return {str(key).split("module.")[-1]: value for key, value in checkpoint.items()}


def build_hdgcn_model(checkpoint_path, num_class, device):
    model = Model(
        num_class=num_class,
        num_point=NUM_POINT,
        num_person=NUM_PERSON,
        graph="graph.coco_hierarchy.Graph",
        graph_args={"labeling_mode": "spatial", "CoM": 12},
        in_channels=IN_CHANNELS
    )
    state_dict = load_checkpoint_weights(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def yolo26_pose_to_pose_results(result):
    frame_res = []
    if result.boxes is None or result.keypoints is None:
        return frame_res
    if len(result.boxes) == 0:
        return frame_res

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    boxes_conf = result.boxes.conf.detach().cpu().numpy()
    kpts_xy = result.keypoints.xy.detach().cpu().numpy()
    kpts_conf = result.keypoints.conf.detach().cpu().numpy()

    for idx in range(len(boxes_xyxy)):
        bbox = np.array([
            boxes_xyxy[idx][0],
            boxes_xyxy[idx][1],
            boxes_xyxy[idx][2],
            boxes_xyxy[idx][3],
            boxes_conf[idx]
        ], dtype=np.float32)
        frame_res.append({
            "keypoints": kpts_xy[idx].astype(np.float32),
            "keypoint_scores": kpts_conf[idx].astype(np.float32),
            "bbox": bbox
        })
    return frame_res


def _legacy_simple_tracking(pose_results_list, dist_thr=100, max_missed=20, min_hits=3, score_thr=0.4):
    next_id = 0
    tracks = {}
    active_tracks = {}

    for frame_idx, persons in enumerate(pose_results_list):
        valid_persons = []
        if persons:
            for person in persons:
                if person["bbox"][4] > score_thr:
                    valid_persons.append(person)

        curr_torsos = []
        for person in valid_persons:
            kpts = person["keypoints"]
            valid_points = []
            for index in [5, 6, 11, 12]:
                if kpts[index][0] > 1 and kpts[index][1] > 1:
                    valid_points.append(kpts[index])
            if valid_points:
                torso_center = np.mean(valid_points, axis=0)
            else:
                bbox = person["bbox"]
                torso_center = np.array(
                    [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                    dtype=np.float32
                )
            curr_torsos.append(torso_center)

        track_ids = list(active_tracks.keys())
        predicted_centers = []
        for track_id in track_ids:
            track = active_tracks[track_id]
            predicted_centers.append(track["last_center"] + track["velocity"])

        matched_track_indices = set()
        matched_person_indices = set()

        if track_ids and curr_torsos:
            dists = np.zeros((len(track_ids), len(curr_torsos)), dtype=np.float32)
            for track_index, pred_center in enumerate(predicted_centers):
                for person_index, curr_center in enumerate(curr_torsos):
                    dists[track_index, person_index] = np.linalg.norm(pred_center - curr_center)

            potential_matches = []
            for track_index in range(len(track_ids)):
                for person_index in range(len(curr_torsos)):
                    if dists[track_index, person_index] < dist_thr:
                        potential_matches.append((dists[track_index, person_index], track_index, person_index))

            potential_matches.sort(key=lambda item: item[0])

            for _dist, track_index, person_index in potential_matches:
                if track_index in matched_track_indices or person_index in matched_person_indices:
                    continue
                track_id = track_ids[track_index]
                track = active_tracks[track_id]
                new_center = curr_torsos[person_index]
                new_velocity = new_center - track["last_center"]
                track["velocity"] = 0.7 * new_velocity + 0.3 * track["velocity"]
                track["last_center"] = new_center
                track["missed"] = 0
                track["hits"] += 1

                if track["hits"] >= min_hits:
                    if track_id not in tracks:
                        tracks[track_id] = []
                    valid_persons[person_index]["track_id"] = track_id
                    tracks[track_id].append({"frame": frame_idx, "data": valid_persons[person_index]})

                matched_track_indices.add(track_index)
                matched_person_indices.add(person_index)

        for track_index, track_id in enumerate(track_ids):
            if track_index in matched_track_indices:
                continue
            track = active_tracks[track_id]
            track["missed"] += 1
            track["hits"] = 0
            track["last_center"] += track["velocity"] * 0.5
            if track["missed"] > max_missed:
                del active_tracks[track_id]

        for person_index in range(len(valid_persons)):
            if person_index in matched_person_indices:
                continue
            active_tracks[next_id] = {
                "last_center": curr_torsos[person_index],
                "velocity": np.array([0.0, 0.0], dtype=np.float32),
                "missed": 0,
                "hits": 1
            }
            next_id += 1

    return tracks


def smooth_tracks(tracks, window_size=5):
    weights = np.array([1, 2, 4, 2, 1], dtype=np.float32) if window_size == 5 else np.ones(window_size, dtype=np.float32)
    weights = weights / weights.sum()

    for _track_id, track_data in tracks.items():
        if len(track_data) < window_size:
            continue
        raw_kpts = np.array([item["data"]["keypoints"] for item in track_data], dtype=np.float32)
        smoothed_kpts = np.zeros_like(raw_kpts)
        pad = window_size // 2
        for joint_index in range(raw_kpts.shape[1]):
            for coord_index in range(raw_kpts.shape[2]):
                signal = raw_kpts[:, joint_index, coord_index]
                padded_signal = np.pad(signal, (pad, pad), mode="edge")
                smoothed_signal = np.convolve(padded_signal, weights, mode="valid")
                smoothed_kpts[:, joint_index, coord_index] = smoothed_signal
        for item_index, item in enumerate(track_data):
            item["data"]["keypoints"] = smoothed_kpts[item_index]
    return tracks


def simple_tracking(pose_results_list, dist_thr=100, max_missed=20, min_hits=3, score_thr=0.4):
    from tracking_utils import simple_tracking as occlusion_aware_tracking
    _clean_pose_results, tracks = occlusion_aware_tracking(
        pose_results_list,
        dist_thr=dist_thr,
        max_missed=max_missed,
        min_hits=min_hits,
        score_thr=score_thr,
        log_prefix="hdgcn_eval",
    )
    return tracks


def sample_frames(data, target_t=64):
    num_person, num_frames, num_points, num_channels = data.shape
    if num_frames == target_t:
        return data
    if num_frames > target_t:
        indices = np.linspace(0, num_frames - 1, target_t).astype(np.int64)
        return data[:, indices, :, :]

    out = np.zeros((num_person, target_t, num_points, num_channels), dtype=data.dtype)
    out[:, :num_frames, :, :] = data
    if num_frames > 0:
        out[:, num_frames:, :, :] = data[:, num_frames - 1:num_frames, :, :]
    return out


def fix_num_person(data, target_m=2):
    num_person, num_frames, num_points, num_channels = data.shape
    if num_person == target_m:
        return data
    if num_person > target_m:
        return data[:target_m]
    out = np.zeros((target_m, num_frames, num_points, num_channels), dtype=data.dtype)
    out[:num_person] = data
    return out


def run_hdgcn_window(model, window_data, runtime_config, device):
    num_person = 1
    num_frames = len(window_data)
    num_points = 17
    keypoint = np.zeros((num_person, num_frames, num_points, 2), dtype=np.float32)
    keypoint_score = np.zeros((num_person, num_frames, num_points), dtype=np.float32)

    for frame_index, item in enumerate(window_data):
        person = item["data"]
        keypoint[0, frame_index] = person["keypoints"]
        keypoint_score[0, frame_index] = person["keypoint_scores"]

    keypoint = np.nan_to_num(keypoint, nan=0.0, posinf=0.0, neginf=0.0)
    keypoint_score = np.nan_to_num(keypoint_score, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.concatenate([keypoint, keypoint_score[..., None]], axis=-1)
    data = fix_num_person(data, NUM_PERSON)
    data = sample_frames(data, INFER_CLIP_LEN)
    data = np.transpose(data, (3, 1, 2, 0))
    data = np.expand_dims(data, axis=0).astype(np.float32)
    tensor = torch.from_numpy(data).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    result = fall_detection.classify_probabilities(probabilities, runtime_config)
    result["probabilities"] = probabilities.tolist()
    return result


def run_video(video_path, runtime_config, compact_label_names, action_model, yolo_model, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(video_path))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    pose_results = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = yolo_model.predict(
            source=frame,
            device=args.yolo_device,
            imgsz=args.yolo_imgsz,
            conf=args.yolo_conf,
            verbose=False
        )
        pose_results.append(yolo26_pose_to_pose_results(results[0]))
    cap.release()

    tracks = simple_tracking(
        pose_results,
        dist_thr=args.dist_thr,
        max_missed=args.max_missed,
        min_hits=args.min_hits,
        score_thr=args.score_thr
    )
    tracks = smooth_tracks(tracks, window_size=5)

    track_summaries = []
    all_window_results = []
    for track_id, track_data in sorted(tracks.items()):
        total_len = len(track_data)
        if total_len < args.min_window_len:
            continue

        windows = []
        for start_idx in range(0, max(1, total_len - args.window_frames + args.stride_frames), args.stride_frames):
            end_idx = min(start_idx + args.window_frames, total_len)
            window_data = track_data[start_idx:end_idx]
            if len(window_data) < args.min_window_len:
                continue

            result = run_hdgcn_window(action_model, window_data, runtime_config, args.device)
            window_summary = {
                "track_id": int(track_id),
                "start_frame": int(window_data[0]["frame"]),
                "end_frame": int(window_data[-1]["frame"]),
                "top1_label_id": int(result["top1_label_id"]),
                "top1_source_id": int(result["top1_source_id"]),
                "top1_label_name": compact_label_names[result["top1_label_id"]],
                "top3_label_ids": [int(value) for value in result["top3_label_ids"]],
                "top3_source_ids": [int(value) for value in result["top3_source_ids"]],
                "group_label": result["group_label"],
                "fall_score": float(result["fall_score"]),
            }
            windows.append(window_summary)
            all_window_results.append(window_summary)

        if windows:
            consecutive_fall_like = 0
            max_consecutive_fall_like = 0
            for window in windows:
                if window["group_label"] == "fall-like":
                    consecutive_fall_like += 1
                    max_consecutive_fall_like = max(max_consecutive_fall_like, consecutive_fall_like)
                else:
                    consecutive_fall_like = 0

            track_summaries.append({
                "track_id": int(track_id),
                "num_windows": len(windows),
                "max_consecutive_fall_like": int(max_consecutive_fall_like),
                "windows": windows,
            })

    predicted_fall_strict = any(item["group_label"] == "fall" for item in all_window_results)
    predicted_fall_like = any(item["group_label"] == "fall-like" for item in all_window_results)
    max_fall_score = max((item["fall_score"] for item in all_window_results), default=0.0)
    max_consecutive_fall_like = max((track["max_consecutive_fall_like"] for track in track_summaries), default=0)
    predicted_fall_rescue = (
        (not predicted_fall_strict)
        and max_consecutive_fall_like >= int(args.rescue_min_consecutive_fall_like)
        and max_fall_score >= float(args.rescue_fall_prob_threshold)
    )
    predicted_fall = predicted_fall_strict or predicted_fall_rescue
    predicted_label = 0 if predicted_fall else 1
    if predicted_fall_strict:
        trigger_rule = "strict_top1_fall"
    elif predicted_fall_rescue:
        trigger_rule = "fall_like_run_ge_{}_pge_{:.2f}".format(
            int(args.rescue_min_consecutive_fall_like),
            float(args.rescue_fall_prob_threshold)
        )
    else:
        trigger_rule = "none"

    return {
        "video_path": video_path,
        "frame_count": int(frame_count or len(pose_results)),
        "fps": float(fps),
        "num_tracks": len(track_summaries),
        "num_windows": len(all_window_results),
        "predicted_label": int(predicted_label),
        "predicted_fall": bool(predicted_fall),
        "predicted_fall_strict": bool(predicted_fall_strict),
        "predicted_fall_rescue": bool(predicted_fall_rescue),
        "predicted_fall_like": bool(predicted_fall_like),
        "trigger_rule": trigger_rule,
        "max_consecutive_fall_like": int(max_consecutive_fall_like),
        "max_fall_score": float(max_fall_score),
        "tracks": track_summaries,
    }


def compute_metrics(results):
    tp = sum(1 for item in results if item["gt_label"] == 0 and item["predicted_label"] == 0)
    fn = sum(1 for item in results if item["gt_label"] == 0 and item["predicted_label"] == 1)
    fp = sum(1 for item in results if item["gt_label"] == 1 and item["predicted_label"] == 0)
    tn = sum(1 for item in results if item["gt_label"] == 1 and item["predicted_label"] == 1)
    total = tp + fn + fp + tn

    def safe_div(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    fall_precision = safe_div(tp, tp + fp)
    fall_recall = safe_div(tp, tp + fn)
    fall_f1 = safe_div(2 * fall_precision * fall_recall, fall_precision + fall_recall)

    return {
        "num_videos": total,
        "num_fall_videos": tp + fn,
        "num_normal_videos": tn + fp,
        "accuracy": round(safe_div(tp + tn, total), 6),
        "fall_precision": round(fall_precision, 6),
        "fall_recall": round(fall_recall, 6),
        "fall_f1": round(fall_f1, 6),
        "normal_precision": round(safe_div(tn, tn + fn), 6),
        "normal_recall": round(safe_div(tn, tn + fp), 6),
        "confusion_matrix": {
            "labels": ["fall(0)", "normal(1)"],
            "matrix": [[int(tp), int(fn)], [int(fp), int(tn)]],
        },
        "counts": {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)},
    }


def save_results(output_dir, results, metrics, args):
    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, "evaluation_results.json")
    csv_path = os.path.join(output_dir, "evaluation_results.csv")

    payload = {"metrics": metrics, "args": vars(args), "results": results}
    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([
            "video_path",
            "annotation_path",
            "gt_label",
            "predicted_label",
            "correct",
            "predicted_fall",
            "predicted_fall_strict",
            "predicted_fall_rescue",
            "predicted_fall_like",
            "trigger_rule",
            "max_consecutive_fall_like",
            "max_fall_score",
            "num_tracks",
            "num_windows",
            "error",
        ])
        for item in results:
            writer.writerow([
                item["video_path"],
                item["annotation_path"],
                item["gt_label"],
                item["predicted_label"],
                int(item["gt_label"] == item["predicted_label"]) if item["predicted_label"] in (0, 1) else 0,
                int(bool(item.get("predicted_fall"))),
                int(bool(item.get("predicted_fall_strict"))),
                int(bool(item.get("predicted_fall_rescue"))),
                int(bool(item.get("predicted_fall_like"))),
                item.get("trigger_rule", ""),
                item.get("max_consecutive_fall_like", 0),
                item.get("max_fall_score", 0.0),
                item.get("num_tracks", 0),
                item.get("num_windows", 0),
                item.get("error", ""),
            ])
    return json_path, csv_path


def main():
    args = parse_args()
    init_seed(args.seed)
    ensure_dir(args.output_dir)

    if args.dataset_root:
        entries, missing_videos, discovered_scenes = build_dataset_entries(args.dataset_root)
        if not discovered_scenes:
            raise RuntimeError("No scene folders with Annotation_benchmark and sibling Videos were found.")
        print("Discovered {} scene folders.".format(len(discovered_scenes)))
    else:
        if not args.annotations_root or not args.videos_root:
            raise RuntimeError("Either --dataset-root or both --videos-root and --annotations-root must be provided.")
        entries, missing_videos = build_annotation_entries(args.annotations_root, args.videos_root)

    if not entries:
        raise RuntimeError("No matching annotation/video pairs found.")

    if missing_videos:
        print("Warning: {} annotation files have no matching video.".format(len(missing_videos)))
        missing_path = os.path.join(args.output_dir, "missing_videos.txt")
        with open(missing_path, "w", encoding="utf-8") as file_obj:
            for path in missing_videos:
                file_obj.write(path + "\n")

    runtime_config, compact_label_names, source_label_names = load_runtime_settings(args.config)
    action_model = build_hdgcn_model(args.checkpoint, runtime_config["num_classes"], args.device)
    yolo_model = YOLO(args.yolo_weights)

    print(
        "Loaded HD-GCN: positive_source_id={} ({})".format(
            runtime_config["positive_source_id"],
            source_label_names[runtime_config["positive_source_id"]]
        )
    )
    print("Matched {} videos for evaluation.".format(len(entries)))

    results = []
    for index, entry in enumerate(entries, start=1):
        annotation_path = entry["annotation_path"]
        video_path = entry["video_path"]
        print("[{}/{}] {}".format(index, len(entries), os.path.basename(video_path)))
        result = {"annotation_path": annotation_path, "video_path": video_path}
        try:
            gt_label = parse_annotation_label(annotation_path)
            video_result = run_video(video_path, runtime_config, compact_label_names, action_model, yolo_model, args)
            result.update(video_result)
            result["gt_label"] = int(gt_label)
            print(
                "  gt={} pred={} rule={} tracks={} windows={} max_fall_score={:.3f}".format(
                    gt_label,
                    result["predicted_label"],
                    result["trigger_rule"],
                    result["num_tracks"],
                    result["num_windows"],
                    result["max_fall_score"],
                )
            )
        except Exception as exc:
            result["gt_label"] = -1
            result["predicted_label"] = -1
            result["error"] = str(exc)
            print("  error: {}".format(exc))
        results.append(result)

    valid_results = [
        item for item in results
        if item["gt_label"] in (0, 1) and item["predicted_label"] in (0, 1)
    ]
    if not valid_results:
        raise RuntimeError("No valid evaluation results were produced.")

    metrics = compute_metrics(valid_results)
    json_path, csv_path = save_results(args.output_dir, results, metrics, args)

    print("Evaluation done.")
    print("Accuracy: {:.4f}".format(metrics["accuracy"]))
    print("Fall precision: {:.4f}".format(metrics["fall_precision"]))
    print("Fall recall: {:.4f}".format(metrics["fall_recall"]))
    print("Fall F1: {:.4f}".format(metrics["fall_f1"]))
    print("Confusion matrix:", metrics["confusion_matrix"]["matrix"])
    print("Saved JSON:", json_path)
    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    main()
