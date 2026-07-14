import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

import yolo26_ctrgcn_ntu60 as inferlib


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}
ANNOTATION_EXTENSIONS = {".txt", ".csv", ".ann"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fall detection on Le2i with Annotation_benchmark labels."
    )
    parser.add_argument(
        "--dataset-root",
        default=r"D:\PyCharm\data\Le2i_Fall_Detection",
        help="Le2i dataset root. If set, the script auto-discovers Annotation_benchmark and sibling Videos folders."
    )
    parser.add_argument("--videos-root", help="Root directory containing Le2i videos.")
    parser.add_argument("--annotations-root", help="Root directory containing Annotation_benchmark files.")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            "work_dir", "fall-coco", "xsub",
            "ctrgcn_absrel5_60_benchmark", "best_stage1.pt"
        ),
        help="Path to the CTR-GCN checkpoint."
    )
    parser.add_argument(
        "--config",
        default=os.path.join("config", "fall-coco", "absrel5.yaml"),
        help="Path to config yaml. model_args.in_channels selects 3- or 5-channel inference."
    )
    parser.add_argument("--yolo-weights", default="yolo26x-pose.pt", help="YOLO pose checkpoint path.")
    parser.add_argument(
        "--output-dir",
        default=r"D:\Server_download\le2i_eval",
        help="Directory to save evaluation outputs."
    )
    parser.add_argument("--device", default=inferlib.DEVICE, help="Device for CTR-GCN inference.")
    parser.add_argument("--yolo-device", default=inferlib.YOLO_DEVICE, help="Device for YOLO pose inference.")
    parser.add_argument("--yolo-imgsz", type=int, default=inferlib.YOLO_IMGSZ, help="YOLO pose inference image size.")
    parser.add_argument("--yolo-conf", type=float, default=inferlib.YOLO_CONF, help="YOLO pose confidence threshold.")
    parser.add_argument("--window-frames", type=int, default=inferlib.WINDOW_FRAMES, help="Sliding window size in frames.")
    parser.add_argument("--stride-frames", type=int, default=inferlib.STRIDE_FRAMES, help="Sliding window stride in frames.")
    parser.add_argument("--dist-thr", type=float, default=inferlib.DIST_THR, help="Tracking distance threshold.")
    parser.add_argument("--max-missed", type=int, default=inferlib.MAX_MISSED, help="Maximum missed frames for tracking.")
    parser.add_argument("--min-hits", type=int, default=inferlib.MIN_HITS, help="Minimum hits for a valid track.")
    parser.add_argument("--score-thr", type=float, default=inferlib.SCORE_THR, help="Minimum detection score for tracking.")
    parser.add_argument("--min-window-len", type=int, default=15, help="Minimum number of tracked frames required for a window.")
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
                exact_suffix = None
                for candidate in candidates:
                    candidate_rel = os.path.splitext(os.path.relpath(candidate, videos_dir))[0].replace("\\", "/")
                    if candidate_rel.lower().endswith(relative_no_ext.lower()):
                        exact_suffix = candidate
                        break
                video_path = exact_suffix

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

    # Case 1: benchmark file contains a single 0/1 label.
    if len(lines) == 1:
        value = parse_single_value(lines[0])
        if value in (0, 1):
            return value

    # Case 2: first non-empty line is a video-level 0/1 label.
    first_value = parse_single_value(lines[0])
    if first_value in (0, 1) and "," not in lines[0]:
        return first_value

    # Case 3: frame-level records where the second column has 0/1 labels.
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


def load_runtime(args):
    inferlib.CTRGCN_CHECKPOINT = os.path.abspath(args.checkpoint)
    inferlib.CONFIG_PATH = os.path.abspath(args.config)
    inferlib.YOLO_POSE_WEIGHTS = os.path.abspath(args.yolo_weights)
    inferlib.DEVICE = args.device
    inferlib.YOLO_DEVICE = args.yolo_device
    inferlib.YOLO_IMGSZ = args.yolo_imgsz
    inferlib.YOLO_CONF = args.yolo_conf
    inferlib.WINDOW_FRAMES = args.window_frames
    inferlib.STRIDE_FRAMES = args.stride_frames
    inferlib.DIST_THR = args.dist_thr
    inferlib.MAX_MISSED = args.max_missed
    inferlib.MIN_HITS = args.min_hits
    inferlib.SCORE_THR = args.score_thr

    runtime_config, compact_label_names, source_label_names = inferlib.load_runtime_settings()
    action_model = inferlib.build_model(
        runtime_config["num_classes"],
        runtime_config["model_in_channels"],
    )
    yolo_model = YOLO(inferlib.YOLO_POSE_WEIGHTS)
    return runtime_config, compact_label_names, source_label_names, action_model, yolo_model


def run_video(video_path, runtime_config, compact_label_names, action_model, yolo_model, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(video_path))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if hasattr(inferlib, "set_video_shape"):
        inferlib.set_video_shape(width, height)

    pose_results = []
    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_index += 1
        results = yolo_model.predict(
            source=frame,
            device=inferlib.YOLO_DEVICE,
            imgsz=inferlib.YOLO_IMGSZ,
            conf=inferlib.YOLO_CONF,
            verbose=False
        )
        pose_results.append(inferlib.yolo26_pose_to_pose_results(results[0]))

    cap.release()

    clean_pose_results, tracks = inferlib.simple_tracking(
        pose_results,
        dist_thr=inferlib.DIST_THR,
        max_missed=inferlib.MAX_MISSED,
        min_hits=inferlib.MIN_HITS,
        score_thr=inferlib.SCORE_THR
    )
    if runtime_config["model_in_channels"] == 3:
        tracks = inferlib.smooth_tracks(tracks, window_size=5)

    track_summaries = []
    all_window_results = []
    for track_id, track_data in sorted(tracks.items()):
        total_len = len(track_data)
        if total_len < args.min_window_len:
            continue

        windows = []
        for start_idx in range(0, max(1, total_len - inferlib.WINDOW_FRAMES + inferlib.STRIDE_FRAMES), inferlib.STRIDE_FRAMES):
            end_idx = min(start_idx + inferlib.WINDOW_FRAMES, total_len)
            window_data = track_data[start_idx:end_idx]
            if len(window_data) < args.min_window_len:
                continue

            result = inferlib.run_native_ctrgcn_window(
                action_model,
                window_data,
                runtime_config
            )
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
            track_summaries.append({
                "track_id": int(track_id),
                "num_windows": len(windows),
                "windows": windows,
            })

    predicted_fall = any(item["group_label"] == "fall" for item in all_window_results)
    predicted_fall_like = any(item["group_label"] == "fall-like" for item in all_window_results)
    max_fall_score = max((item["fall_score"] for item in all_window_results), default=0.0)
    predicted_label = 0 if predicted_fall else 1
    trigger_rule = "strict_top1_fall" if predicted_fall else "none"

    return {
        "video_path": video_path,
        "frame_count": int(frame_count or len(clean_pose_results)),
        "fps": float(fps),
        "input_channels": int(runtime_config["model_in_channels"]),
        "num_tracks": len(track_summaries),
        "num_windows": len(all_window_results),
        "predicted_label": int(predicted_label),
        "predicted_fall": bool(predicted_fall),
        "predicted_fall_like": bool(predicted_fall_like),
        "trigger_rule": trigger_rule,
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

    accuracy = safe_div(tp + tn, total)
    fall_precision = safe_div(tp, tp + fp)
    fall_recall = safe_div(tp, tp + fn)
    normal_precision = safe_div(tn, tn + fn)
    normal_recall = safe_div(tn, tn + fp)
    fall_f1 = safe_div(2 * fall_precision * fall_recall, fall_precision + fall_recall)

    return {
        "num_videos": total,
        "num_fall_videos": tp + fn,
        "num_normal_videos": tn + fp,
        "accuracy": round(accuracy, 6),
        "fall_precision": round(fall_precision, 6),
        "fall_recall": round(fall_recall, 6),
        "fall_f1": round(fall_f1, 6),
        "normal_precision": round(normal_precision, 6),
        "normal_recall": round(normal_recall, 6),
        "confusion_matrix": {
            "labels": ["fall(0)", "normal(1)"],
            "matrix": [
                [int(tp), int(fn)],
                [int(fp), int(tn)],
            ],
        },
        "counts": {
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
        },
    }


def save_results(output_dir, results, metrics, args):
    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, "evaluation_results.json")
    csv_path = os.path.join(output_dir, "evaluation_results.csv")

    payload = {
        "metrics": metrics,
        "args": vars(args),
        "results": results,
    }
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
            "predicted_fall_like",
            "trigger_rule",
            "max_fall_score",
            "input_channels",
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
                int(bool(item.get("predicted_fall_like"))),
                item.get("trigger_rule", ""),
                item.get("max_fall_score", 0.0),
                item.get("input_channels", 0),
                item.get("num_tracks", 0),
                item.get("num_windows", 0),
                item.get("error", ""),
            ])

    return json_path, csv_path


def main():
    args = parse_args()
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

    runtime_config, compact_label_names, source_label_names, action_model, yolo_model = load_runtime(args)
    print(
        "Loaded runtime: C={} positive_source_id={} ({})".format(
            runtime_config["model_in_channels"],
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
        result = {
            "annotation_path": annotation_path,
            "video_path": video_path,
        }
        try:
            gt_label = parse_annotation_label(annotation_path)
            video_result = run_video(
                video_path,
                runtime_config,
                compact_label_names,
                action_model,
                yolo_model,
                args
            )
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

    valid_results = [item for item in results if item["gt_label"] in (0, 1) and item["predicted_label"] in (0, 1)]
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
