import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO

import fall_detection
from feeders import tools

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.ctrgcn import Model


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# VIDEO_PATH = r"D:\PyCharm\data\MPFDD-main\Scene_4\S4-P4-F2-FALL-1.mp4"
# OUTPUT_PATH = r"D:\Webdownload\result\S4-P4-F2-FALL-1_native_ctrgcn.mp4"
# VIDEO_PATH = r"D:\PyCharm\data\MPFDD-main\Scene_2\S2-P2-F2-FALL-3.mp4"
# OUTPUT_PATH = r"D:\Webdownload\result\S2-P2-F2-FALL-3_native_ctrgcn.mp4"
VIDEO_PATH = r"D:\PyCharm\data\Le2i_Fall_Detection\Coffee_room_01\Coffee_room_01\Videos\video (1).avi"
OUTPUT_PATH = r"D:\Webdownload\result\video (1)_benchmark.mp4"
# VIDEO_PATH = r"D:\PyCharm\data\Le2i_Fall_Detection\Office\Office\Videos\video (30).avi"
# OUTPUT_PATH = r"D:\Webdownload\result\video (30)_native_ctrgcn.mp4"


YOLO_POSE_WEIGHTS = "yolo26x-pose.pt"
YOLO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_IMGSZ = 640
YOLO_CONF = 0.25

CTRGCN_CHECKPOINT = r"C:\Users\Tommy\Desktop\model-pt\ntu120t60_2d_xsub.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_FRAMES = 64
STRIDE_FRAMES = 15

NUM_CLASS = 60
NUM_POINT = 17
NUM_PERSON = 2
IN_CHANNELS = 3
INFER_CLIP_LEN = 64
WINDOW_PREPROCESS_MODE = "feeder_like"
INFER_P_INTERVAL = [0.95]
NORMALIZE_KEYPOINTS = True
VIDEO_WIDTH = None
VIDEO_HEIGHT = None

DIST_THR = 100
MAX_MISSED = 20
MIN_HITS = 3
SCORE_THR = 0.4

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "fall-coco", "default.yaml")
LABEL_PATH = os.path.join(PROJECT_ROOT, "label.txt")

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def load_runtime_settings():
    raw_config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as file_obj:
            raw_config = yaml.safe_load(file_obj) or {}

    model_num_class = int(raw_config.get("model_args", {}).get("num_class", NUM_CLASS))
    runtime_config = fall_detection.build_config(raw_config, num_classes=model_num_class)
    raw_label_names = fall_detection.load_label_names(LABEL_PATH)
    source_label_names = fall_detection.expand_source_label_names(raw_label_names, runtime_config)
    compact_label_names = fall_detection.build_compact_label_names(raw_label_names, runtime_config)
    return runtime_config, compact_label_names, source_label_names


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

    print("Running anti-jitter tracking...")

    for frame_idx, persons in enumerate(pose_results_list):
        valid_persons = []
        if persons:
            for person in persons:
                if person['bbox'][4] > score_thr:
                    valid_persons.append(person)

        curr_torsos = []
        for person in valid_persons:
            kpts = person['keypoints']
            valid_points = []
            for index in [5, 6, 11, 12]:
                if kpts[index][0] > 1 and kpts[index][1] > 1:
                    valid_points.append(kpts[index])

            if valid_points:
                torso_center = np.mean(valid_points, axis=0)
            else:
                bbox = person['bbox']
                torso_center = np.array(
                    [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                    dtype=np.float32
                )
            curr_torsos.append(torso_center)

        track_ids = list(active_tracks.keys())
        predicted_centers = []
        for track_id in track_ids:
            track = active_tracks[track_id]
            predicted_centers.append(track['last_center'] + track['velocity'])

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

                new_velocity = new_center - track['last_center']
                track['velocity'] = 0.7 * new_velocity + 0.3 * track['velocity']
                track['last_center'] = new_center
                track['missed'] = 0
                track['hits'] += 1

                if track['hits'] >= min_hits:
                    if track_id not in tracks:
                        tracks[track_id] = []
                    valid_persons[person_index]['track_id'] = track_id
                    tracks[track_id].append({'frame': frame_idx, 'data': valid_persons[person_index]})

                matched_track_indices.add(track_index)
                matched_person_indices.add(person_index)

        for track_index, track_id in enumerate(track_ids):
            if track_index in matched_track_indices:
                continue
            track = active_tracks[track_id]
            track['missed'] += 1
            track['hits'] = 0
            track['last_center'] += track['velocity'] * 0.5
            if track['missed'] > max_missed:
                del active_tracks[track_id]

        for person_index in range(len(valid_persons)):
            if person_index in matched_person_indices:
                continue
            active_tracks[next_id] = {
                'last_center': curr_torsos[person_index],
                'velocity': np.array([0.0, 0.0], dtype=np.float32),
                'missed': 0,
                'hits': 1
            }
            next_id += 1

    print("Tracking done. Valid IDs:", len(tracks))

    clean_pose_results = []
    for persons in pose_results_list:
        clean_frame_persons = []
        if persons:
            for person in persons:
                if 'track_id' in person:
                    clean_frame_persons.append(person)
        clean_pose_results.append(clean_frame_persons)

    return clean_pose_results, tracks


def smooth_tracks(tracks, window_size=5):
    print("Applying improved smoothing (window_size={})...".format(window_size))
    weights = np.array([1, 2, 4, 2, 1], dtype=np.float32) if window_size == 5 else np.ones(window_size, dtype=np.float32)
    weights = weights / weights.sum()

    for track_id, track_data in tracks.items():
        if len(track_data) < window_size:
            continue

        raw_kpts = np.array([item['data']['keypoints'] for item in track_data], dtype=np.float32)
        smoothed_kpts = np.zeros_like(raw_kpts)
        pad = window_size // 2

        for joint_index in range(raw_kpts.shape[1]):
            for coord_index in range(raw_kpts.shape[2]):
                signal = raw_kpts[:, joint_index, coord_index]
                padded_signal = np.pad(signal, (pad, pad), mode='edge')
                smoothed_signal = np.convolve(padded_signal, weights, mode='valid')
                smoothed_kpts[:, joint_index, coord_index] = smoothed_signal

        for item_index, item in enumerate(track_data):
            item['data']['keypoints'] = smoothed_kpts[item_index]

    return tracks


def simple_tracking(pose_results_list, dist_thr=100, max_missed=20, min_hits=3, score_thr=0.4):
    from tracking_utils import simple_tracking as occlusion_aware_tracking
    return occlusion_aware_tracking(
        pose_results_list,
        dist_thr=dist_thr,
        max_missed=max_missed,
        min_hits=min_hits,
        score_thr=score_thr,
        log_prefix="ctrgcn",
    )


def load_checkpoint_weights(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format: {}".format(path))


def build_model(num_class):
    model = Model(
        num_class=num_class,
        num_point=NUM_POINT,
        num_person=NUM_PERSON,
        graph="graph.coco.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=IN_CHANNELS
    )
    state_dict = load_checkpoint_weights(CTRGCN_CHECKPOINT)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


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


def set_video_shape(width, height):
    global VIDEO_WIDTH, VIDEO_HEIGHT
    VIDEO_WIDTH = float(width) if width and width > 0 else None
    VIDEO_HEIGHT = float(height) if height and height > 0 else None


def normalize_keypoints_to_model_space(keypoints):
    if not NORMALIZE_KEYPOINTS:
        return keypoints
    if VIDEO_WIDTH is None or VIDEO_HEIGHT is None:
        raise ValueError("VIDEO_WIDTH/VIDEO_HEIGHT must be set before normalized inference.")
    output = keypoints.copy()
    valid = np.any(np.abs(output) > 1e-6, axis=-1)
    output[..., 0] = output[..., 0] / (VIDEO_WIDTH / 2.0) - 1.0
    output[..., 1] = output[..., 1] / (VIDEO_HEIGHT / 2.0) - 1.0
    output[~valid] = 0.0
    return output


def build_window_array(window_data):
    num_person = 1
    num_frames = len(window_data)
    num_points = 17

    keypoint = np.zeros((num_person, num_frames, num_points, 2), dtype=np.float32)
    keypoint_score = np.zeros((num_person, num_frames, num_points), dtype=np.float32)

    for frame_index, item in enumerate(window_data):
        person = item['data']
        keypoint[0, frame_index] = person['keypoints']
        keypoint_score[0, frame_index] = person['keypoint_scores']

    keypoint = np.nan_to_num(keypoint, nan=0.0, posinf=0.0, neginf=0.0)
    keypoint_score = np.nan_to_num(keypoint_score, nan=0.0, posinf=0.0, neginf=0.0)
    keypoint = normalize_keypoints_to_model_space(keypoint)

   
    data = np.concatenate([keypoint, keypoint_score[..., None]], axis=-1)
    data = fix_num_person(data, NUM_PERSON)
    return data


def window_data_to_model_input(window_data, mode=None):
    mode = mode or WINDOW_PREPROCESS_MODE
    data = build_window_array(window_data)
    if mode == "sample_frames":
        data = sample_frames(data, INFER_CLIP_LEN)
        return np.transpose(data, (3, 1, 2, 0)).astype(np.float32)

    if mode != "feeder_like":
        raise ValueError("unsupported window preprocess mode: {}".format(mode))

    data = np.transpose(data, (3, 1, 2, 0)).astype(np.float32)
    valid_frame_num = np.count_nonzero(
        np.any(np.abs(data) > 1e-6, axis=(0, 2, 3))
    )
    if valid_frame_num <= 0:
        valid_frame_num = 1
    return tools.valid_crop_resize(
        data,
        valid_frame_num,
        INFER_P_INTERVAL,
        INFER_CLIP_LEN,
    ).astype(np.float32)


def run_native_ctrgcn_window(model, window_data, runtime_config):
    data = window_data_to_model_input(window_data)
    data = np.expand_dims(data, axis=0).astype(np.float32)
    tensor = torch.from_numpy(data).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    result = fall_detection.classify_probabilities(probabilities, runtime_config)
    result["probabilities"] = probabilities.tolist()
    return result


def update_frame_state(current_state, new_state):
    if current_state is None:
        return new_state
    priority = {"normal": 0, "fall-like": 1, "fall": 2}
    current_priority = priority.get(current_state["group_label"], 0)
    new_priority = priority.get(new_state["group_label"], 0)
    if new_priority > current_priority:
        return new_state
    if new_priority < current_priority:
        return current_state
    if new_state["fall_score"] >= current_state["fall_score"]:
        return new_state
    return current_state


def main():
    runtime_config, compact_label_names, source_label_names = load_runtime_settings()
    positive_source_id = runtime_config["positive_source_id"]
    print(
        "Positive class: compact={} source={} {}".format(
            runtime_config["positive_class_id"],
            positive_source_id,
            source_label_names[positive_source_id]
        )
    )
    print("Retained source IDs:", runtime_config["retained_source_ids"])
    print("Fall-like source IDs:", runtime_config["fall_like_source_ids"])

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(VIDEO_PATH))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    set_video_shape(width, height)

    yolo_pose_model = YOLO(YOLO_POSE_WEIGHTS)
    start_time = time.time()

    print("Step 1: YOLO pose estimation...")
    frames = []
    pose_results = []
    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_index += 1
        frames.append(frame.copy())

        predict_start = time.time()
        results = yolo_pose_model.predict(
            source=frame,
            device=YOLO_DEVICE,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            verbose=False
        )
        det_time = time.time() - predict_start
        print("[Frame {}] det: {:.4f}s".format(frame_index, det_time))
        pose_results.append(yolo26_pose_to_pose_results(results[0]))

    cap.release()
    if not frames:
        raise RuntimeError("No frames read from video.")

    pose_results, tracks = simple_tracking(
        pose_results,
        dist_thr=DIST_THR,
        max_missed=MAX_MISSED,
        min_hits=MIN_HITS,
        score_thr=SCORE_THR
    )
    tracks = smooth_tracks(tracks, window_size=5)

    print("Step 2: Native CTR-GCN action recognition...")
    action_model = build_model(runtime_config["num_classes"])
    tracks_labels = {}

    for track_id, track_data in tracks.items():
        total_len = len(track_data)
        if total_len < 15:
            continue

        frame_states = {}
        for start_idx in range(0, max(1, total_len - WINDOW_FRAMES + STRIDE_FRAMES), STRIDE_FRAMES):
            end_idx = min(start_idx + WINDOW_FRAMES, total_len)
            window_data = track_data[start_idx:end_idx]
            if len(window_data) < 15:
                continue

            result = run_native_ctrgcn_window(
                action_model,
                window_data,
                runtime_config
            )
            top1_label = compact_label_names[result["top1_label_id"]]
            print(
                "Track {} frames {}-{}: top1={} group={} fall_score={:.3f}".format(
                    track_id,
                    window_data[0]["frame"],
                    window_data[-1]["frame"],
                    top1_label,
                    result["group_label"],
                    result["fall_score"],
                )
            )

            state = {
                "top1_label_id": result["top1_label_id"],
                "top1_source_id": result["top1_source_id"],
                "top1_label_name": top1_label,
                "top3_label_ids": result["top3_label_ids"],
                "top3_source_ids": result["top3_source_ids"],
                "fall_score": result["fall_score"],
                "group_label": result["group_label"],
                "internal_state": result["internal_state"],
                "external_alarm": result["external_alarm"],
            }
            for item in window_data:
                frame_id = item["frame"]
                frame_states[frame_id] = update_frame_state(frame_states.get(frame_id), state)

        tracks_labels[track_id] = frame_states

    print("Step 3: Visualizing...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    for frame_id, (frame_persons, frame_img) in enumerate(zip(pose_results, frames)):
        img = frame_img.copy()
        if frame_persons:
            for person in frame_persons:
                kpts = person['keypoints']
                scores = person['keypoint_scores']
                bbox = person['bbox']
                track_id = person.get('track_id', -1)
                frame_state = tracks_labels.get(track_id, {}).get(frame_id)

                color = (0, 255, 0)
                text = "ID:{} Normal".format(track_id)
                if frame_state is not None:
                    if frame_state["group_label"] == "fall":
                        color = (0, 0, 255)
                        text = "ID:{} FALL {:.2f}".format(track_id, frame_state["fall_score"])
                    elif frame_state["group_label"] == "fall-like":
                        color = (0, 255, 255)
                        text = "ID:{} FALL-LIKE {} {:.2f}".format(
                            track_id,
                            frame_state["top1_label_name"],
                            frame_state["fall_score"]
                        )

                for p1, p2 in COCO_SKELETON:
                    if scores[p1] > 0.3 and scores[p2] > 0.3:
                        pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                        pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                        cv2.line(img, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

                for joint_index in range(len(kpts)):
                    if scores[joint_index] > 0.3:
                        pt = (int(kpts[joint_index][0]), int(kpts[joint_index][1]))
                        cv2.circle(img, pt, radius=3, color=color, thickness=-1)

                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2
                (text_w, text_h), _baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_top = max(0, y1 - text_h - 10)
                cv2.rectangle(img, (x1, text_top), (x1 + text_w, y1), color, -1)
                cv2.putText(img, text, (x1, max(10, y1 - 5)), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        video_writer.write(img)

    video_writer.release()

    total_time = time.time() - start_time
    print("Done! Result saved to:", OUTPUT_PATH)
    print("[INFERENCE] Frames:", len(frames))
    print("[INFERENCE] Time: {:.3f} s".format(total_time))
    print("[INFERENCE] FPS: {:.2f}".format(len(frames) / total_time))


if __name__ == "__main__":
    main()
