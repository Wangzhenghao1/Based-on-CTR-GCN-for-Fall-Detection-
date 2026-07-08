import os
import sys
import time
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ===== 原生 CTR-GCN 项目路径 =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from model.ctrgcn import Model

# ===== rtmlib =====
from rtmlib import Body


# =========================================================
# 0. 固定随机性
# =========================================================
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================================
# 1. 配置区域
# =========================================================
VIDEO_PATH = r"D:\PyCharm\data\Le2i_Fall_Detection\Coffee_room_01\Coffee_room_01\Videos\video (9).avi"
OUTPUT_PATH = r"D:\Webdownload\结果\video (9)_native_ctrgcn.mp4"

# rtmlib
RTM_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RTM_BACKEND = 'onnxruntime'   # opencv / onnxruntime / openvino
OPENPOSE_SKELETON = False

# 原生 CTR-GCN
CTRGCN_CHECKPOINT = r"D:\Server_download\runs-49-69874.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 三分类
CLASS_NAMES = ["Normal", "Fall-like", "Fall"]

# 滑窗参数
WINDOW_FRAMES = 75
STRIDE_FRAMES = 15

# 原生 CTR-GCN 输入参数（要和训练时一致）
NUM_CLASS = 3
NUM_POINT = 17
NUM_PERSON = 2
IN_CHANNELS = 3
INFER_CLIP_LEN = 64   # 和训练时 window_size 一致

# tracking 参数
DIST_THR = 100
MAX_MISSED = 20
MIN_HITS = 3
SCORE_THR = 0.4

# COCO17 骨骼
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


# =========================================================
# 2. rtmlib 输出转统一格式
# =========================================================
def rtmlib_to_pose_results(keypoints, scores):
    """
    rtmlib 输出转成和你原来 tracking 代码兼容的格式

    返回:
        frame_res = [
            {
                'keypoints': (17,2),
                'keypoint_scores': (17,),
                'bbox': [x1,y1,x2,y2,score]
            },
            ...
        ]
    """
    frame_res = []

    if keypoints is None or len(keypoints) == 0:
        return frame_res

    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
    if scores.ndim == 1:
        scores = scores[None, ...]

    num_person = keypoints.shape[0]

    for i in range(num_person):
        kpts = keypoints[i]          # (17,2)
        kpt_scores = scores[i]       # (17,)

        valid = kpt_scores > 0.3
        if np.sum(valid) == 0:
            continue

        valid_pts = kpts[valid]
        x1 = float(np.min(valid_pts[:, 0]))
        y1 = float(np.min(valid_pts[:, 1]))
        x2 = float(np.max(valid_pts[:, 0]))
        y2 = float(np.max(valid_pts[:, 1]))
        bbox_score = float(np.mean(kpt_scores[valid]))

        frame_res.append({
            'keypoints': kpts.astype(np.float32),
            'keypoint_scores': kpt_scores.astype(np.float32),
            'bbox': np.array([x1, y1, x2, y2, bbox_score], dtype=np.float32)
        })

    return frame_res


# =========================================================
# 3. Tracking（保留你的逻辑）
# =========================================================
def _legacy_simple_tracking(pose_results_list, dist_thr=100, max_missed=20, min_hits=3, score_thr=0.4):
    next_id = 0
    tracks = {}
    active_tracks = {}

    print("Running Anti-Jitter Tracking...")

    for frame_idx, persons in enumerate(pose_results_list):
        valid_persons = []
        if persons:
            for p in persons:
                if p['bbox'][4] > score_thr:
                    valid_persons.append(p)

        curr_torsos = []
        for p in valid_persons:
            kpts = p['keypoints']
            valid_points = []
            for idx in [5, 6, 11, 12]:
                if kpts[idx][0] > 1 and kpts[idx][1] > 1:
                    valid_points.append(kpts[idx])

            if valid_points:
                torso_center = np.mean(valid_points, axis=0)
            else:
                bbox = p['bbox']
                torso_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            curr_torsos.append(torso_center)

        track_ids = list(active_tracks.keys())
        predicted_centers = []
        for tid in track_ids:
            track = active_tracks[tid]
            pred = track['last_center'] + track['velocity']
            predicted_centers.append(pred)

        matched_track_indices = set()
        matched_person_indices = set()

        if len(track_ids) > 0 and len(curr_torsos) > 0:
            dists = np.zeros((len(track_ids), len(curr_torsos)), dtype=np.float32)
            for i, pred_c in enumerate(predicted_centers):
                for j, curr_c in enumerate(curr_torsos):
                    dists[i, j] = np.linalg.norm(pred_c - curr_c)

            potential_matches = []
            for i in range(len(track_ids)):
                for j in range(len(curr_torsos)):
                    if dists[i, j] < dist_thr:
                        potential_matches.append((dists[i, j], i, j))

            potential_matches.sort(key=lambda x: x[0])

            for d, t_idx, p_idx in potential_matches:
                if t_idx in matched_track_indices or p_idx in matched_person_indices:
                    continue

                tid = track_ids[t_idx]
                track = active_tracks[tid]
                new_center = curr_torsos[p_idx]

                new_velocity = new_center - track['last_center']
                track['velocity'] = 0.7 * new_velocity + 0.3 * track['velocity']
                track['last_center'] = new_center
                track['missed'] = 0
                track['hits'] += 1

                if track['hits'] >= min_hits:
                    if tid not in tracks:
                        tracks[tid] = []
                    valid_persons[p_idx]['track_id'] = tid
                    tracks[tid].append({'frame': frame_idx, 'data': valid_persons[p_idx]})

                matched_track_indices.add(t_idx)
                matched_person_indices.add(p_idx)

        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_track_indices:
                track = active_tracks[tid]
                track['missed'] += 1
                track['hits'] = 0
                track['last_center'] += track['velocity'] * 0.5

                if track['missed'] > max_missed:
                    del active_tracks[tid]

        for p_idx in range(len(valid_persons)):
            if p_idx not in matched_person_indices:
                active_tracks[next_id] = {
                    'last_center': curr_torsos[p_idx],
                    'velocity': np.array([0., 0.], dtype=np.float32),
                    'missed': 0,
                    'hits': 1
                }
                next_id += 1

    print(f"Tracking Done. Valid IDs: {len(tracks)}")

    clean_pose_results = []
    for persons in pose_results_list:
        clean_frame_persons = []
        if persons:
            for p in persons:
                if 'track_id' in p:
                    clean_frame_persons.append(p)
        clean_pose_results.append(clean_frame_persons)

    return clean_pose_results, tracks


# =========================================================
# 4. Smooth（保留你的逻辑）
# =========================================================
def smooth_tracks(tracks, window_size=5):
    print(f"Applying improved smoothing (window_size={window_size})...")

    if window_size == 5:
        weights = np.array([1, 2, 4, 2, 1], dtype=np.float32)
    else:
        weights = np.ones(window_size, dtype=np.float32)

    weights = weights / weights.sum()

    for tid, track_data in tracks.items():
        if len(track_data) < window_size:
            continue

        raw_kpts = np.array([item['data']['keypoints'] for item in track_data], dtype=np.float32)
        T, V, C = raw_kpts.shape
        smoothed_kpts = np.zeros_like(raw_kpts)
        pad = window_size // 2

        for v in range(V):
            for c in range(C):
                signal = raw_kpts[:, v, c]
                padded_signal = np.pad(signal, (pad, pad), mode='edge')
                smooth_signal = np.convolve(padded_signal, weights, mode='valid')
                smoothed_kpts[:, v, c] = smooth_signal

        for i, item in enumerate(track_data):
            item['data']['keypoints'] = smoothed_kpts[i]

    return tracks


# =========================================================
# 5. 原生 CTR-GCN 推理工具
# =========================================================
def simple_tracking(pose_results_list, dist_thr=100, max_missed=20, min_hits=3, score_thr=0.4):
    from tracking_utils import simple_tracking as occlusion_aware_tracking
    return occlusion_aware_tracking(
        pose_results_list,
        dist_thr=dist_thr,
        max_missed=max_missed,
        min_hits=min_hits,
        score_thr=score_thr,
        log_prefix="rtmo_ctrgcn",
    )


def load_checkpoint_weights(path: str):
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt

    raise ValueError(f"Unsupported checkpoint format: {path}")


def build_model():
    model = Model(
        num_class=NUM_CLASS,
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


def sample_frames(x: np.ndarray, target_t: int = 64) -> np.ndarray:
    M, T, V, C = x.shape

    if T == target_t:
        return x

    if T > target_t:
        idx = np.linspace(0, T - 1, target_t).astype(np.int64)
        return x[:, idx, :, :]

    out = np.zeros((M, target_t, V, C), dtype=x.dtype)
    out[:, :T, :, :] = x
    if T > 0:
        out[:, T:, :, :] = x[:, T - 1:T, :, :]
    return out


def fix_num_person(x: np.ndarray, target_m: int = 2) -> np.ndarray:
    M, T, V, C = x.shape

    if M == target_m:
        return x
    if M > target_m:
        return x[:target_m]

    out = np.zeros((target_m, T, V, C), dtype=x.dtype)
    out[:M] = x
    return out


def run_native_ctrgcn_window(model, window_data):
    M = 1
    T = len(window_data)
    V = 17

    keypoint = np.zeros((M, T, V, 2), dtype=np.float32)
    keypoint_score = np.zeros((M, T, V), dtype=np.float32)

    for t, item in enumerate(window_data):
        person = item['data']
        keypoint[0, t] = person['keypoints']
        keypoint_score[0, t] = person['keypoint_scores']

    keypoint = np.nan_to_num(keypoint, nan=0.0, posinf=0.0, neginf=0.0)
    keypoint_score = np.nan_to_num(keypoint_score, nan=0.0, posinf=0.0, neginf=0.0)

    x = np.concatenate([keypoint, keypoint_score[..., None]], axis=-1)  # (M,T,V,3)
    x = fix_num_person(x, NUM_PERSON)
    x = sample_frames(x, INFER_CLIP_LEN)

    x = np.transpose(x, (3, 1, 2, 0))   # (C,T,V,M)
    x = np.expand_dims(x, axis=0).astype(np.float32)  # (1,C,T,V,M)

    x_tensor = torch.from_numpy(x).to(DEVICE)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1)[0]

    probs_np = probs.detach().cpu().numpy().tolist()
    pred_idx = int(torch.argmax(probs).item())
    pred_label = CLASS_NAMES[pred_idx]

    return pred_idx, pred_label, probs_np


# =========================================================
# 6. 主逻辑
# =========================================================
def main():
    print(f"Processing video: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # rtmlib body
    body = Body(
        pose='rtmo',
        to_openpose=OPENPOSE_SKELETON,
        mode='balanced',
        backend=RTM_BACKEND,
        device=RTM_DEVICE
    )

    t_start = time.time()

    # ---- Step 1: rtmlib RTMO ----
    print("Step 1: RTMO Pose Estimation by rtmlib...")

    frames = []
    pose_results = []

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        frames.append(frame.copy())

        s = time.time()
        keypoints, scores = body(frame)
        det_time = time.time() - s
        print(f"[Frame {frame_idx}] det: {det_time:.4f}s")

        frame_res = rtmlib_to_pose_results(keypoints, scores)
        pose_results.append(frame_res)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames read from video.")

    # ---- Step 2: tracking + smoothing ----
    pose_results, tracks = simple_tracking(
        pose_results,
        dist_thr=DIST_THR,
        max_missed=MAX_MISSED,
        min_hits=MIN_HITS,
        score_thr=SCORE_THR
    )
    tracks = smooth_tracks(tracks, window_size=5)

    # ---- Step 3: native CTR-GCN ----
    print("Step 2: Native CTR-GCN Action Recognition...")
    action_model = build_model()
    tracks_labels = {}

    for tid, track_data in tracks.items():
        total_len = len(track_data)
        if total_len < 15:
            continue

        frame_labels = {item['frame']: "Normal" for item in track_data}
        has_fallen = False
        already_reported_fall = False

        for start_idx in range(0, max(1, total_len - WINDOW_FRAMES + STRIDE_FRAMES), STRIDE_FRAMES):
            end_idx = min(start_idx + WINDOW_FRAMES, total_len)
            window_data = track_data[start_idx:end_idx]

            if len(window_data) < 15:
                continue

            pred_idx, pred_label, probs = run_native_ctrgcn_window(action_model, window_data)
            print(probs)

            pred_score = probs[pred_idx]

            if pred_idx == 2 or has_fallen:
                has_fallen = True
                display_label = f"WARNING: FALL ({pred_score:.2f})"

                real_start_f = window_data[0]['frame']
                real_end_f = window_data[-1]['frame']

                if not already_reported_fall:
                    print(f"⚠️ Track ID {tid} [帧 {real_start_f}-{real_end_f}]: {display_label}")
                    already_reported_fall = True

                for item in window_data:
                    f = item['frame']
                    frame_labels[f] = display_label

            elif pred_idx == 1:
                display_label = f"FALL-LIKE ({pred_score:.2f})"

                real_start_f = window_data[0]['frame']
                real_end_f = window_data[-1]['frame']
                print(f"△ Track ID {tid} [帧 {real_start_f}-{real_end_f}]: {display_label}")

                for item in window_data:
                    f = item['frame']
                    if "WARNING" not in frame_labels.get(f, ""):
                        frame_labels[f] = display_label

        tracks_labels[tid] = frame_labels

    # ---- Step 4: visualization ----
    print("Step 3: Visualizing...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    for i, (frame_persons, frame_img) in enumerate(zip(pose_results, frames)):
        img = frame_img.copy()

        if frame_persons:
            for person in frame_persons:
                kpts = person['keypoints']
                scores = person['keypoint_scores']
                bbox = person['bbox']

                tid = person.get('track_id', -1)
                label_dict = tracks_labels.get(tid, {})
                cur_label = label_dict.get(i, "")

                color = (0, 255, 0)  # Normal
                if "WARNING" in cur_label:
                    color = (0, 0, 255)     # Fall
                elif "FALL-LIKE" in cur_label:
                    color = (0, 255, 255)   # Fall-like

                # skeleton
                for p1, p2 in COCO_SKELETON:
                    if scores[p1] > 0.3 and scores[p2] > 0.3:
                        pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                        pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                        cv2.line(img, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

                # keypoints
                for j in range(len(kpts)):
                    if scores[j] > 0.3:
                        pt = (int(kpts[j][0]), int(kpts[j][1]))
                        cv2.circle(img, pt, radius=3, color=color, thickness=-1)

                # bbox
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

                # text
                if cur_label:
                    text = f"ID:{tid} {cur_label}"
                else:
                    text = f"ID:{tid} Normal"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(img, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        video_writer.write(img)

    video_writer.release()

    t_end = time.time()
    infer_time = t_end - t_start

    print(f"Done! Result saved to: {OUTPUT_PATH}")
    print(f"[INFERENCE] Frames: {len(frames)}")
    print(f"[INFERENCE] Time: {infer_time:.3f} s")
    print(f"[INFERENCE] FPS: {len(frames) / infer_time:.2f}")


if __name__ == "__main__":
    main()
