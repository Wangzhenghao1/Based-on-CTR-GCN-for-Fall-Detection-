import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent

# Edit these three defaults directly if you do not want to pass CLI args.
DEFAULT_VIDEO_PATH = r"D:\PyCharm\data\MPFDD-main\Scene_4\S4-P4-F2-FALL-1.mp4"
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "person_skeleton_gifs")
DEFAULT_YOLO_POSE_WEIGHTS = str(PROJECT_ROOT / "yolo26x-pose.pt")

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

TORSO_JOINTS = [5, 6, 11, 12]


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
    kpts_xy = result.keypoints.xy.detach().cpu().numpy()
    kpts_conf = result.keypoints.conf.detach().cpu().numpy()

    order = np.argsort(-boxes_conf)
    if max_people is not None and max_people > 0:
        order = order[:max_people]

    persons = []
    for idx in order:
        bbox = np.array([
            boxes_xyxy[idx][0],
            boxes_xyxy[idx][1],
            boxes_xyxy[idx][2],
            boxes_xyxy[idx][3],
            boxes_conf[idx],
        ], dtype=np.float32)
        persons.append({
            "keypoints": kpts_xy[idx].astype(np.float32),
            "keypoint_scores": kpts_conf[idx].astype(np.float32),
            "bbox": bbox,
        })
    return persons


def torso_center(person):
    kpts = person["keypoints"]
    scores = person["keypoint_scores"]
    valid_points = []
    for index in TORSO_JOINTS:
        if scores[index] > 0.2 and kpts[index][0] > 1 and kpts[index][1] > 1:
            valid_points.append(kpts[index])

    if valid_points:
        return np.mean(valid_points, axis=0)

    bbox = person["bbox"]
    return np.array([
        (bbox[0] + bbox[2]) / 2.0,
        (bbox[1] + bbox[3]) / 2.0,
    ], dtype=np.float32)


def _legacy_simple_tracking(pose_results_list, dist_thr=100, max_missed=20, min_hits=3, score_thr=0.4):
    next_id = 0
    tracks = {}
    active_tracks = {}

    for frame_idx, persons in enumerate(pose_results_list):
        valid_persons = [person for person in persons if person["bbox"][4] >= score_thr]
        curr_centers = [torso_center(person) for person in valid_persons]

        track_ids = list(active_tracks.keys())
        predicted_centers = []
        for track_id in track_ids:
            track = active_tracks[track_id]
            predicted_centers.append(track["last_center"] + track["velocity"])

        matched_track_indices = set()
        matched_person_indices = set()

        if track_ids and curr_centers:
            dists = np.zeros((len(track_ids), len(curr_centers)), dtype=np.float32)
            for track_index, pred_center in enumerate(predicted_centers):
                for person_index, curr_center in enumerate(curr_centers):
                    dists[track_index, person_index] = np.linalg.norm(pred_center - curr_center)

            potential_matches = []
            for track_index in range(len(track_ids)):
                for person_index in range(len(curr_centers)):
                    if dists[track_index, person_index] < dist_thr:
                        potential_matches.append((dists[track_index, person_index], track_index, person_index))
            potential_matches.sort(key=lambda item: item[0])

            for _dist, track_index, person_index in potential_matches:
                if track_index in matched_track_indices or person_index in matched_person_indices:
                    continue

                track_id = track_ids[track_index]
                track = active_tracks[track_id]
                new_center = curr_centers[person_index]
                new_velocity = new_center - track["last_center"]
                track["velocity"] = 0.7 * new_velocity + 0.3 * track["velocity"]
                track["last_center"] = new_center
                track["missed"] = 0
                track["hits"] += 1

                valid_persons[person_index]["track_id"] = track_id
                if track["hits"] >= min_hits:
                    tracks.setdefault(track_id, [])
                    tracks[track_id].append({
                        "frame": frame_idx,
                        "data": valid_persons[person_index],
                    })

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

        for person_index, person in enumerate(valid_persons):
            if person_index in matched_person_indices:
                continue
            active_tracks[next_id] = {
                "last_center": curr_centers[person_index],
                "velocity": np.array([0.0, 0.0], dtype=np.float32),
                "missed": 0,
                "hits": 1,
            }
            next_id += 1

    return tracks


def smooth_tracks(tracks, window_size=5):
    if window_size <= 1:
        return tracks

    if window_size == 5:
        weights = np.array([1, 2, 4, 2, 1], dtype=np.float32)
    else:
        weights = np.ones(window_size, dtype=np.float32)
    weights = weights / weights.sum()
    pad = window_size // 2

    for track_data in tracks.values():
        if len(track_data) < window_size:
            continue

        raw_kpts = np.array([item["data"]["keypoints"] for item in track_data], dtype=np.float32)
        smoothed_kpts = np.zeros_like(raw_kpts)
        for joint_index in range(raw_kpts.shape[1]):
            for coord_index in range(raw_kpts.shape[2]):
                signal = raw_kpts[:, joint_index, coord_index]
                padded_signal = np.pad(signal, (pad, pad), mode="edge")
                smoothed_kpts[:, joint_index, coord_index] = np.convolve(padded_signal, weights, mode="valid")

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
        log_prefix="skeleton_gif",
    )
    return tracks


def collect_pose_results(args):
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(args.video))

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25.0

    model = YOLO(args.weights)
    pose_results = []
    raw_frame_index = 0
    kept_frame_index = 0
    start_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break
        raw_frame_index += 1

        if args.max_frames > 0 and raw_frame_index > args.max_frames:
            break
        if (raw_frame_index - 1) % args.frame_step != 0:
            continue

        result = model.predict(
            source=frame,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]
        pose_results.append(yolo_pose_to_persons(result, max_people=args.max_people))
        kept_frame_index += 1

        if kept_frame_index == 1 or kept_frame_index % args.log_interval == 0:
            elapsed = time.time() - start_time
            print("[pose] kept_frame={} raw_frame={} elapsed={:.1f}s".format(
                kept_frame_index, raw_frame_index, elapsed
            ))

    cap.release()
    if not pose_results:
        raise RuntimeError("No frames were processed.")

    effective_fps = source_fps / max(1, args.frame_step)
    return pose_results, effective_fps


def compute_track_view(track_data, keypoint_thr, pad_ratio=0.18):
    points = []
    for item in track_data:
        kpts = item["data"]["keypoints"]
        scores = item["data"]["keypoint_scores"]
        mask = scores >= keypoint_thr
        if np.any(mask):
            points.append(kpts[mask])
    if not points:
        for item in track_data:
            bbox = item["data"]["bbox"]
            points.append(np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32))

    all_points = np.concatenate(points, axis=0)
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)
    width = max(1.0, float(xmax - xmin))
    height = max(1.0, float(ymax - ymin))
    pad = max(width, height) * pad_ratio
    return xmin - pad, ymin - pad, xmax + pad, ymax + pad


def load_fonts():
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, 18), ImageFont.truetype(name, 24)
        except OSError:
            continue
    font = ImageFont.load_default()
    return font, font


def render_track_gif(track_id, track_data, output_dir, args, effective_fps):
    if len(track_data) < args.min_track_frames:
        return None

    if len(track_data) > args.max_gif_frames > 0:
        indices = np.linspace(0, len(track_data) - 1, args.max_gif_frames, dtype=int)
        render_items = [track_data[int(index)] for index in indices]
    else:
        render_items = track_data

    xmin, ymin, xmax, ymax = compute_track_view(render_items, args.keypoint_thr)
    view_w = max(1.0, xmax - xmin)
    view_h = max(1.0, ymax - ymin)
    canvas_w, canvas_h = args.canvas_width, args.canvas_height
    margin_x, margin_top, margin_bottom = 42, 78, 42
    draw_w = canvas_w - 2 * margin_x
    draw_h = canvas_h - margin_top - margin_bottom
    scale = min(draw_w / view_w, draw_h / view_h)
    offset_x = margin_x + (draw_w - view_w * scale) / 2.0
    offset_y = margin_top + (draw_h - view_h * scale) / 2.0

    def project(point):
        x, y = point
        return (
            float(offset_x + (x - xmin) * scale),
            float(offset_y + (y - ymin) * scale),
        )

    font, font_big = load_fonts()
    duration_ms = int(1000 / max(1.0, min(args.gif_fps, effective_fps)))
    frames = []
    bg = (247, 243, 232)
    red = (210, 38, 38)
    black = (18, 18, 18)
    muted = (82, 74, 64)
    faint = (216, 205, 184)

    for item in render_items:
        img = Image.new("RGB", (canvas_w, canvas_h), bg)
        draw = ImageDraw.Draw(img)
        frame_id = item["frame"]
        person = item["data"]
        kpts = person["keypoints"]
        scores = person["keypoint_scores"]

        draw.text((24, 16), "Person track {}".format(track_id), fill=black, font=font_big)
        draw.text((24, 48), "frame={}  score={:.2f}  detections={}".format(
            frame_id + 1, float(person["bbox"][4]), len(track_data)
        ), fill=muted, font=font)
        draw.rectangle((18, 72, canvas_w - 18, canvas_h - 18), outline=faint, width=2)

        for p1, p2 in COCO_SKELETON:
            if scores[p1] >= args.keypoint_thr and scores[p2] >= args.keypoint_thr:
                draw.line([project(kpts[p1]), project(kpts[p2])], fill=red, width=args.line_width)

        radius = args.joint_radius
        for joint_index, point in enumerate(kpts):
            if scores[joint_index] < args.keypoint_thr:
                continue
            cx, cy = project(point)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=black)

        frames.append(img)

    output_path = output_dir / "person_{:03d}_{}frames.gif".format(track_id, len(track_data))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return output_path


def write_summary(summary_path, rows):
    with summary_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=[
            "track_id",
            "frames",
            "first_frame",
            "last_frame",
            "gif_path",
        ])
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO pose on a video and export one skeleton-only GIF per tracked person."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output GIF files.")
    parser.add_argument("--weights", default=DEFAULT_YOLO_POSE_WEIGHTS, help="YOLO pose weights path.")
    parser.add_argument("--device", default=default_device(), help="cuda, cpu, or CUDA index.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-people", type=int, default=0, help="Keep only top N people per frame. 0 means no limit.")
    parser.add_argument("--score-thr", type=float, default=0.4, help="Tracking bbox confidence threshold.")
    parser.add_argument("--keypoint-thr", type=float, default=0.3, help="Keypoint confidence threshold for rendering.")
    parser.add_argument("--dist-thr", type=float, default=100.0, help="Tracking center-distance threshold.")
    parser.add_argument("--max-missed", type=int, default=20, help="Max missed frames before a track is dropped.")
    parser.add_argument("--min-hits", type=int, default=3, help="Minimum consecutive hits before a track is exported.")
    parser.add_argument("--min-track-frames", type=int, default=8, help="Skip tracks shorter than this.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Odd smoothing window for keypoints. 1 disables smoothing.")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many raw frames. 0 means full video.")
    parser.add_argument("--max-gif-frames", type=int, default=160, help="Uniformly downsample long tracks in GIF. 0 disables cap.")
    parser.add_argument("--gif-fps", type=float, default=12.0, help="Output GIF playback FPS cap.")
    parser.add_argument("--canvas-width", type=int, default=640, help="Output GIF width.")
    parser.add_argument("--canvas-height", type=int, default=480, help="Output GIF height.")
    parser.add_argument("--line-width", type=int, default=5, help="Skeleton line width.")
    parser.add_argument("--joint-radius", type=int, default=5, help="Skeleton joint radius.")
    parser.add_argument("--log-interval", type=int, default=30, help="Progress print interval in processed frames.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.video = str(Path(args.video))
    args.weights = str(Path(args.weights))
    if args.max_people <= 0:
        args.max_people = None
    args.frame_step = max(1, args.frame_step)
    args.log_interval = max(1, args.log_interval)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Video:", args.video)
    print("YOLO pose weights:", args.weights)
    print("Output dir:", output_dir)

    pose_results, effective_fps = collect_pose_results(args)
    print("Pose frames:", len(pose_results))
    print("Effective FPS:", "{:.2f}".format(effective_fps))

    tracks = simple_tracking(
        pose_results,
        dist_thr=args.dist_thr,
        max_missed=args.max_missed,
        min_hits=args.min_hits,
        score_thr=args.score_thr,
    )
    tracks = smooth_tracks(tracks, window_size=args.smooth_window)
    print("Tracks:", len(tracks))

    summary_rows = []
    for track_id in sorted(tracks):
        track_data = tracks[track_id]
        gif_path = render_track_gif(track_id, track_data, output_dir, args, effective_fps)
        if gif_path is None:
            print("[skip] track={} frames={} shorter than min_track_frames={}".format(
                track_id, len(track_data), args.min_track_frames
            ))
            continue
        first_frame = track_data[0]["frame"] + 1
        last_frame = track_data[-1]["frame"] + 1
        summary_rows.append({
            "track_id": track_id,
            "frames": len(track_data),
            "first_frame": first_frame,
            "last_frame": last_frame,
            "gif_path": str(gif_path),
        })
        print("[gif] track={} frames={} range={}..{} -> {}".format(
            track_id, len(track_data), first_frame, last_frame, gif_path
        ))

    summary_path = output_dir / "tracks_summary.csv"
    write_summary(summary_path, summary_rows)
    print("Summary:", summary_path)
    print("Done. Exported GIFs:", len(summary_rows))


if __name__ == "__main__":
    main()
