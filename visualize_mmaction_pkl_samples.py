import argparse
import math
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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

PERSON_COLORS = [
    ((210, 38, 38), (28, 28, 28)),
    ((39, 94, 199), (18, 18, 18)),
]

LABEL_TITLES = {
    42: "A043 falling down",
    79: "A080 squat down",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize selected MMACTION pose samples as GIFs and an overview PNG."
    )
    parser.add_argument(
        "--input-pkl",
        default=r"D:\Server_download\mmaction_120t60\ntu120t60_2d_benchmark.pkl",
        help="Input MMACTION benchmark pickle.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\Tommy\.codex\visualizations\2026\07\08\019f40df-0762-72b0-a07b-9023eed1420d\mmaction_fall_squat",
        help="Directory for GIF and PNG outputs.",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=[42, 79],
        help="Source label ids to visualize.",
    )
    parser.add_argument(
        "--per-label",
        type=int,
        default=2,
        help="How many samples to render for each label.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Minimum keypoint score for rendering joints and bones.",
    )
    parser.add_argument(
        "--max-gif-frames",
        type=int,
        default=96,
        help="Uniformly downsample longer sequences for GIF export.",
    )
    parser.add_argument(
        "--preview-frames",
        type=int,
        default=5,
        help="Number of evenly spaced preview frames per sample in the overview PNG.",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=540,
        help="Canvas width.",
    )
    parser.add_argument(
        "--canvas-height",
        type=int,
        default=540,
        help="Canvas height.",
    )
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=12.0,
        help="GIF playback fps.",
    )
    return parser.parse_args()


def load_fonts():
    for font_name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return (
                ImageFont.truetype(font_name, 20),
                ImageFont.truetype(font_name, 28),
                ImageFont.truetype(font_name, 16),
            )
        except OSError:
            continue
    fallback = ImageFont.load_default()
    return fallback, fallback, fallback


def valid_mask(keypoints, scores, score_thr):
    return (
        np.isfinite(keypoints[:, 0])
        & np.isfinite(keypoints[:, 1])
        & np.isfinite(scores)
        & (scores >= score_thr)
        & (np.any(np.abs(keypoints) > 1e-8, axis=1))
    )


def choose_samples(data, labels, per_label):
    annotations = data["annotations"]
    chosen = []
    for label in labels:
        candidates = []
        for item in annotations:
            if int(item["label"]) != int(label):
                continue
            score = np.asarray(item["keypoint_score"], dtype=np.float32)
            candidates.append(
                {
                    "item": item,
                    "mean_score": float(score.mean()),
                    "total_frames": int(item["total_frames"]),
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["mean_score"],
                abs(row["total_frames"] - 64),
                row["item"]["frame_dir"],
            )
        )
        for row in candidates[:per_label]:
            chosen.append(row["item"])
    return chosen


def compute_view(item, score_thr):
    keypoint = np.asarray(item["keypoint"], dtype=np.float32)
    score = np.asarray(item["keypoint_score"], dtype=np.float32)
    all_points = []
    for person_index in range(keypoint.shape[0]):
        person_kp = keypoint[person_index]
        person_score = score[person_index]
        for frame_index in range(person_kp.shape[0]):
            mask = valid_mask(person_kp[frame_index], person_score[frame_index], score_thr)
            if np.any(mask):
                all_points.append(person_kp[frame_index][mask])

    if not all_points:
        all_points = [np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)]

    stacked = np.concatenate(all_points, axis=0)
    xmin, ymin = np.min(stacked, axis=0)
    xmax, ymax = np.max(stacked, axis=0)
    width = max(0.1, float(xmax - xmin))
    height = max(0.1, float(ymax - ymin))
    pad = max(width, height) * 0.18
    return xmin - pad, ymin - pad, xmax + pad, ymax + pad


def evenly_spaced_indices(length, max_count):
    if length <= max_count:
        return list(range(length))
    return np.linspace(0, length - 1, max_count, dtype=int).tolist()


def render_frames(item, args, title_text):
    keypoint = np.asarray(item["keypoint"], dtype=np.float32)
    score = np.asarray(item["keypoint_score"], dtype=np.float32)
    num_person, num_frames, _, _ = keypoint.shape

    render_indices = evenly_spaced_indices(num_frames, args.max_gif_frames)
    xmin, ymin, xmax, ymax = compute_view(item, args.score_thr)
    view_w = xmax - xmin
    view_h = ymax - ymin

    outer_margin = 28
    header_h = 70
    footer_h = 28
    draw_w = args.canvas_width - outer_margin * 2
    draw_h = args.canvas_height - header_h - footer_h - outer_margin
    scale = min(draw_w / view_w, draw_h / view_h)
    offset_x = outer_margin + (draw_w - view_w * scale) / 2.0
    offset_y = header_h + (draw_h - view_h * scale) / 2.0

    font, font_big, font_small = load_fonts()

    def project(point):
        x, y = point
        return (
            float(offset_x + (x - xmin) * scale),
            float(offset_y + (y - ymin) * scale),
        )

    frames = []
    preview_images = []
    bg = (246, 242, 234)
    faint = (219, 209, 192)
    text = (24, 24, 24)
    muted = (92, 86, 76)

    preview_indices = set(evenly_spaced_indices(len(render_indices), args.preview_frames))

    for render_pos, frame_index in enumerate(render_indices):
        img = Image.new("RGB", (args.canvas_width, args.canvas_height), bg)
        draw = ImageDraw.Draw(img)

        draw.text((24, 14), title_text, fill=text, font=font_big)
        draw.text(
            (24, 44),
            "{} | frame {}/{} | people={}".format(
                item["frame_dir"], frame_index + 1, num_frames, num_person
            ),
            fill=muted,
            font=font_small,
        )
        draw.rectangle(
            (18, header_h - 6, args.canvas_width - 18, args.canvas_height - 18),
            outline=faint,
            width=2,
        )

        for person_index in range(num_person):
            line_color, joint_color = PERSON_COLORS[person_index % len(PERSON_COLORS)]
            person_kp = keypoint[person_index, frame_index]
            person_score = score[person_index, frame_index]

            for p1, p2 in COCO_SKELETON:
                if person_score[p1] >= args.score_thr and person_score[p2] >= args.score_thr:
                    draw.line(
                        [project(person_kp[p1]), project(person_kp[p2])],
                        fill=line_color,
                        width=5,
                    )

            for joint_index, point in enumerate(person_kp):
                if person_score[joint_index] < args.score_thr:
                    continue
                cx, cy = project(point)
                draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=joint_color)

        frames.append(img)
        if render_pos in preview_indices:
            preview_images.append(img.copy())

    return frames, preview_images


def save_gif(frames, output_path, gif_fps):
    duration_ms = int(1000.0 / max(1.0, gif_fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def make_overview(rows, output_path, preview_frames):
    if not rows:
        return None

    font, font_big, font_small = load_fonts()
    thumb_w = 220
    thumb_h = 220
    row_h = 74 + thumb_h
    canvas_w = 320 + preview_frames * (thumb_w + 16) + 24
    canvas_h = 24 + len(rows) * (row_h + 20)
    bg = (251, 248, 241)
    text = (24, 24, 24)
    muted = (92, 86, 76)

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "MMACTION samples: fall vs squat", fill=text, font=font_big)

    y = 58
    for row in rows:
        draw.text((24, y), row["title"], fill=text, font=font)
        draw.text((24, y + 28), row["subtitle"], fill=muted, font=font_small)
        x = 320
        for thumb in row["preview_images"]:
            thumb_resized = thumb.resize((thumb_w, thumb_h))
            canvas.paste(thumb_resized, (x, y))
            x += thumb_w + 16
        y += row_h + 20

    canvas.save(output_path)
    return output_path


def main():
    args = parse_args()
    input_path = Path(args.input_pkl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as file_obj:
        data = pickle.load(file_obj)

    selected_items = choose_samples(data, args.labels, args.per_label)
    if not selected_items:
        raise RuntimeError("No samples matched the requested labels.")

    rows = []
    for item in selected_items:
        label = int(item["label"])
        title = LABEL_TITLES.get(label, "A{:03d}".format(label + 1))
        full_title = "{} | {}".format(title, item["frame_dir"])
        frames, preview_images = render_frames(item, args, full_title)
        gif_path = output_dir / "{}_{}.gif".format(title.split()[0], item["frame_dir"])
        save_gif(frames, gif_path, args.gif_fps)
        print("saved gif:", gif_path)
        rows.append(
            {
                "title": title,
                "subtitle": "{} | frames={} | people={}".format(
                    item["frame_dir"],
                    int(item["total_frames"]),
                    int(np.asarray(item["keypoint"]).shape[0]),
                ),
                "preview_images": preview_images,
            }
        )

    overview_path = output_dir / "overview.png"
    make_overview(rows, overview_path, args.preview_frames)
    print("saved overview:", overview_path)


if __name__ == "__main__":
    main()
