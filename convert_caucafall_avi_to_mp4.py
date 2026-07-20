import argparse
import csv
import os
import shutil
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CAUCAFall AVI videos to H.264 MP4 while preserving directories."
    )
    parser.add_argument(
        "--input-root",
        default=r"D:\PyCharm\data\Dataset CAUCAFall",
        help="Source CAUCAFall directory containing AVI videos.",
    )
    parser.add_argument(
        "--output-root",
        default=r"D:\PyCharm\data\Dataset CAUCAFall_mp4",
        help="Destination root. Only converted MP4 videos and the manifest are written.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path.")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF; lower is higher quality.")
    parser.add_argument("--preset", default="medium", help="libx264 encoding preset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MP4 files.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned conversions.")
    return parser.parse_args()


def iter_avi_files(root):
    for current_root, _dirs, files in os.walk(root):
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() == ".avi":
                yield os.path.join(current_root, filename)


def parse_progress_frame_count(stdout):
    frame_count = -1
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "frame":
            try:
                frame_count = int(value.strip())
            except ValueError:
                pass
    return frame_count


def decoded_frame_count(path, ffmpeg):
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-i", path,
        "-map", "0:v:0",
        "-progress", "pipe:1",
        "-nostats",
        "-f", "null",
        "-",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_progress_frame_count(completed.stdout)


def convert_video(source, destination, args):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    command = [
        args.ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y" if args.overwrite else "-n",
        "-i", source,
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-preset", args.preset,
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        "-nostats",
        destination,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_progress_frame_count(completed.stdout)


def main():
    args = parse_args()
    if not os.path.isdir(args.input_root):
        raise RuntimeError("Input root does not exist: {}".format(args.input_root))
    if shutil.which(args.ffmpeg) is None and not os.path.isfile(args.ffmpeg):
        raise RuntimeError("ffmpeg was not found: {}".format(args.ffmpeg))

    sources = list(iter_avi_files(args.input_root))
    if not sources:
        raise RuntimeError("No AVI files found under {}".format(args.input_root))

    print("Found {} AVI videos.".format(len(sources)))
    if args.dry_run:
        for source in sources:
            relative = os.path.relpath(source, args.input_root)
            destination = os.path.join(args.output_root, os.path.splitext(relative)[0] + ".mp4")
            print("{} -> {}".format(source, destination))
        return

    os.makedirs(args.output_root, exist_ok=True)
    manifest_rows = []
    failures = 0
    for index, source in enumerate(sources, start=1):
        relative = os.path.relpath(source, args.input_root)
        destination = os.path.join(args.output_root, os.path.splitext(relative)[0] + ".mp4")
        source_frames = -1
        status = "converted"
        error = ""
        print("[{}/{}] {}".format(index, len(sources), relative))
        try:
            if os.path.isfile(destination) and not args.overwrite:
                status = "skipped_existing"
                source_frames = decoded_frame_count(source, args.ffmpeg)
            else:
                source_frames = convert_video(source, destination, args)
            output_frames = decoded_frame_count(destination, args.ffmpeg)
            if output_frames <= 0:
                raise RuntimeError("Converted MP4 contains no decodable video frames")
            if source_frames > 0 and abs(source_frames - output_frames) > 1:
                raise RuntimeError(
                    "Frame count mismatch: source={} output={}".format(source_frames, output_frames)
                )
        except Exception as exc:
            failures += 1
            status = "failed"
            error = str(exc)
            output_frames = -1
            print("  error: {}".format(exc))

        manifest_rows.append({
            "source": source,
            "output": destination,
            "source_frames": source_frames,
            "output_frames": output_frames,
            "status": status,
            "error": error,
        })

    manifest_path = os.path.join(args.output_root, "conversion_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Converted/skipped: {}".format(len(sources) - failures))
    print("Failed: {}".format(failures))
    print("Manifest: {}".format(manifest_path))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
