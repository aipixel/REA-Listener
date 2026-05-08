import argparse
import os


def load_video_file_clip():
    try:
        from moviepy.editor import VideoFileClip
    except ModuleNotFoundError:
        from moviepy import VideoFileClip
    return VideoFileClip


def trim_video(input_path: str, output_path: str, trim_start: float = 0.0, trim_end: float = 0.0) -> None:
    VideoFileClip = load_video_file_clip()

    with VideoFileClip(input_path) as video:
        start_time = trim_start
        end_time = max(start_time, video.duration - trim_end)
        subclip = video.subclipped(start_time, end_time) if hasattr(video, "subclipped") else video.subclip(start_time, end_time)
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply manual timing fixes to a small list of problematic ViCo clips.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--trim-start", nargs="*", default=[], help="Video filenames that should drop one second from the beginning.")
    parser.add_argument("--trim-end", nargs="*", default=[], help="Video filenames that should drop one second from the end.")
    parser.add_argument("--seconds", type=float, default=1.0)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for video_name in args.trim_start:
        trim_video(os.path.join(args.input_dir, video_name), os.path.join(args.output_dir, video_name), trim_start=args.seconds)
    for video_name in args.trim_end:
        trim_video(os.path.join(args.input_dir, video_name), os.path.join(args.output_dir, video_name), trim_end=args.seconds)


if __name__ == "__main__":
    main()
