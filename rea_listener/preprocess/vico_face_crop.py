import argparse
import os
from pathlib import Path

DEFAULT_OVERRIDE_BOXES = Path(__file__).resolve().parents[1] / "assets" / "vico" / "override_boxes_original.csv"


def load_video_file_clip():
    try:
        from moviepy.editor import VideoFileClip
    except ModuleNotFoundError:
        from moviepy import VideoFileClip
    return VideoFileClip


def expand_bbox_to_square_and_margin_with_shift(bbox, image_width, image_height, margin_ratio=0.15):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    new_side = max(width, height)
    margin = new_side * margin_ratio

    x_min_expanded = center_x - new_side / 2 - margin
    x_max_expanded = center_x + new_side / 2 + margin
    y_min_expanded = center_y - new_side / 2 - margin
    y_max_expanded = center_y + new_side / 2 + margin

    dx, dy = 0, 0
    if x_min_expanded < 0:
        dx = -x_min_expanded
    elif x_max_expanded > image_width:
        dx = image_width - x_max_expanded
    if y_min_expanded < 0:
        dy = -y_min_expanded
    elif y_max_expanded > image_height:
        dy = image_height - y_max_expanded

    x_min_final = max(0, min(image_width, x_min_expanded + dx))
    x_max_final = max(0, min(image_width, x_max_expanded + dx))
    y_min_final = max(0, min(image_height, y_min_expanded + dy))
    y_max_final = max(0, min(image_height, y_max_expanded + dy))
    return [int(x_min_final), int(y_min_final), int(x_max_final), int(y_max_final)]


def get_bbox_from_multi_frames(face_aligner, video):
    import cv2
    import numpy as np

    x_min_list, x_max_list, y_min_list, y_max_list = [], [], [], []
    for sample_time in [video.duration * factor for factor in [0.0, 0.5, 0.9]]:
        frame = video.get_frame(sample_time)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = face_aligner.get_landmarks(rgb_image)
        if not predictions:
            continue
        max_size = 0
        bbox = None
        for face_landmarks in predictions:
            x_coords = face_landmarks[:, 0]
            y_coords = face_landmarks[:, 1]
            x_min, y_min = np.min(x_coords), np.min(y_coords)
            x_max, y_max = np.max(x_coords), np.max(y_coords)
            y_min = int(y_min - (0.3 * (y_max - y_min)))
            area = (y_max - y_min) * (x_max - x_min)
            if area > max_size:
                max_size = area
                bbox = [x_min, y_min, x_max, y_max]
        if bbox is not None:
            x_min_list.append(bbox[0])
            x_max_list.append(bbox[2])
            y_min_list.append(bbox[1])
            y_max_list.append(bbox[3])
    if not x_min_list:
        raise RuntimeError("No face detected in sampled frames.")
    return [np.min(x_min_list), np.min(y_min_list), np.max(x_max_list), np.max(y_max_list)]


def crop_and_save_video(video_path: str, output_path: str, bbox_override=None) -> None:
    import face_alignment

    VideoFileClip = load_video_file_clip()

    face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    with VideoFileClip(video_path) as video:
        bbox = bbox_override if bbox_override is not None else get_bbox_from_multi_frames(face_aligner, video)
        first_frame = video.get_frame(0)
        image_height, image_width, _ = first_frame.shape
        bbox = expand_bbox_to_square_and_margin_with_shift(bbox, image_width, image_height, 0.15)
        cropped = video.cropped(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]) if hasattr(video, "cropped") else video.crop(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
        resized = cropped.resized(new_size=(256, 256)) if hasattr(cropped, "resized") else cropped.resize(newsize=(256, 256))
        resized.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=30.0)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop ViCo speaker/listener faces from the original paired videos.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--override-boxes", type=str, default=str(DEFAULT_OVERRIDE_BOXES), help="Optional CSV with columns filename,x_min,y_min,x_max,y_max. Defaults to the bundled ViCo manual overrides from the original project.")
    return parser


def main() -> None:
    import pandas as pd

    args = get_parser().parse_args()
    metadata = pd.read_excel(args.metadata_path)
    listener_names = set(metadata["listener"].tolist())
    speaker_names = set(metadata["speaker"].tolist())
    override_boxes = {}
    if args.override_boxes:
        overrides = pd.read_csv(args.override_boxes)
        for row in overrides.itertuples(index=False):
            override_boxes[row.filename] = [row.x_min, row.y_min, row.x_max, row.y_max]

    listener_output = os.path.join(args.output_dir, "listener")
    speaker_output = os.path.join(args.output_dir, "speaker")
    os.makedirs(listener_output, exist_ok=True)
    os.makedirs(speaker_output, exist_ok=True)

    file_names = sorted(name for name in os.listdir(args.input_dir) if name.endswith(".mp4"))
    for file_name in file_names[args.start_index :]:
        stem = file_name[:-4]
        if stem in listener_names:
            output_path = os.path.join(listener_output, file_name)
        elif stem in speaker_names:
            output_path = os.path.join(speaker_output, file_name)
        else:
            continue
        crop_and_save_video(os.path.join(args.input_dir, file_name), output_path, override_boxes.get(file_name))


if __name__ == "__main__":
    main()
