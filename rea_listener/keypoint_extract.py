import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

LEFT_EYE_UPPER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LEFT_EYE_LOWER = [130, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_UPPER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
RIGHT_EYE_LOWER = [359, 384, 385, 386, 387, 388, 466]
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324]
FACE_CONTOUR = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 104, 66, 105, 107]
SELECTED_LANDMARKS = sorted(set(LEFT_EYE_UPPER + LEFT_EYE_LOWER + RIGHT_EYE_UPPER + RIGHT_EYE_LOWER + LIPS_OUTER + LIPS_INNER + FACE_CONTOUR))
DEFAULT_TEMPLATE_104 = Path(__file__).resolve().parent / "assets" / "vico" / "template_keypoint_refine.npy"
DEFAULT_TEMPLATE_468 = Path(__file__).resolve().parent / "assets" / "vico" / "template_keypoint_468.npy"


def load_templates(template_104: str, template_468: str | None):
    import numpy as np

    template_104_path = Path(template_104)
    if not template_104_path.exists():
        raise FileNotFoundError(f"Missing 104-point template: {template_104_path}")

    template_468_path = Path(template_468) if template_468 else None
    if template_468_path is not None and not template_468_path.exists():
        raise FileNotFoundError(f"Missing 468-point template: {template_468_path}")

    return np.load(template_104_path), np.load(template_468_path) if template_468_path else None


def process_video(video_path: str, output_468: str | None, output_104: str, template_104, template_468, allow_missing: bool) -> bool:
    import cv2
    import mediapipe as mp
    import numpy as np
    from tqdm import tqdm

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints_104 = []
    keypoints_468 = []
    progress = tqdm(total=total_frames, desc=os.path.basename(video_path), unit="frame", leave=False)
    success = True

    while capture.isOpened():
        readable, frame = capture.read()
        if not readable:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            keypoints_104.append([[landmarks.landmark[idx].x, landmarks.landmark[idx].y] for idx in SELECTED_LANDMARKS])
            if output_468 is not None:
                keypoints_468.append([[landmarks.landmark[idx].x, landmarks.landmark[idx].y] for idx in range(468)])
        elif allow_missing:
            keypoints_104.append(deepcopy(template_104))
            if output_468 is not None and template_468 is not None:
                keypoints_468.append(deepcopy(template_468))
        else:
            success = False
            break
        progress.update(1)

    progress.close()
    capture.release()
    face_mesh.close()
    if not success:
        return False

    np.save(output_104, np.array(keypoints_104))
    if output_468 is not None and keypoints_468:
        np.save(output_468, np.array(keypoints_468))
    return True


def collect_vico_videos(input_dir: str, metadata_path: str | None) -> list[str]:
    import pandas as pd

    if metadata_path is None:
        return sorted(name for name in os.listdir(input_dir) if name.endswith(".mp4"))
    metadata = pd.read_excel(metadata_path, usecols=["speaker"])
    return [f"{name}.mp4" for name in sorted(metadata["speaker"].drop_duplicates().tolist())]


def collect_realtalk_records(input_dir: str, split_json: str | None) -> tuple[list[str], dict | None]:
    if split_json is None:
        return sorted(name for name in os.listdir(input_dir) if name.endswith(".mp4")), None
    with open(split_json, "r", encoding="utf-8") as file:
        split_data = json.load(file)
    file_names = []
    for subset in ["train", "test"]:
        for record in split_data[subset]:
            file_names.append("_".join([record["id"][0], str(record["id"][1]), str(record["id"][2])]) + ".mp4")
    return sorted(set(file_names)), split_data


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract speaker facial keypoints for REA-Listener training or inference.")
    parser.add_argument("--dataset", type=str, choices=["vico", "realtalk"], required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--template-104", type=str, default=str(DEFAULT_TEMPLATE_104), help="Path to the 104-point fallback template. Defaults to the bundled ViCo template in clean_code.")
    parser.add_argument("--template-468", type=str, default=str(DEFAULT_TEMPLATE_468), help="Path to the 468-point fallback template. Defaults to the bundled ViCo template in clean_code.")
    parser.add_argument("--metadata-path", type=str, default=None, help="ViCo metadata file used to enumerate speaker clips.")
    parser.add_argument("--split-json", type=str, default=None, help="RealTalk split file used to enumerate and optionally filter records.")
    parser.add_argument("--write-filtered-split", type=str, default=None, help="Optional output JSON for filtered RealTalk splits.")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    template_104, template_468 = load_templates(args.template_104, args.template_468)

    if args.dataset == "vico":
        video_names = collect_vico_videos(args.input_dir, args.metadata_path)
        for video_name in video_names:
            stem = video_name[:-4]
            process_video(
                os.path.join(args.input_dir, video_name),
                os.path.join(args.output_dir, f"{stem}_468.npy") if args.template_468 else None,
                os.path.join(args.output_dir, f"{stem}_104.npy"),
                template_104,
                template_468,
                allow_missing=True,
            )
        return

    video_names, split_data = collect_realtalk_records(args.input_dir, args.split_json)
    successful = set()
    for video_name in video_names:
        stem = video_name[:-4]
        if process_video(
            os.path.join(args.input_dir, video_name),
            None,
            os.path.join(args.output_dir, f"{stem}_104.npy"),
            template_104,
            None,
            allow_missing=False,
        ):
            successful.add(stem)

    if args.write_filtered_split and split_data is not None:
        filtered = {"train": [], "test": []}
        for subset in ["train", "test"]:
            for record in split_data[subset]:
                stem = "_".join([record["id"][0], str(record["id"][1]), str(record["id"][2])])
                if stem in successful:
                    filtered[subset].append(record)
        with open(args.write_filtered_split, "w", encoding="utf-8") as file:
            json.dump(filtered, file, indent=2)


if __name__ == "__main__":
    main()
