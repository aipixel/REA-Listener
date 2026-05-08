import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

from rea_listener.paths import build_realtalk_layout, build_vico_layout


def run_gagavatar_inference(row: dict) -> None:
    inference_path = Path(row["gagavatar_dir"]) / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError(f"GAGAvatar inference entrypoint not found: {inference_path}")

    command = [
        sys.executable,
        str(inference_path),
        "-i",
        row["reference_image"],
        "-d",
        row["driver_path"],
        "-o",
        row["save_name"],
    ]
    if row["gagavatar_resume_path"]:
        command.extend(["--resume-path", row["gagavatar_resume_path"]])
    if row["gagavatar_checkpoints_dir"]:
        command.extend(["--checkpoints-dir", row["gagavatar_checkpoints_dir"]])
    if row["gagavatar_assets_dir"]:
        command.extend(["--assets-dir", row["gagavatar_assets_dir"]])
    if row["gagavatar_track_assets_dir"]:
        command.extend(["--track-assets-dir", row["gagavatar_track_assets_dir"]])
    if row["gagavatar_tracked_cache"]:
        command.extend(["--tracked-cache", row["gagavatar_tracked_cache"]])
    if row["gagavatar_cache_dir"]:
        command.extend(["--cache-dir", row["gagavatar_cache_dir"]])
    if row["dataset"] == "realtalk":
        command.extend(["--fps", "25"])
    if row["force_retrack"]:
        command.append("--force_retrack")
    if not row["with_gt"]:
        command.append("--only_rendered_results")

    if os.path.exists(row["save_name"]):
        return
    subprocess.run(command, cwd=row["gagavatar_dir"], check=True)


def build_vico_rows(args: argparse.Namespace) -> list[dict]:
    layout = build_vico_layout(args.data_root)
    metadata = pd.read_excel(layout.metadata_path)
    subset = metadata[metadata["data_split"].isin(["ood", "test"])]
    num_render = len(subset) if args.num == -1 else args.num
    rows = []
    for index in range(num_render):
        row = subset.iloc[index].to_dict()
        rows.append(
            {
                "dataset": "vico",
                "reference_image": str(layout.reference_image_dir / f"{row['listener']}.png"),
                "driver_path": os.path.join(args.coef_path, row["listener"]),
                "save_name": os.path.join(args.save_path, row["data_split"], f"{row['listener']}.mp4"),
                "force_retrack": args.force_retrack,
                "with_gt": args.with_gt,
                "gagavatar_dir": args.gagavatar_dir,
                "gagavatar_resume_path": args.gagavatar_resume_path,
                "gagavatar_checkpoints_dir": args.gagavatar_checkpoints_dir,
                "gagavatar_assets_dir": args.gagavatar_assets_dir,
                "gagavatar_track_assets_dir": args.gagavatar_track_assets_dir,
                "gagavatar_tracked_cache": args.gagavatar_tracked_cache,
                "gagavatar_cache_dir": args.gagavatar_cache_dir,
            }
        )
    return rows


def build_realtalk_rows(args: argparse.Namespace) -> list[dict]:
    layout = build_realtalk_layout(args.data_root)
    with open(layout.split_path, "r", encoding="utf-8") as file:
        data = json.load(file)["test"]
    num_render = len(data) if args.num == -1 else args.num
    rows = []
    for index in range(num_render):
        item = deepcopy(data[index])
        name = "_".join(map(str, item["id"]))
        rows.append(
            {
                "dataset": "realtalk",
                "reference_image": str(layout.reference_image_dir / f"{name}.png"),
                "driver_path": os.path.join(args.coef_path, name),
                "save_name": os.path.join(args.save_path, f"{name}.mp4"),
                "force_retrack": args.force_retrack,
                "with_gt": args.with_gt,
                "gagavatar_dir": args.gagavatar_dir,
                "gagavatar_resume_path": args.gagavatar_resume_path,
                "gagavatar_checkpoints_dir": args.gagavatar_checkpoints_dir,
                "gagavatar_assets_dir": args.gagavatar_assets_dir,
                "gagavatar_track_assets_dir": args.gagavatar_track_assets_dir,
                "gagavatar_tracked_cache": args.gagavatar_tracked_cache,
                "gagavatar_cache_dir": args.gagavatar_cache_dir,
            }
        )
    return rows


def get_parser() -> argparse.ArgumentParser:
    default_gagavatar_dir = Path(__file__).resolve().parents[1] / "GAGAvatar"
    parser = argparse.ArgumentParser(description="Render predicted coefficients with the vendored GAGAvatar code.")
    parser.add_argument("--dataset", type=str, choices=["vico", "realtalk"], required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--coef-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--gagavatar-dir", type=str, default=str(default_gagavatar_dir))
    parser.add_argument("--gagavatar-resume-path", type=str, default=None)
    parser.add_argument("--gagavatar-checkpoints-dir", type=str, default=None)
    parser.add_argument("--gagavatar-assets-dir", type=str, default=None)
    parser.add_argument("--gagavatar-track-assets-dir", type=str, default=None)
    parser.add_argument("--gagavatar-tracked-cache", type=str, default=None)
    parser.add_argument("--gagavatar-cache-dir", type=str, default=None)
    parser.add_argument("--force-retrack", action="store_true")
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--with-gt", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    rows = build_vico_rows(args) if args.dataset == "vico" else build_realtalk_rows(args)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(run_gagavatar_inference, rows))


if __name__ == "__main__":
    main()
