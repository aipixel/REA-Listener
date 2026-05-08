#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import argparse
import os
from pathlib import Path

import lightning
import numpy as np
import torch
import torchvision
from tqdm.rich import tqdm

from core.data import DriverData
from core.libs.path_utils import (
    project_root,
    resolve_resume_path,
    resolve_tracked_cache_path,
    set_runtime_overrides,
)
from core.libs.utils import ConfigDict
from core.models import build_model


PROJECT_ROOT = project_root()


def build_track_engine(device: str = "cuda", matting_type: str = "old"):
    from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine

    return TrackEngine(focal_length=12.0, device=device, matting_type=matting_type)


def build_default_dump_dir(meta_cfg: ConfigDict, save_dir: str | None) -> Path:
    base_dir = PROJECT_ROOT / "render_results"
    model_name = meta_cfg.MODEL.NAME.split("_")[0]
    if save_dir:
        return base_dir / save_dir / model_name
    return base_dir / model_name


def inference(
    image_path,
    driver_path,
    resume_path=None,
    force_retrack=False,
    device="cuda",
    save_name=None,
    save_dir=None,
    only_rendered_results=False,
    matting_type="old",
    fps=30,
    with_gt=False,
    checkpoints_dir=None,
    assets_dir=None,
    track_assets_dir=None,
    tracked_cache=None,
    cache_dir=None,
    dinov2_dir=None,
):
    del with_gt
    set_runtime_overrides(
        checkpoints_dir=checkpoints_dir,
        assets_dir=assets_dir,
        track_assets_dir=track_assets_dir,
        tracked_cache_path=tracked_cache,
        cache_dir=cache_dir,
        dinov2_dir=dinov2_dir,
        resume_path=resume_path,
    )

    lightning.fabric.seed_everything(42)
    driver_path = driver_path[:-1] if driver_path.endswith("/") else driver_path
    driver_name = os.path.basename(driver_path).split(".")[0]
    resolved_resume_path = str(resolve_resume_path(resume_path))

    print("Loading model...")
    lightning_fabric = lightning.Fabric(accelerator=device, strategy="auto", devices=[0])
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resolved_resume_path)
    meta_cfg = ConfigDict(init_dict=full_checkpoint["meta_cfg"])
    model = build_model(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint["model"])
    model = lightning_fabric.setup(model)
    model.eval()
    print(str(meta_cfg))

    track_engine = None

    def get_track_engine():
        nonlocal track_engine
        if track_engine is None:
            track_engine = build_track_engine(device=device, matting_type=matting_type)
        return track_engine

    feature_name = os.path.basename(image_path).split(".")[0]
    feature_data = get_tracked_results(
        image_path,
        track_engine_factory=get_track_engine,
        force_retrack=force_retrack,
        tracked_cache_path=tracked_cache,
    )
    if feature_data is None:
        print(f"Finish inference, no face in input: {image_path}.")
        return

    if os.path.isdir(driver_path):
        driver_name = os.path.basename(driver_path)
        driver_dataset = DriverData(driver_path, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
        driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)
    else:
        driver_name = os.path.basename(driver_path).split(".")[0]
        driver_data = get_tracked_results(
            driver_path,
            track_engine_factory=get_track_engine,
            force_retrack=force_retrack,
            tracked_cache_path=tracked_cache,
        )
        if driver_data is None:
            print(f"Finish inference, no face in driver: {driver_path}.")
            return
        driver_dataset = DriverData({driver_name: driver_data}, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
        driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)

    print("track loader done")
    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)
    images = []
    dump_dir = build_default_dump_dir(meta_cfg, save_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    if only_rendered_results:
        for batch in tqdm(driver_dataloader):
            render_results = model.forward_expression(batch)
            pred_sr_rgb = render_results["sr_gen_image"].clamp(0, 1)
            visualize_rgbs = torchvision.utils.make_grid([pred_sr_rgb[0]], nrow=4, padding=0)
            images.append(visualize_rgbs.cpu())
    else:
        for batch in tqdm(driver_dataloader):
            render_results = model.forward_expression(batch)
            gt_rgb = render_results["t_image"].clamp(0, 1)
            pred_sr_rgb = render_results["sr_gen_image"].clamp(0, 1)
            visualize_rgbs = torchvision.utils.make_grid([gt_rgb[0], pred_sr_rgb[0]], nrow=4, padding=0)
            images.append(visualize_rgbs.cpu())

    if driver_dataset._is_video:
        if save_name is not None:
            dump_path = Path(save_name)
        else:
            dump_path = dump_dir / f"{driver_name}_{feature_name}.mp4"
        merged_images = torch.stack(images)
        merged_images = (merged_images * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        torchvision.io.write_video(str(dump_path), merged_images, fps=fps)
    else:
        dump_path = dump_dir / f"{driver_name}_{feature_name}.jpg"
        merged_images = torchvision.utils.make_grid(images, nrow=5, padding=0)
        feature_images = torchvision.utils.make_grid(
            [feature_data["image"]] * (merged_images.shape[-2] // 512),
            nrow=1,
            padding=0,
        )
        merged_images = torch.cat([feature_images, merged_images], dim=-1)
        torchvision.utils.save_image(merged_images, str(dump_path))

    print(f"Finish inference: {dump_path}.")


def get_tracked_results(
    image_path,
    track_engine_factory=None,
    force_retrack=False,
    tracked_cache_path=None,
):
    if not is_image(image_path):
        print(f"Please input an image path, got {image_path}.")
        return None

    tracked_pt_path = resolve_tracked_cache_path(tracked_cache_path)
    tracked_pt_path.parent.mkdir(parents=True, exist_ok=True)
    if tracked_pt_path.exists():
        tracked_data = torch.load(tracked_pt_path, weights_only=False)
    else:
        tracked_data = {}

    image_path = str(Path(image_path))
    image_base = os.path.basename(image_path)
    if image_base in tracked_data and not force_retrack:
        print(f"Load tracking result from cache: {tracked_pt_path}.")
    else:
        if track_engine_factory is None:
            raise RuntimeError(
                "Tracking cache is missing and no tracking engine is available. "
                "Provide the tracking assets explicitly before running render."
            )
        print(f"Tracking {image_path}...")
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float()
        track_engine = track_engine_factory()
        tracked_feature_data = track_engine.track_image([image], [image_base])
        if tracked_feature_data is None or image_base not in tracked_feature_data:
            print(f"No face detected in {image_path}.")
            return None

        feature_data = tracked_feature_data[image_base]
        tracked_data[image_base] = feature_data
        if "vis_image" in feature_data:
            torchvision.utils.save_image(
                torch.tensor(feature_data["vis_image"]),
                str(tracked_pt_path.parent / f"{Path(image_base).stem}.jpg"),
            )

        image_dir = Path(image_path).parent
        other_names = [path.name for path in image_dir.iterdir() if path.is_file() and is_image(path.name)]
        other_paths = [image_dir / name for name in other_names]
        if len(other_paths) <= 35:
            print("Track on all images in this folder to save time.")
            other_images = [
                torchvision.io.read_image(str(other_path), mode=torchvision.io.ImageReadMode.RGB).float()
                for other_path in other_paths
            ]
            try:
                other_feature_data = track_engine.track_image(other_images, other_names)
                if other_feature_data is not None:
                    for key, value in other_feature_data.items():
                        if "vis_image" in value:
                            torchvision.utils.save_image(
                                torch.tensor(value["vis_image"]),
                                str(tracked_pt_path.parent / f"{Path(key).stem}.jpg"),
                            )
                    tracked_data.update(other_feature_data)
            except Exception as exc:
                print(f"Error: {exc}.")
        torch.save(tracked_data, tracked_pt_path)

    feature_data = tracked_data[image_base]
    for key in list(feature_data.keys()):
        if isinstance(feature_data[key], np.ndarray):
            feature_data[key] = torch.tensor(feature_data[key])
    return feature_data


def is_image(image_path):
    extension_name = image_path.split(".")[-1].lower()
    return extension_name in ["jpg", "png", "jpeg"]


def add_water_mark(image, water_mark):
    _water_mark_rgb = water_mark[None, :3]
    _water_mark_alpha = water_mark[None, 3:4].expand(-1, 3, -1, -1) * 0.8
    _mark_patch = image[..., -water_mark.shape[-2] :, -water_mark.shape[-1] :]
    _mark_patch = _mark_patch * (1 - _water_mark_alpha) + _water_mark_rgb * _water_mark_alpha
    image[..., -water_mark.shape[-2] :, -water_mark.shape[-1] :] = _mark_patch
    return image


def build_camera(angle, ori_transforms=None, device="cuda"):
    from pytorch3d.renderer.cameras import look_at_view_transform

    if ori_transforms is None:
        distance = 9.3
    else:
        distance = ori_transforms[..., 3].square().sum(dim=-1).sqrt()[0].item() * 1.0
        device = ori_transforms.device
    print(f"Camera distance: {distance}, angle: {angle}.")
    R, T = look_at_view_transform(distance, 5, angle, device=device)
    rotate_trans = torch.cat([R, T[:, :, None]], dim=-1)
    return rotate_trans


def speed_test():
    driver_path = str(PROJECT_ROOT / "demos" / "vfhq_driver")
    resume_path = str(resolve_resume_path())
    lightning.fabric.seed_everything(42)
    print("Loading model...")
    lightning_fabric = lightning.Fabric(accelerator="cuda", strategy="auto", devices=[0])
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resume_path)
    meta_cfg = ConfigDict(init_dict=full_checkpoint["meta_cfg"])
    model = build_model(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint["model"])
    model = lightning_fabric.setup(model)
    print(str(meta_cfg))
    driver_dataset = DriverData(driver_path, None, meta_cfg.DATASET.POINT_PLANE_SIZE)
    driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)
    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)
    for batch in tqdm(driver_dataloader):
        render_results = model.forward_expression(batch)
        _ = render_results["t_image"].clamp(0, 1)
        _ = render_results["sr_gen_image"].clamp(0, 1)
    print("Finish speed test.")


if __name__ == "__main__":
    import warnings

    from tqdm.std import TqdmExperimentalWarning

    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", "--image_path", "-i", required=True, type=str, dest="image_path")
    parser.add_argument("--driver-path", "--driver_path", "-d", required=True, type=str, dest="driver_path")
    parser.add_argument("--force-retrack", "--force_retrack", "-f", action="store_true", dest="force_retrack")
    parser.add_argument("--with-gt", action="store_true")
    parser.add_argument("--resume-path", "--resume_path", "-r", default=None, type=str, dest="resume_path")
    parser.add_argument("--checkpoints-dir", default=None, type=str)
    parser.add_argument("--assets-dir", default=None, type=str)
    parser.add_argument("--track-assets-dir", default=None, type=str)
    parser.add_argument("--tracked-cache", default=None, type=str)
    parser.add_argument("--cache-dir", default=None, type=str)
    parser.add_argument("--dinov2-dir", default=None, type=str)
    parser.add_argument("--save-name", "--save_name", "-o", default="noname", type=str, dest="save_name")
    parser.add_argument("--save-dir", "--save_dir", "-s", default="", type=str, dest="save_dir")
    parser.add_argument("--only-rendered-results", "--only_rendered_results", action="store_true", dest="only_rendered_results")
    parser.add_argument("--matting-type", "--matting_type", default="old", type=str, help="old, new", dest="matting_type")
    parser.add_argument("--fps", default=30, type=int)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    if args.save_name == "noname":
        args.save_name = None
    if args.save_dir == "":
        args.save_dir = None

    inference(
        args.image_path,
        args.driver_path,
        args.resume_path,
        args.force_retrack,
        save_name=args.save_name,
        save_dir=args.save_dir,
        only_rendered_results=args.only_rendered_results,
        matting_type=args.matting_type,
        fps=args.fps,
        with_gt=args.with_gt,
        checkpoints_dir=args.checkpoints_dir,
        assets_dir=args.assets_dir,
        track_assets_dir=args.track_assets_dir,
        tracked_cache=args.tracked_cache,
        cache_dir=args.cache_dir,
        dinov2_dir=args.dinov2_dir,
    )
