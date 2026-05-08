from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
LEGACY_ASSETS_ROOT = PROJECT_ROOT / "assets"
LEGACY_TRACK_ASSETS_ROOT = PROJECT_ROOT / "core" / "libs" / "GAGAvatar_track" / "assets"

TRACK_CHECKPOINT_LAYOUT = {
    "emica/EMICA-CVT_flame2020_notexture.pt": "emica/EMICA-CVT_flame2020_notexture.pt",
    "emica/ins_scrfd_10g_bnkps.onnx": "emica/ins_scrfd_10g_bnkps.onnx",
    "vgghead/vgg_heads_l.trcd": "vgghead/vgg_heads_l.trcd",
    "vgghead/lmks_2d.pt": "vgghead/lmks_2d.pt",
    "matting/stylematte_synth.pt": "matting/stylematte_synth.pt",
    "flame/FLAME_with_eye.pt": "FLAME_with_eye.pt",
}

MASK2FORMER_MODEL_DIRNAME = "mask2former-swin-tiny-coco-instance"


def project_root() -> Path:
    return PROJECT_ROOT


def _normalize_path(path_like: str | os.PathLike[str] | None) -> Path | None:
    if not path_like:
        return None
    return Path(path_like).expanduser()


def resolve_existing_path(description: str, *candidates: str | os.PathLike[str] | None) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        path = _normalize_path(candidate)
        if path is None:
            continue
        checked.append(str(path))
        if path.exists():
            return path.resolve()
    checked_text = ", ".join(checked) if checked else "<none>"
    raise FileNotFoundError(f"{description} not found. Checked: {checked_text}")


def resolve_checkpoints_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    return resolve_existing_path(
        "GAGAvatar checkpoints directory",
        explicit,
        os.environ.get("GAGAVATAR_CHECKPOINTS_DIR"),
        CHECKPOINTS_ROOT,
    )


def resolve_assets_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    return resolve_existing_path(
        "GAGAvatar assets directory",
        explicit,
        os.environ.get("GAGAVATAR_ASSETS_DIR"),
        os.environ.get("GAGAVATAR_CHECKPOINTS_DIR"),
        CHECKPOINTS_ROOT,
        LEGACY_ASSETS_ROOT,
    )


def resolve_track_assets_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    return resolve_existing_path(
        "GAGAvatar tracking assets directory",
        explicit,
        os.environ.get("GAGAVATAR_TRACK_ASSETS_DIR"),
        os.environ.get("GAGAVATAR_CHECKPOINTS_DIR"),
        CHECKPOINTS_ROOT,
        LEGACY_TRACK_ASSETS_ROOT,
    )


def resolve_asset_file(filename: str, assets_dir: str | os.PathLike[str] | None = None) -> Path:
    assets_path = resolve_assets_dir(assets_dir)
    return resolve_existing_path(
        f"GAGAvatar asset {filename}",
        os.environ.get(f"GAGAVATAR_{filename.upper().replace('.', '_')}_PATH"),
        assets_path / filename,
    )


def resolve_track_asset_file(
    relative_path: str,
    track_assets_dir: str | os.PathLike[str] | None = None,
) -> Path:
    track_assets_path = resolve_track_assets_dir(track_assets_dir)
    mapped_relative = TRACK_CHECKPOINT_LAYOUT.get(relative_path, relative_path)
    return resolve_existing_path(
        f"GAGAvatar tracking asset {relative_path}",
        track_assets_path / relative_path,
        track_assets_path / mapped_relative,
    )


def resolve_resume_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return resolve_existing_path("GAGAvatar checkpoint", explicit)
    return resolve_existing_path(
        "GAGAvatar checkpoint",
        os.environ.get("GAGAVATAR_RESUME_PATH"),
        os.environ.get("GAGAVATAR_CHECKPOINTS_DIR") and Path(os.environ["GAGAVATAR_CHECKPOINTS_DIR"]) / "GAGAvatar.pt",
        CHECKPOINTS_ROOT / "GAGAvatar.pt",
        LEGACY_ASSETS_ROOT / "GAGAvatar.pt",
    )


def resolve_tracked_cache_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    path = _normalize_path(explicit)
    if path is None:
        path = _normalize_path(os.environ.get("GAGAVATAR_TRACK_CACHE"))
    if path is None:
        path = PROJECT_ROOT / "render_results" / "tracked" / "tracked.pt"
    return path.resolve()


def resolve_dinov2_repo(explicit: str | os.PathLike[str] | None = None) -> Path:
    return resolve_existing_path(
        "DINOv2 local source directory",
        explicit,
        os.environ.get("GAGAVATAR_DINOV2_DIR"),
        PROJECT_ROOT / "third_party" / "dinov2",
    )


def resolve_dinov2_pretrain_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    return resolve_existing_path(
        "DINOv2 pretrained checkpoint",
        explicit,
        os.environ.get("GAGAVATAR_DINOV2_PRETRAIN_PATH"),
        os.environ.get("GAGAVATAR_CHECKPOINTS_DIR") and Path(os.environ["GAGAVATAR_CHECKPOINTS_DIR"]) / "dinov2_vitb14_pretrain.pth",
        CHECKPOINTS_ROOT / "dinov2_vitb14_pretrain.pth",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth",
    )


def resolve_mask2former_model_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    return resolve_existing_path(
        "Mask2Former local model directory",
        explicit,
        os.environ.get("GAGAVATAR_MASK2FORMER_DIR"),
        os.environ.get("GAGAVATAR_CHECKPOINTS_DIR") and Path(os.environ["GAGAVATAR_CHECKPOINTS_DIR"]) / MASK2FORMER_MODEL_DIRNAME,
        CHECKPOINTS_ROOT / MASK2FORMER_MODEL_DIRNAME,
    )


def resolve_cache_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    path = _normalize_path(explicit)
    if path is None:
        path = _normalize_path(os.environ.get("GAGAVATAR_CACHE_DIR"))
    if path is None:
        path = Path.home() / ".cache" / "gagavatar"
    return path.resolve()


def resolve_mediapipe_segmenter_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    return resolve_existing_path(
        "MediaPipe selfie segmenter model",
        explicit,
        os.environ.get("GAGAVATAR_MEDIAPIPE_SEGMENTER_PATH"),
        os.environ.get("GAGAVATAR_CHECKPOINTS_DIR") and Path(os.environ["GAGAVATAR_CHECKPOINTS_DIR"]) / "mediapipe" / "selfie_multiclass_256x256.tflite",
        CHECKPOINTS_ROOT / "mediapipe" / "selfie_multiclass_256x256.tflite",
        resolve_cache_dir() / "mediapipe" / "selfie_multiclass_256x256.tflite",
    )


def set_runtime_overrides(
    *,
    checkpoints_dir: str | os.PathLike[str] | None = None,
    assets_dir: str | os.PathLike[str] | None = None,
    track_assets_dir: str | os.PathLike[str] | None = None,
    tracked_cache_path: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    dinov2_dir: str | os.PathLike[str] | None = None,
    dinov2_pretrain_path: str | os.PathLike[str] | None = None,
    mask2former_dir: str | os.PathLike[str] | None = None,
    mediapipe_segmenter_path: str | os.PathLike[str] | None = None,
    resume_path: str | os.PathLike[str] | None = None,
) -> None:
    overrides = {
        "GAGAVATAR_CHECKPOINTS_DIR": checkpoints_dir,
        "GAGAVATAR_ASSETS_DIR": assets_dir,
        "GAGAVATAR_TRACK_ASSETS_DIR": track_assets_dir,
        "GAGAVATAR_TRACK_CACHE": tracked_cache_path,
        "GAGAVATAR_CACHE_DIR": cache_dir,
        "GAGAVATAR_DINOV2_DIR": dinov2_dir,
        "GAGAVATAR_DINOV2_PRETRAIN_PATH": dinov2_pretrain_path,
        "GAGAVATAR_MASK2FORMER_DIR": mask2former_dir,
        "GAGAVATAR_MEDIAPIPE_SEGMENTER_PATH": mediapipe_segmenter_path,
        "GAGAVATAR_RESUME_PATH": resume_path,
    }
    for env_name, value in overrides.items():
        path = _normalize_path(value)
        if path is not None:
            os.environ[env_name] = str(path.resolve())
