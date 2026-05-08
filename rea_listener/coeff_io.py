import copy
import pickle
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from scipy.spatial.transform import Rotation as R

from rea_listener.paths import build_realtalk_layout, build_vico_layout


def rotation_matrix_to_angles(rotation_matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(rotation_matrix).as_euler("zyx", degrees=False)


def angles_to_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    return R.from_euler("zyx", angles, degrees=False).as_matrix()


def _smooth_params(data: np.ndarray, alpha: float) -> list[np.ndarray]:
    smoothed = [data[0]]
    for index in range(1, len(data)):
        smoothed.append(alpha * data[index] + (1 - alpha) * smoothed[index - 1])
    return smoothed


def run_smoothing(lightning_result: dict, smooth_alpha: list[float] | tuple[float, float, float] = (0.6, 0.6, 0.7)) -> dict:
    smoothed_results = {}
    rotates, translates, eyecode = [], [], []
    frames = sorted(lightning_result.keys(), key=lambda key: int(key.split("_")[-1]))
    for frame_name in frames:
        smoothed_results[frame_name] = copy.deepcopy(lightning_result[frame_name])
        transform_matrix = smoothed_results[frame_name]["transform_matrix"]
        rotates.append(matrix_to_rotation_6d(torch.tensor(transform_matrix[:3, :3])).numpy())
        translates.append(transform_matrix[:3, 3])
        eyecode.append(smoothed_results[frame_name]["eyecode"])

    rotates = _smooth_params(np.stack(rotates), alpha=smooth_alpha[0])
    translates = _smooth_params(np.stack(translates), alpha=smooth_alpha[1])
    eyecode = _smooth_params(np.stack(eyecode), alpha=smooth_alpha[2])

    for frame_index, frame_name in enumerate(frames):
        rotation = rotation_6d_to_matrix(torch.tensor(rotates[frame_index])).numpy()
        affine_matrix = np.concatenate([rotation, translates[frame_index][:, None]], axis=-1)
        smoothed_results[frame_name]["transform_matrix"] = affine_matrix
        smoothed_results[frame_name]["eyecode"] = eyecode[frame_index]
    return smoothed_results


def save_vico_coefficients(prediction: torch.Tensor, row: dict, data_root: str | Path) -> dict:
    layout = build_vico_layout(data_root)
    with open(layout.coef_mean_path, "rb") as file:
        norm_mean = pickle.load(file)
    with open(layout.coef_std_path, "rb") as file:
        norm_std = pickle.load(file)

    listener_coef_gt = layout.listener_original_dir / f"{row['listener']}.mp4" / "smoothed.pkl"
    with open(listener_coef_gt, "rb") as file:
        ground_truth = pickle.load(file)
        coefficient_prefix = "_".join(list(ground_truth.keys())[0].split("_")[:-1])

    prediction_np = prediction.detach().squeeze(0).cpu().numpy()
    for frame_idx in range(1, row["num_frames"]):
        pred_115 = prediction_np[frame_idx : frame_idx + 1, :]
        exp_100 = pred_115[0, 0:100] * norm_std["expcode"] + norm_mean["expcode"]
        pose_6 = np.concatenate([np.zeros((3)), pred_115[0, 100:103]], axis=0) * norm_std["posecode"] + norm_mean["posecode"]
        eye_6 = pred_115[0, 103:109] * norm_std["eyecode"] + norm_mean["eyecode"]
        rotation_3 = pred_115[0, 109:112] * norm_std["transform_matrix_angles"] + norm_mean["transform_matrix_angles"]
        translation_3 = pred_115[0, 112:115] * norm_std["transform_matrix_trans"] + norm_mean["transform_matrix_trans"]

        transform_matrix = np.concatenate([angles_to_rotation_matrix(rotation_3), translation_3[:, None]], axis=-1)
        frame_key = f"{coefficient_prefix}_{frame_idx}"
        ground_truth[frame_key]["expcode"] = copy.deepcopy(exp_100.astype(np.float32))
        ground_truth[frame_key]["posecode"] = copy.deepcopy(pose_6.astype(np.float32))
        ground_truth[frame_key]["eyecode"] = copy.deepcopy(eye_6.astype(np.float32))
        ground_truth[frame_key]["transform_matrix"] = copy.deepcopy(transform_matrix.astype(np.float32))

    return ground_truth


def save_realtalk_coefficients(prediction: torch.Tensor, row: dict, data_root: str | Path) -> dict:
    layout = build_realtalk_layout(data_root)
    with open(layout.coef_norm_path, "rb") as file:
        norm_mean_std = pickle.load(file)
    norm_mean = norm_mean_std["mean"]
    norm_std = norm_mean_std["std"]

    name = "_".join(map(str, row["id"]))
    listener_coef_gt = layout.listener_track_dir / f"{name}.mp4" / "smoothed.pkl"
    with open(listener_coef_gt, "rb") as file:
        ground_truth = pickle.load(file)
        coefficient_prefix = "_".join(list(ground_truth.keys())[0].split("_")[:-1])

    prediction_np = prediction.detach().squeeze(0).cpu().numpy()
    for frame_idx in range(1, 384):
        pred_115 = prediction_np[frame_idx : frame_idx + 1, :]
        exp_100 = pred_115[0, 0:100] * norm_std["expcode"] + norm_mean["expcode"]
        pose_6 = np.concatenate([np.zeros((3)), pred_115[0, 100:103]], axis=0) * norm_std["posecode"] + norm_mean["posecode"]
        eye_6 = pred_115[0, 103:109] * norm_std["eyecode"] + norm_mean["eyecode"]
        rotation_3 = pred_115[0, 109:112] * norm_std["transform_matrix_angles"] + norm_mean["transform_matrix_angles"]
        translation_3 = pred_115[0, 112:115] * norm_std["transform_matrix_trans"] + norm_mean["transform_matrix_trans"]

        transform_matrix = np.concatenate([angles_to_rotation_matrix(rotation_3), translation_3[:, None]], axis=-1)
        frame_key = f"{coefficient_prefix}_{frame_idx}"
        ground_truth[frame_key]["expcode"] = copy.deepcopy(exp_100.astype(np.float32))
        ground_truth[frame_key]["posecode"] = copy.deepcopy(pose_6.astype(np.float32))
        ground_truth[frame_key]["eyecode"] = copy.deepcopy(eye_6.astype(np.float32))
        ground_truth[frame_key]["transform_matrix"] = copy.deepcopy(transform_matrix.astype(np.float32))

    return ground_truth

