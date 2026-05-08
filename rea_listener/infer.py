import argparse
import os
import os.path as osp
import pickle
import shutil
import time
from pathlib import Path

EMOTION_MAP = {
    "Anger": 0,
    "Disgust": 0,
    "Fear": 5,
    "Happiness": 3,
    "Neutral": 4,
    "Sadness": 5,
    "Surprise": 3,
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cleaned REA-Listener inference.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["listener"], default="listener")
    parser.add_argument("--dataset", type=str, choices=["vico", "realtalk"], required=True)
    parser.add_argument("--model-arch", type=str, choices=["moe_route"], required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--emotion", type=str, default="GT", help="GT, none, a fixed category, or a predicted-emotion directory.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--temporal-size", type=int, default=None)
    parser.add_argument("--emotion-loss-weight", type=float, default=0.0)
    parser.add_argument("--emotion-checkpoint", type=str, default=None)
    parser.add_argument("--encode-type", type=str, default="kps")
    parser.add_argument("--balance-moe", action="store_true", default=False)
    parser.add_argument("--change-epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_emotion_classifier(checkpoint_path: str):
    import torch
    from rea_listener.models.networks.coef_emotion_classifier import mlp_emotion_classifier

    model = mlp_emotion_classifier()
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in state_dict.items()})
    return model


def build_model(config, args: argparse.Namespace, emotion_classifier):
    from rea_listener.models.networks.moe_listener import REAListener

    if args.model_arch != "moe_route":
        raise NotImplementedError(args.model_arch)
    return REAListener(config, emotion_classifier, balance_moe=args.balance_moe, train=False)


def resolve_loader(config, args: argparse.Namespace):
    from rea_listener.data.data_gag_eval import get_data_loader as get_vico_loader
    from rea_listener.data.data_gag_eval_realtalk import get_data_loader as get_realtalk_loader

    if args.emotion not in {"GT", "none"} and args.emotion not in EMOTION_MAP and Path(args.emotion).exists():
        emotion_source = args.emotion
    else:
        emotion_source = None

    if args.dataset == "vico":
        return get_vico_loader(config, args.task, pred_emotion_path=emotion_source)
    return get_realtalk_loader(config, args.task, pred_emotion_path=emotion_source)


def copy_image_lmdb(dataset: str, data_root: str, row: dict, target_folder: str) -> None:
    from rea_listener.paths import build_realtalk_layout, build_vico_layout

    if dataset == "vico":
        source = build_vico_layout(data_root).listener_original_dir / f"{row['listener']}.mp4" / "img_lmdb"
    else:
        name = "_".join(map(str, row["id"]))
        source = build_realtalk_layout(data_root).listener_track_dir / f"{name}.mp4" / "img_lmdb"
    if source.exists():
        shutil.copytree(source, target_folder, dirs_exist_ok=True)


def main() -> None:
    args = get_parser().parse_args()

    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    from box import Box
    from rea_listener.coeff_io import run_smoothing, save_realtalk_coefficients, save_vico_coefficients
    from rea_listener.logger import create_logger
    from rea_listener.utils_parallel import get_config, prepare_sub_folder

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    config = Box(get_config(args.config))
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.temporal_size is not None:
        config.temporal_size = args.temporal_size
    config.task = args.task
    config.emotion_loss_weight = args.emotion_loss_weight
    config.balance_moe = args.balance_moe
    config.change_epoch = args.change_epoch

    output_directory = os.path.join(args.output_path)
    prepare_sub_folder(output_directory)
    logger = create_logger(output_directory, 0, os.path.basename(output_directory.strip("/")))
    shutil.copy(args.config, os.path.join(output_directory, "config.yaml"))

    emotion_classifier = None
    if args.emotion_loss_weight > 0.0:
        if not args.emotion_checkpoint:
            raise ValueError("--emotion-checkpoint is required when --emotion-loss-weight > 0.")
        emotion_classifier = load_emotion_classifier(args.emotion_checkpoint)

    loader = resolve_loader(config, args)
    logger.info("Len loader: %s", len(loader))

    model = build_model(config, args, emotion_classifier)
    try:
        state_dict = torch.load(args.resume, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(args.resume, map_location="cpu")
    incompat = model.load_state_dict(
        {k.replace("module.", "", 1): v for k, v in state_dict.items()},
        strict=False,
    )
    model = model.cuda()
    model.eval()
    logger.info(str(model))
    logger.info("load_state_dict missing_keys: %s", list(incompat.missing_keys))
    logger.info("load_state_dict unexpected_keys: %s", list(incompat.unexpected_keys))

    mat_output_dir = osp.join(args.output_path, "recon_coeffs", "test")
    os.makedirs(mat_output_dir, exist_ok=True)
    os.makedirs(osp.join(args.output_path, "vox_lmdb"), exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            audio, driven_signal, init_signal, target_signal, lengths, rows, _, emotion, kps = batch
            if args.emotion in EMOTION_MAP:
                emotion = [torch.ones_like(item).long() * EMOTION_MAP[args.emotion] for item in emotion]

            audio = audio.cuda().float()
            driven_signal = driven_signal.cuda().float()
            init_signal = init_signal.cuda().float()
            target_signal = target_signal.cuda().float()
            lengths = lengths.cuda().long()
            kps = kps.cuda().float()
            emotion = [item.cuda() for item in emotion]

            start_time = time.time()
            prediction = model(audio, driven_signal, init_signal, lengths, None, args.change_epoch, target=None, emotion=emotion, speaker_kps=kps, encode_type=args.encode_type)
            logger.info("forward time %.4fs", time.time() - start_time)

            row = rows[0]
            if args.dataset == "realtalk":
                predicted_coef = save_realtalk_coefficients(prediction, row, config.root)
                predicted_coef = run_smoothing(predicted_coef)
                sample_name = "_".join(map(str, row["id"]))
            else:
                predicted_coef = save_vico_coefficients(prediction, row, config.root)
                predicted_coef = run_smoothing(predicted_coef)
                sample_name = row["listener"]

            save_path = osp.join(mat_output_dir, sample_name)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "smoothed.pkl"), "wb+") as file:
                pickle.dump(predicted_coef, file)
            copy_image_lmdb(args.dataset, config.root, row, osp.join(save_path, "img_lmdb"))


if __name__ == "__main__":
    main()
