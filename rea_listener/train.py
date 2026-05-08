import argparse
import os
import shutil
import signal
import sys
import time

import yaml

def handle_sigterm(signum, frame):
    print("Received SIGTERM, exiting cleanly.")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the cleaned REA-Listener paper model.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--task", type=str, choices=["listener"], default="listener")
    parser.add_argument("--dataset", type=str, choices=["vico", "realtalk"], required=True)
    parser.add_argument("--model-arch", type=str, choices=["moe_route"], required=True)
    parser.add_argument("--time-size", type=int, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--emotion-checkpoint", type=str, default=None, help="Optional classifier checkpoint required when emotion loss is enabled.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--change-epoch", type=int, required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--temporal-size", type=int, default=None)
    parser.add_argument("--loss-weights", type=float, nargs=8, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-mask", type=float, default=-1.0)
    parser.add_argument("--balance-moe", action="store_true", default=False)
    parser.add_argument("--emotion-loss-weight", type=float, default=0.0)
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


def build_model(config, model_arch: str, emotion_classifier):
    from rea_listener.models.networks.moe_listener import REAListener

    if model_arch != "moe_route":
        raise NotImplementedError(model_arch)
    return REAListener(config, emotion_classifier, balance_moe=config.balance_moe, train=True)


def maybe_override_config(config, args: argparse.Namespace):
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.temporal_size is not None:
        config.temporal_size = args.temporal_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.lr is not None:
        config.lr = args.lr
    if args.loss_weights is not None:
        keys = sorted(list(config.loss_weights.keys()))
        config.loss_weights = {key: args.loss_weights[index] for index, key in enumerate(keys)}
    config.task = args.task
    config.balance_moe = args.balance_moe
    config.emotion_loss_weight = args.emotion_loss_weight
    config.change_epoch = args.change_epoch
    return config


def get_loader(config, args: argparse.Namespace):
    from rea_listener.data.data_gag import get_data_loader as get_vico_loader
    from rea_listener.data.data_gag_realtalk import get_data_loader as get_realtalk_loader

    if args.dataset == "vico":
        return get_vico_loader(config, args.task, args.time_size, random_mask=args.random_mask)
    return get_realtalk_loader(config, args.task, args.time_size, random_mask=args.random_mask)


def init_distributed(local_rank: int) -> tuple[int, int]:
    import torch

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.distributed.barrier()
    return rank, world_size


def main() -> None:
    args = get_parser().parse_args()

    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    import torch.distributed as dist
    import torch.optim as optim
    from box import Box
    from rea_listener.logger import create_logger
    from rea_listener.utils_parallel import DictAverageMeter, get_config, get_scheduler, prepare_sub_folder, write_log

    config = Box(get_config(args.config))
    config = maybe_override_config(config, args)
    config.LOCAL_RANK = args.local_rank

    _, _ = init_distributed(args.local_rank)
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    output_directory = os.path.join(args.output_path)
    checkpoint_directory = prepare_sub_folder(output_directory)
    logger = create_logger(output_directory, dist.get_rank(), os.path.basename(output_directory.strip("/")))
    if dist.get_rank() == 0:
        shutil.copy(args.config, os.path.join(output_directory, "config.yaml"))

    try:
        loader = get_loader(config, args)
        logger.info("Len loader: %s", len(loader))

        emotion_classifier = None
        if args.emotion_loss_weight > 0.0:
            if not args.emotion_checkpoint:
                raise ValueError("--emotion-checkpoint is required when --emotion-loss-weight > 0.")
            emotion_classifier = load_emotion_classifier(args.emotion_checkpoint)

        model = build_model(config, args.model_arch, emotion_classifier)
        model_config_path = os.path.join(output_directory, "model_config.yaml")
        if dist.get_rank() == 0:
            with open(model_config_path, "w") as file:
                yaml.dump(config.to_dict(), file, default_flow_style=False, allow_unicode=True)

        if args.resume is not None:
            state_dict = torch.load(args.resume, map_location="cpu", weights_only=True)
            model.load_state_dict({k.replace("module.", "", 1): v for k, v in state_dict.items()})

        model = model.cuda()
        optimizer = optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

        logger.info(str(model))
        logger.info("number of params: %s", sum(param.numel() for param in model.parameters() if param.requires_grad))

        lr_scheduler = get_scheduler(optimizer, config)
        meter_keys = ["iter_time", "TOTAL_LOSS", "loss_angle", "loss_angle_spiky", "loss_exp", "loss_exp_spiky", "loss_trans", "loss_trans_spiky", "loss_emotion", "acc_emotion"]
        meter = DictAverageMeter(*meter_keys)
        iterations = 0
        max_iter = config.max_epochs * len(loader)
        start_data = time.time()

        for epoch in range(config.max_epochs):
            meter.reset()
            if lr_scheduler is not None:
                lr_scheduler.step()
            loader.sampler.set_epoch(epoch)
            model.train()

            for batch in loader:
                if args.random_mask > 0:
                    audio, driven_signal, init_signal, target_signal, lengths, _, _, emotion, kps, modal_type, lengths_kps = batch
                else:
                    audio, driven_signal, init_signal, target_signal, lengths, _, _, emotion, kps = batch

                audio = audio.cuda().float()
                driven_signal = driven_signal.cuda().float()
                init_signal = init_signal.cuda().float()
                target_signal = target_signal.cuda().float()
                lengths = lengths.cuda().long()
                emotion = emotion.cuda()
                kps = kps.cuda().float()
                elapse_data = time.time() - start_data

                if args.random_mask > 0:
                    modal_type = modal_type.cuda().long()
                    lengths_kps = lengths_kps.cuda().long()
                else:
                    modal_type = torch.full((audio.shape[0],), 2, dtype=torch.long, device=audio.device)
                    lengths_kps = lengths
                loss, loss_dict, _ = model(audio, driven_signal, init_signal, lengths, epoch + 1, args.change_epoch, target_signal, emotion, kps, modal_type, lengths_kps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                meter.update({"iter_time": {"val": elapse_data}, **loss_dict})
                if (iterations + 1) % config["log_iter"] == 0:
                    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                    loss_log = "".join(
                        f"{loss_key} {meter[loss_key].val:.4f}*{loss_item['weight']} ({meter[loss_key].avg:.4f})\t"
                        for loss_key, loss_item in loss_dict.items()
                    )
                    log = f"Train: [{iterations + 1}/{max_iter}]\ttime {meter.iter_time.val:.2f} ({meter.iter_time.avg:.2f})\ttime_data {elapse_data:.2f}\t{loss_log}mem {memory:.0f}MB"
                    logger.info(log)
                    write_log(log, output_directory)
                iterations += 1

            if (epoch + 1) % 50 == 0 and dist.get_rank() == 0:
                model_state_dict_name = os.path.join(checkpoint_directory, f"Epoch_{epoch + 1:03d}.bin")
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_state_dict_name)
                logger.info("save epoch %s model to %s", epoch + 1, model_state_dict_name)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
