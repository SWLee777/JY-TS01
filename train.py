import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import ValPre, get_train_loader
from engine.engine import Engine
from engine.logger import get_logger
from eval import SegEvaluator
from models.builder import EncoderDecoder as SegModel
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.pyt_utils import all_reduce_tensor


parser = argparse.ArgumentParser()
logger = get_logger()
os.environ["MASTER_PORT"] = "16005"


def load_config(dataset_name):
    if dataset_name == "mfnet":
        from configs.config_MFNet import config
    elif dataset_name == "mfnet_small":
        from configs.config_MFNet_small import config
    elif dataset_name == "pst":
        from configs.config_pst900 import config
    elif dataset_name == "pst_small":
        from configs.config_pst900_small import config
    elif dataset_name == "nyu":
        from configs.config_nyu import config
    elif dataset_name == "nyu_small":
        from configs.config_nyu_small import config
    elif dataset_name == "nyu_d0":
        from configs.config_nyu_d0 import config
    elif dataset_name == "nyu_d1":
        from configs.config_nyu_d1 import config
    elif dataset_name == "nyu_d2":
        from configs.config_nyu_d2 import config
    elif dataset_name == "nyu_d3":
        from configs.config_nyu_d3 import config
    elif dataset_name == "nyu_d4":
        from configs.config_nyu_d4 import config
    elif dataset_name == "nyu_d6_1_4":
        from configs.config_nyu_d6_1_4 import config
    elif dataset_name == "nyu_d6_1_8":
        from configs.config_nyu_d6_1_8 import config
    elif dataset_name == "nyu_hp_aux0":
        from configs.config_nyu_hp_aux0 import config
    elif dataset_name == "nyu_hp_aux01":
        from configs.config_nyu_hp_aux01 import config
    elif dataset_name == "nyu_hp_aux05":
        from configs.config_nyu_hp_aux05 import config
    elif dataset_name == "nyu_hp_aux005":
        from configs.config_nyu_hp_aux005 import config
    elif dataset_name == "nyu_hp_aux03":
        from configs.config_nyu_hp_aux03 import config
    elif dataset_name == "nyu_aux005_full":
        from configs.config_nyu_aux005_full import config
    elif dataset_name == "nyu_s500_high":
        from configs.config_nyu_s500_high import config
    elif dataset_name == "nyu_s500_low":
        from configs.config_nyu_s500_low import config
    elif dataset_name == "nyu_s500_simple":
        from configs.config_nyu_s500_simple import config
    elif dataset_name == "nyu_s500_mlp":
        from configs.config_nyu_s500_mlp import config
    elif dataset_name == "nyu_s500_linear":
        from configs.config_nyu_s500_linear import config
    elif dataset_name == "sun":
        from configs.config_sunrgbd import config
    elif dataset_name == "sun_small":
        from configs.config_sunrgbd_small import config
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return config


def set_random_seed(seed, config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = bool(getattr(config, "cudnn_benchmark", True))
    cudnn.deterministic = bool(
        getattr(config, "cudnn_deterministic", False)
    )


def build_validation_dataset(config):
    settings = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
    }
    return RGBXDataset(settings, "val", ValPre())


def decoder_diagnostic_text(network):
    core = network.module if hasattr(network, "module") else network
    head = getattr(core, "decode_head", None)
    if head is None or not hasattr(head, "diagnostic_summary"):
        return ""
    summary = head.diagnostic_summary()
    pieces = []
    for scale in ("1_8", "1_4"):
        keys = (
            f"alpha_{scale}",
            f"reject_{scale}",
            f"rgb_{scale}",
            f"depth_{scale}",
        )
        if all(key in summary for key in keys):
            pieces.append(
                "{} a={:.3f} reject/rgb/depth={:.2f}/{:.2f}/{:.2f}".format(
                    scale,
                    summary[keys[0]],
                    summary[keys[1]],
                    summary[keys[2]],
                    summary[keys[3]],
                )
            )
    return " | " + " | ".join(pieces) if pieces else ""


def should_validate(epoch, config):
    validation_start_epoch = getattr(
        config, "validation_start_epoch", config.checkpoint_start_epoch
    )
    validation_step = getattr(
        config, "validation_step", config.checkpoint_step
    )
    return (
        epoch >= validation_start_epoch
        and (epoch - validation_start_epoch) % validation_step == 0
    )


def should_save(epoch, config):
    return (
        epoch == config.nepochs
        or (
            epoch >= config.checkpoint_start_epoch
            and (epoch - config.checkpoint_start_epoch)
            % config.checkpoint_step
            == 0
        )
    )


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = load_config(args.dataset_name)
    is_main_process = not engine.distributed or engine.local_rank == 0

    seed = int(config.seed) + (engine.local_rank if engine.distributed else 0)
    set_random_seed(seed, config)

    train_loader, train_sampler = get_train_loader(
        engine, RGBXDataset, config
    )
    if is_main_process:
        timestamp = time.strftime("%b%d_%H-%M", time.localtime())
        tb_dir = os.path.join(config.tb_dir, timestamp)
        tb_link = os.path.join(config.tb_dir, "tb")
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, tb_link)

    criterion = nn.CrossEntropyLoss(
        reduction="mean", ignore_index=config.background
    )
    norm_layer = nn.SyncBatchNorm if engine.distributed else nn.BatchNorm2d
    model = SegModel(cfg=config, criterion=criterion, norm_layer=norm_layer)

    parameter_groups = group_weight([], model, norm_layer, config.lr)
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "SGDM":
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError(config.optimizer)

    total_iterations = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        config.lr,
        config.lr_power,
        total_iterations,
        config.niters_per_epoch * config.warm_up_epoch,
    )

    if engine.distributed:
        logger.info("Distributed training on rank %d", engine.local_rank)
        device = torch.device("cuda", engine.local_rank)
        model.to(device)
        model = DistributedDataParallel(
            model,
            device_ids=[engine.local_rank],
            output_device=engine.local_rank,
            find_unused_parameters=bool(
                getattr(config, "find_unused_parameters", False)
            ),
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(
        dataloader=train_loader, model=model, optimizer=optimizer
    )
    if engine.continue_state_object:
        engine.restore_checkpoint()

    validation_dataset = build_validation_dataset(config)
    train_stop_epoch = int(
        getattr(config, "train_stop_epoch", config.nepochs)
    )
    if not 1 <= train_stop_epoch <= config.nepochs:
        raise ValueError(
            "train_stop_epoch must be within [1, config.nepochs], "
            f"got {train_stop_epoch} with nepochs={config.nepochs}"
        )
    best_mean_iou = float("-inf")
    best_epoch = None
    last_saved_epoch = None
    log_interval = int(getattr(config, "log_interval", 20))
    model.train()
    logger.info("Begin training")
    logger.info(
        "LR schedule epochs: %d; training stop epoch: %d",
        config.nepochs,
        train_stop_epoch,
    )

    for epoch in range(engine.state.epoch, train_stop_epoch + 1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)

        progress = tqdm(
            range(config.niters_per_epoch),
            file=sys.stdout,
            bar_format="{desc}[{elapsed}<{remaining},{rate_fmt}]",
            disable=not is_main_process,
        )
        data_iterator = iter(train_loader)
        sum_loss = 0.0

        for index in progress:
            engine.update_iteration(epoch, index)
            global_index = (
                (epoch - 1) * config.niters_per_epoch + index
            )
            learning_rate = lr_policy.get_lr(global_index)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            minibatch = next(data_iterator)
            images = minibatch["data"].to(device, non_blocking=True)
            labels = minibatch["label"].to(device, non_blocking=True)
            modal_x = minibatch["modal_x"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = model(images, modal_x, labels)
            loss.backward()
            optimizer.step()

            display_loss = (
                all_reduce_tensor(
                    loss.detach(), world_size=engine.world_size
                )
                if engine.distributed
                else loss.detach()
            )
            if is_main_process:
                loss_value = display_loss.item()
                sum_loss += loss_value
                diagnostic = ""
                if (index + 1) % log_interval == 0:
                    diagnostic = decoder_diagnostic_text(model)
                progress.set_description(
                    "Epoch {}/{} Iter {}/{} lr={:.4e} loss={:.4f} "
                    "mean={:.4f}{}".format(
                        epoch,
                        config.nepochs,
                        index + 1,
                        config.niters_per_epoch,
                        learning_rate,
                        loss_value,
                        sum_loss / (index + 1),
                        diagnostic,
                    ),
                    refresh=False,
                )

        if is_main_process:
            tb.add_scalar(
                "train_loss", sum_loss / config.niters_per_epoch, epoch
            )

        if should_save(epoch, config) and is_main_process:
            engine.save_and_link_checkpoint(
                config.checkpoint_dir,
                config.log_dir,
                config.log_dir_link,
            )
            if (
                not getattr(config, "keep_all_checkpoints", False)
                and last_saved_epoch is not None
                and last_saved_epoch != best_epoch
            ):
                previous_path = os.path.join(
                    config.checkpoint_dir,
                    f"epoch-{last_saved_epoch}.pth",
                )
                if os.path.exists(previous_path):
                    os.remove(previous_path)
            last_saved_epoch = epoch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if should_validate(epoch, config) and is_main_process:
            evaluation_model = model.module if hasattr(model, "module") else model
            evaluation_model.eval()
            with torch.no_grad():
                evaluator = SegEvaluator(
                    dataset=validation_dataset,
                    class_num=config.num_classes,
                    norm_mean=config.norm_mean,
                    norm_std=config.norm_std,
                    network=evaluation_model,
                    multi_scales=config.eval_scale_array,
                    is_flip=config.eval_flip,
                    devices=[device],
                    verbose=False,
                    config=config,
                )
                _, mean_iou = evaluator.run(
                    config.checkpoint_dir,
                    str(epoch),
                    config.val_log_file,
                    config.link_val_log_file,
                )
            logger.info("Epoch %d mean IoU: %.6f", epoch, mean_iou)

            if mean_iou > best_mean_iou:
                if (
                    not getattr(config, "keep_all_checkpoints", False)
                    and best_epoch is not None
                ):
                    old_path = os.path.join(
                        config.checkpoint_dir, f"epoch-{best_epoch}.pth"
                    )
                    if os.path.exists(old_path) and best_epoch != last_saved_epoch:
                        os.remove(old_path)
                best_epoch = epoch
                best_mean_iou = mean_iou
            evaluation_model.train()

        if engine.distributed:
            dist.barrier()

    if is_main_process:
        tb.close()
