#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Modified to process a list of videos

"""Extract features for videos using pre-trained networks"""

import numpy as np
import torch
import os
import time

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc

from models import model_builder
from datasets import loader
from datasets.videoset import VideoSet

logger = logging.get_logger(__name__)


def calculate_time_taken(start_time, end_time):
    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = int((end_time - start_time) % 60)
    return hours, minutes, seconds


def multi_view_test(test_loader, model, cfg):
    """
    Perform mutli-view testing that samples a segment of frames from a video
    and extract features from a pre-trained model.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    feat_arr = None

    for inputs in test_loader:
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Perform the forward pass.
        preds, feat = model(inputs)
        feat = feat.cpu().numpy()
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])

        if feat_arr is None:
            feat_arr = feat
        else:
            feat_arr = np.concatenate((feat_arr, feat), axis=0)

    return feat_arr


def test(cfg):
    """
    Perform multi-view testing/feature extraction on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    vid_root = cfg.DATA.PATH_TO_DATA_DIR
    videos_list_file = os.path.join(vid_root, "vid_list.csv")

    print("Loading Video List ...")
    with open(videos_list_file) as f:
        videos = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    print("Done")
    print("----------------------------------------------------------")

    print("{} videos to be processed...".format(len(videos)))
    print("----------------------------------------------------------")

    start_time = time.time()
    for vid in videos:
        # Create video testing loaders.
        path_to_vid = os.path.join(vid_root, os.path.split(vid)[0])
        vid_id = os.path.split(vid)[1]
        print("Processing {}...".format(vid))

        dataset = VideoSet(cfg, path_to_vid, vid_id)
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            sampler=None,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
        )

        # Perform multi-view test on the entire dataset.
        feat_arr = multi_view_test(test_loader, model, cfg)
        out_path = os.path.join(cfg.OUTPUT_DIR, os.path.split(vid)[0])
        out_file = vid_id.split(".")[0] + "_{}.npy".format(cfg.DATA.NUM_FRAMES)
        os.makedirs(out_path, exist_ok=True)
        np.save(os.path.join(out_path, out_file), feat_arr)
        print("Done.")
        print("----------------------------------------------------------")
    end_time = time.time()
    hours, minutes, seconds = calculate_time_taken(start_time, end_time)
    print(
        "Time taken: {} hour(s), {} minute(s) and {} second(s)".format(
            hours, minutes, seconds
        )
    )
    print("----------------------------------------------------------")
