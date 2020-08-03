#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Modified to process a list of videos

"""Extract features for videos using pre-trained networks"""

import numpy as np
import torch
import os
import time
from tqdm import tqdm
import av
from moviepy.video.io.VideoFileClip import VideoFileClip

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc

from models import build_model
from datasets import VideoSet

logger = logging.get_logger(__name__)


def calculate_time_taken(start_time, end_time):
    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = int((end_time - start_time) % 60)
    return hours, minutes, seconds


@torch.no_grad()
def perform_inference(test_loader, model, cfg):
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

    for inputs in tqdm(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Perform the forward pass.
        preds, feat = model(inputs)
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, feat = du.all_gather([preds, feat])

        feat = feat.cpu().numpy()

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
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    vid_root = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.DATA.PATH_PREFIX)
    videos_list_file = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, "vid_list.csv")

    print("Loading Video List ...")
    with open(videos_list_file) as f:
        videos = sorted([x.strip() for x in f.readlines() if len(x.strip()) > 0])
    print("Done")
    print("----------------------------------------------------------")

    if cfg.DATA.READ_VID_FILE:
        rejected_vids = []

    print("{} videos to be processed...".format(len(videos)))
    print("----------------------------------------------------------")

    start_time = time.time()
    for vid_no, vid in enumerate(videos):
        # Create video testing loaders.
        path_to_vid = os.path.join(vid_root, os.path.split(vid)[0])
        vid_id = os.path.split(vid)[1]

        if cfg.DATA.READ_VID_FILE:
            try:
                _ = VideoFileClip(
                    os.path.join(path_to_vid, vid_id) + cfg.DATA.VID_FILE_EXT,
                    audio=False,
                    fps_source="fps",
                )
            except Exception as e:
                print("{}. {} cannot be read with error {}".format(vid_no, vid, e))
                print("----------------------------------------------------------")
                rejected_vids.append(vid)
                continue

        out_path = os.path.join(cfg.OUTPUT_DIR, os.path.split(vid)[0])
        out_file = vid_id.split(".")[0] + "_{}.npy".format(cfg.DATA.NUM_FRAMES)
        if os.path.exists(os.path.join(out_path, out_file)):
            print("{}. {} already exists".format(vid_no, out_file))
            print("----------------------------------------------------------")
            continue

        print("{}. Processing {}...".format(vid_no, vid))

        dataset = VideoSet(
            cfg, path_to_vid, vid_id, read_vid_file=cfg.DATA.READ_VID_FILE
        )
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
        feat_arr = perform_inference(test_loader, model, cfg)

        os.makedirs(out_path, exist_ok=True)
        np.save(os.path.join(out_path, out_file), feat_arr)
        print("Done.")
        print("----------------------------------------------------------")

    if cfg.DATA.READ_VID_FILE:
        print("Rejected Videos: {}".format(rejected_vids))

    end_time = time.time()
    hours, minutes, seconds = calculate_time_taken(start_time, end_time)
    print(
        "Time taken: {} hour(s), {} minute(s) and {} second(s)".format(
            hours, minutes, seconds
        )
    )
    print("----------------------------------------------------------")
