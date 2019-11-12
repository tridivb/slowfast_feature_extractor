#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified to load and process frames of a single video

import os
import random
from io import BytesIO
import torch
import torch.utils.data
import numpy as np
import av

import slowfast.datasets.decoder as decoder
import slowfast.datasets.transform as transform
import slowfast.datasets.video_container as container
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


class VideoSet(torch.utils.data.Dataset):
    """
    Construct the untrimmed video loader, then sample
    segments from the videos. The videos are segmented by centering
    each frame as per the output size i.e. cfg.DATA.NUM_FRAMES.
    """

    def __init__(self, cfg, vid_path, vid_id):
        """
        Construct the video loader for a given video.
        Args:
            cfg (CfgNode): configs.
            vid_path (string): path to the video
            vid_id (string): video name
        """
        self.cfg = cfg

        self.vid_path = vid_path
        self.vid_id = vid_id
        self.in_fps = cfg.DATA.IN_FPS
        self.out_fps = cfg.DATA.OUT_FPS
        self.step_size = int(self.in_fps / self.out_fps)
        self.out_size = cfg.DATA.NUM_FRAMES
        self.frames = self._get_frames()

    def _get_frames(self):
        """
        Extract frames from the video container
        """
        path_to_vid = os.path.join(self.vid_path, self.vid_id)
        assert os.path.exists(path_to_vid), "{} file not found".format(path_to_vid)

        try:
            # Load video
            self.video_container = container.get_video_container(path_to_vid)

        except Exception as e:
            logger.info(
                "Failed to load video from {} with error {}".format(path_to_vid, e)
            )

        self.orig_width = self.video_container.streams.video[0].width
        self.orig_height = self.video_container.streams.video[0].height

        frames = np.zeros(
            (
                self.video_container.streams.video[0].frames,
                self.orig_height,
                self.orig_width,
                3,
            )
        ).astype(np.float32)

        for ind, in_frame in enumerate(self.video_container.decode(video=0)):
            if "rgb" not in in_frame.format.name:
                in_frame = in_frame.to_rgb()
            frames[ind, :, :, :] = in_frame.to_ndarray()

        # convert to tensor
        frames = torch.from_numpy(frames).float()

        # Normalize the values
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)

        return frames

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """

        frame_seg = torch.zeros(
            (3, self.out_size, self.orig_height, self.orig_width)
        ).float()

        start = int(index - self.step_size * self.out_size / 2)
        end = int(index + self.step_size * self.out_size / 2)
        for ind in range(start, end, self.step_size):
            if ind < 0 or ind >= self.frames.shape[0]:
                continue
            else:
                frame_seg[:, ind, :, :] = self.frames[:, ind, :, :]

        min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
        assert len({min_scale, max_scale, crop_size}) == 1

        # Perform data augmentation.
        frame_seg = self.spatial_sampling(
            frame_seg,
            spatial_idx=1,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )

        # create the pathways
        frame_list = self.pack_pathway_output(frame_seg)

        return frame_list

    def __len__(self):
        """
        Returns:
            (int): the number of frames in the video.
        """
        return self.video_container.streams.video[0].frames

    def pack_pathway_output(self, frames):
        """
        Prepare output as a list of tensors. Each tensor corresponding to a
        unique pathway.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `channel` x `num frames` x `height` x `width`.
        Returns:
            frame_list (list): list of tensors with the dimension of
                `channel` x `num frames` x `height` x `width`.
        """
        if self.cfg.MODEL.ARCH in self.cfg.MODEL.SINGLE_PATHWAY_ARCH:
            frame_list = [frames]
        elif self.cfg.MODEL.ARCH in self.cfg.MODEL.MULTI_PATHWAY_ARCH:
            fast_pathway = frames
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // self.cfg.SLOWFAST.ALPHA,
                ).long(),
            )
            frame_list = [slow_pathway, fast_pathway]
        else:
            raise NotImplementedError(
                "Model arch {} is not in {}".format(
                    self.cfg.MODEL.ARCH,
                    self.cfg.MODEL.SINGLE_PATHWAY_ARCH
                    + self.cfg.MODEL.MULTI_PATHWAY_ARCH,
                )
            )
        return frame_list

    def spatial_sampling(
        self, frames, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames = transform.random_crop(frames, crop_size)
            frames = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
