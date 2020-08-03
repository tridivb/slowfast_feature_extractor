#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified to load and process frames of a single video

import os
import random
from io import BytesIO
import torch
import torch.utils.data
import numpy as np
from parse import parse
import av
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

import slowfast.datasets.decoder as decoder
import slowfast.datasets.transform as transform
import slowfast.datasets.video_container as container
from slowfast.datasets.utils import pack_pathway_output
from slowfast.datasets import DATASET_REGISTRY
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class VideoSet(torch.utils.data.Dataset):
    """
    Construct the untrimmed video loader, then sample
    segments from the videos. The videos are segmented by centering
    each frame as per the output size i.e. cfg.DATA.NUM_FRAMES.
    """

    def __init__(self, cfg, vid_path, vid_id, read_vid_file=False):
        """
        Construct the video loader for a given video.
        Args:
            cfg (CfgNode): configs.
            vid_path (string): path to the video
            vid_id (string): video name
            read_vid_file (bool): flag to turn on/off reading video files.
        """
        self.cfg = cfg

        self.vid_path = vid_path
        self.vid_id = vid_id
        self.read_vid_file = read_vid_file

        self.in_fps = cfg.DATA.IN_FPS
        self.out_fps = cfg.DATA.OUT_FPS
        self.step_size = int(self.in_fps / self.out_fps)

        self.out_size = cfg.DATA.NUM_FRAMES

        if isinstance(cfg.DATA.SAMPLE_SIZE, list):
            self.sample_width, self.sample_height = cfg.DATA.SAMPLE_SIZE
        elif isinstance(cfg.DATA.SAMPLE_SIZE, int):
            self.sample_width = self.sample_height = cfg.DATA.SAMPLE_SIZE
        else:
            raise Exception(
                "Error: Frame sampling size type must be a list [Height, Width] or int"
            )

        self.frames = self._get_frames()

    def _get_frames(self):
        """
        Extract frames from the video container
        Returns:
            frames(tensor or list): A tensor of extracted frames from a video or a list of images to be processed
        """
        if self.read_vid_file:
            path_to_vid = (
                os.path.join(self.vid_path, self.vid_id) + self.cfg.DATA.VID_FILE_EXT
            )
            assert os.path.exists(path_to_vid), "{} file not found".format(path_to_vid)

            try:
                # Load video
                # self.video_container = container.get_video_container(path_to_vid)
                video_clip = VideoFileClip(path_to_vid, audio=False, fps_source="fps")

            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(path_to_vid, e)
                )

            frames = None

            for in_frame in video_clip.iter_frames(fps=self.cfg.DATA.IN_FPS):
                in_frame = cv2.resize(
                    in_frame,
                    (self.sample_width, self.sample_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                if frames is None:
                    frames = in_frame[None, ...]
                else:
                    frames = np.concatenate((frames, in_frame[None, ...]), axis=0)

            frames = self._pre_process_frame(frames)

            return frames

        else:
            path_to_frames = os.path.join(self.vid_path, self.vid_id)
            frames = sorted(
                filter(
                    lambda x: x.endswith(self.cfg.DATA.IMG_FILE_EXT),
                    os.listdir(path_to_frames),
                ),
                key=lambda x: parse(self.cfg.DATA.IMG_FILE_FORMAT, x)[0],
            )
            return frames

    def _pre_process_frame(self, arr):
        """
        Pre process an array
        Args:
            arr (ndarray): an array of frames or a ndarray of an image
                of shape T x H x W x C or W x H x C respectively
        Returns:
            arr (tensor): a normalized torch tensor of shape C x T x H x W 
                or C x W x H respectively
        """
        arr = torch.from_numpy(arr).float()

        # Normalize the values
        arr = arr / 255.0
        arr = arr - torch.tensor(self.cfg.DATA.MEAN)
        arr = arr / torch.tensor(self.cfg.DATA.STD)

        # T H W C -> C T H W.
        if len(arr.shape) == 4:
            arr = arr.permute(3, 0, 1, 2)
        elif len(arr.shape) == 3:
            arr = arr.permute(2, 0, 1)

        return arr

    def _read_img_file(self, path, file):
        """
        Read an image file
        Args:
            path (str): path to the image file
            file (str): name of image file
        Returns:
            img (tensor): a normalized torch tensor
        """
        img = cv2.imread(os.path.join(path, file))

        if len(img.shape) != 3:
            raise Exception(
                "Incorrect image format. Image needs to be read in RGB format."
            )

        img = cv2.resize(
            img,
            (self.sample_width, self.sample_height),
            interpolation=cv2.INTER_LINEAR,
        )
        img = self._pre_process_frame(img)
        return img

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
            (
                3,
                self.out_size,
                self.cfg.DATA.TEST_CROP_SIZE,
                self.cfg.DATA.TEST_CROP_SIZE,
            )
        ).float()

        start = int(index - self.step_size * self.out_size / 2)
        end = int(index + self.step_size * self.out_size / 2)
        max_ind = self.__len__() - 1

        for out_ind, ind in enumerate(range(start, end, self.step_size)):
            if ind < 0 or ind > max_ind:
                continue
            else:
                if self.read_vid_file:
                    frame_seg[:, out_ind, :, :] = self.frames[:, ind, :, :]
                else:
                    frame_seg[:, out_ind, :, :] = self._read_img_file(
                        os.path.join(self.vid_path, self.vid_id), self.frames[ind]
                    )

        # create the pathways
        frame_list = pack_pathway_output(self.cfg, frame_seg)

        return frame_list

    def __len__(self):
        """
        Returns:
            (int): the number of frames in the video.
        """
        # return self.video_container.streams.video[0].frames
        if self.read_vid_file:
            return self.frames.shape[1]
        else:
            return len(self.frames)
