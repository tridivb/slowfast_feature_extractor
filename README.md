# SlowFast Feature Extractor

Extract features from videos with a pre-trained SlowFast model using the PySlowFast framework.

**Update**: The installation instructions has been updated for the latest Pytorch 1.6 and Torchvision 0.7 with Cuda 10.2. Please follow the new instructions to refresh the Pyslowfast installation if you had already done it before.

## Install requirements

1. Ubuntu 16.x/18.x (Only tested on these two systems)
2. Cuda 10.2
3. Python >= 3.7
4. [Pytorch](https://pytorch.org/)  >= 1.6
5. [PySlowFast](https://github.com/facebookresearch/SlowFast.git) >= 1.0
6. PyAv >= 8.x
7. Moviepy >= 1.0
8. OpenCV >= 4.x

It is recommended to use conda environment to install the dependencies.

You can create the conda environment with the command:

```
conda create -n "slowfast" python=3.7
```

Install Pytorch 1.6 and Torchvision 0.7 with conda or pip. (https://pytorch.org/get-started/locally/)

Install the following dependencies with pip:

```
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson av psutil opencv-python tensorboard moviepy cython
```

Install detectron2:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Setup Pyslowfast:
```
git clone https://github.com/facebookresearch/slowfast
export PYTHONPATH=/path/to/slowfast:$PYTHONPATH
cd slowfast
python setup.py build develop
```

## Getting Started

Clone the repo and set it up in your local drive.

```
git clone https://github.com/tridivb/slowfast_feature_extractor.git
```

## Data Preparation

The videos can be setup in the following way:

```
|---<path to dataset>
|   |---vid_list.csv
|   |---video_1.mp4
|   |---video_2
|   |   |---video_2.mp4
|   |   |---.
|   |---.

```

or pre-process the videos and extract the frames like below:
```
|---<path to dataset>
|   |---vid_list.csv
|   |---video_1
|   |   |---frame01.jpg
|   |   |---frame02.jpg
|   |   |---.
|   |---video_2
|   |   |---video_2
|   |   |   |---frame01.jpg
|   |   |   |---frame02.jpg
|   |   |   |---.
|   |---.

```

The vid_list.csv should have the paths of all the videos or subdirectories for extracted frames. 
All the videos/image files should have the same type of extension.
Based on the hierarchy above, it should be like:

```
video_1
video2/video_2
...
...
...
```

## Pretrained weights

Navigate to the slowfast_feature_extractor directory.

```
cd /path/to/slowfast_feature_extractor
```

Download the pre-trained [weights](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md) from the PySlowFast Model Zoo and copy it to your desired location.

## Configure the paramters

Use the existing config file in ./configs or copy over the corresponding one for your desired model from where you cloned the PySlowFast framework.

Set the following paths in the ./configs/<config_file>.yaml file:

```
TRAIN:
  # checkpoint file
  CHECKPOINT_FILE_PATH: ""
  # set this to pytorch or caffe2 depending on your checkpoint configuration
  # the default pre-trained weights from PySlowFast Model Zoo are in caffe2 format
  CHECKPOINT_TYPE: caffe2
DATA:
  # Root dir of dataset
  PATH_TO_DATA_DIR: ""
  # Path prefix for each video or subdirectory where extracted frames are kept
  PATH_PREFIX: ""
  # size of sampled window centered on each frame
  NUM_FRAMES: 32
  # original fps of input video
  IN_FPS: 15
  # fps value to sample videos at
  OUT_FPS: 15
  # Flag to turn on/off processing frames from video files. If False, it will try to read extracted image frames.
  READ_VID_FILE: True
  # File extension of video files (case-sensitive). Set this if you want to read the video files.
  VID_FILE_EXT: ".MP4"
  # File extension of image files (case-sensitive). Set this if you want to read the pre-processed frames.
  IMG_FILE_EXT: ".jpg"
  # File naming format of image files (case-sensitive). Set this if you want to read the pre-processed frames.
  IMG_FILE_FORMAT: "frame_{:010d}.jpg"
  # Sampling height and width of each extracted frame. This can be a list or int value
  SAMPLE_SIZE: [256, 256]

TEST:
  # be careful with this, inference will run faster with a higher value but can cause out of memory error
  BATCH_SIZE: 3

# output directory to save features
OUTPUT_DIR: ""
```

If you don't want to commit the config file, rename it as <config_file>.yaml.local.

## Extracting the features and detections

To extract features, execute the run_net.py as follows:

```
python run_net.py --cfg ./configs/<config_file>.yaml
```

For our case, we used the SlowFast network with a Resnet50 backbone, frame length of 8 and sample rate of 8.\
If you want to use a different model, copy over the corresponding config file and download the weights.

## Results

The detections are saved in the following format for each video:

```
|---<path to output>
|   |---video_1_{NUM_FRAMES}.npy
|   |---video_2
|   |   |---video_2_{NUM_FRAMES}.npy
|   |   |---.
|   |---.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Please note, the original PySlowFast frames is licensed under the Apache 2.0 license. Please respect the original licenses as well.

## Acknowledgments

1. The code was built on top of the [PySlowFast](https://github.com/facebookresearch/SlowFast.git) framework provided by Facebook. Some of the model and dataset code was modified to fit the needs of feature extraction from videos.

2. Readme Template -> https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
