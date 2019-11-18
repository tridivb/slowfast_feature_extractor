#!/usr/bin/env python3

import slowfast.config.defaults as defcfg


# -----------------------------------------------------------------------------
# Additional Data options
# -----------------------------------------------------------------------------

# Fps of the input video
defcfg._C.DATA.IN_FPS = 60

# Fps to sample the frames for output
defcfg._C.DATA.OUT_FPS = 30

# Flag to set video file/image file processing
defcfg._C.DATA.READ_VID_FILE = True

# File extension of video files
defcfg._C.DATA.VID_FILE_EXT = ".MP4"

# File extension of image files
defcfg._C.DATA.IMG_FILE_EXT = ".jpg"

# File naming format of image files
defcfg._C.DATA.IMG_FILE_FORMAT = "frame_{:010d}.jpg"

# Sampling height and width of each frame
defcfg._C.DATA.SAMPLE_SIZE = [256, 256]


def get_cfg():
    """
    Get a copy of the default config.
    """
    return defcfg._assert_and_infer_cfg(defcfg._C.clone())
