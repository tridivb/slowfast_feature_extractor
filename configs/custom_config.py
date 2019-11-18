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


def get_cfg():
    """
    Get a copy of the default config.
    """
    return defcfg._assert_and_infer_cfg(defcfg._C.clone())
