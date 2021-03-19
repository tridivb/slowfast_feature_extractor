#!/usr/bin/env python3

import slowfast.config.defaults as defcfg
import slowfast.utils.checkpoint as cu


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
    return defcfg.assert_and_infer_cfg(defcfg._C.clone())


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    # cu.make_checkpoint_dir(cfg.OUTPUT_DIR)

    return cfg
