import importlib
import os

from models.build import build_model, MODEL_REGISTRY
from .video_model_builder import ResNetFeat, SlowFastFeat
from .head_helper import ResNetBasicHeadFeat

MODEL_REGISTRY.register(ResNetFeat)
MODEL_REGISTRY.register(SlowFastFeat)

# #  walk through all the files in the directory recursively
# for root, dirs, files in os.walk(os.path.dirname(__file__), topdown=True):
#     for file in files:
#         # look for python files which start with an alphabet
#         if file.endswith(".py") and file[0].isalpha():
#             # get the relative path from the current folder and replace the
#             # os seperator with a .
#             rel_path = os.path.relpath(root, os.path.dirname(__file__)).replace(
#                 os.sep, "."
#             )
#             # get the base name of the file
#             file = file[: file.find(".py")]
#             # if the model file is in the current folder rel_path would just be .
#             # handle that by using an if condition
#             module = (
#                 f"models.{file}"
#                 if rel_path == "."
#                 else f"models.{rel_path}.{file}"
#             )
#             importlib.import_module(module)
