import os

from .wan_multitalk_1_3B import multitalk_1_3B


os.environ["TOKENIZERS_PARALLELISM"] = "false"

WAN_CONFIGS = {"infinitetalk-1.3B": multitalk_1_3B}
SIZE_CONFIGS = {
    "infinitetalk-480": (640, 640),
    "infinitetalk-720": (960, 960),
}
SUPPORTED_SIZES = {
    "infinitetalk-1.3B": tuple(SIZE_CONFIGS),
}
