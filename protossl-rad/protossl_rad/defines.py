import os
from typing import Literal

# preprocessed images are cached here, see datasets/_image_cache.py
CACHE_DIR = os.environ.get(
    "PROTOSSL_RAD_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache/protossl-rad"),
)

SPLIT_T = Literal["train", "val", "test"]
# canonical view of an image, raw view labels are kept in the dataset's metadata df
VIEW_T = Literal["frontal", "lateral", "other"]
UNKNOWN_LABEL = -1

# images are cached as 8-bit grayscale, downsampled so the shortest edge is at most this
# (RAD-DINO trains from 518px copies, its global crops are 518px)
CACHE_SHORT_EDGE = 518
# bump when the cached image preprocessing changes so stale caches aren't reused
CACHE_VERSION = 1

# from microsoft/rad-dino preprocessor_config.json, grayscale replicated to 3 channels
RAD_DINO_MEAN = (0.5307, 0.5307, 0.5307)
RAD_DINO_STD = (0.2583, 0.2583, 0.2583)

# raw view labels across datasets (upper-cased) mapped to canonical views, anything else is "other"
FRONTAL_VIEWS = {
    "PA",
    "AP",
    "AP AXIAL",
    "PA LLD",
    "AP LLD",
    "AP RLD",
    "PA RLD",
    "POSTERO_ANTERIOR",
    "ANTERO_POSTERIOR",
    "AP_HORIZONTAL",
    "FRONTAL",
}
LATERAL_VIEWS = {
    "LATERAL",
    "LL",
    "RL",
    "LAT",
    "L",
    "XTABLE LATERAL",
    "LEFT LATERAL",
    "RIGHT LATERAL",
}


def canonical_view(raw: object) -> VIEW_T:
    if not isinstance(raw, str):
        return "other"
    raw = raw.strip().upper()
    if raw in FRONTAL_VIEWS:
        return "frontal"
    if raw in LATERAL_VIEWS:
        return "lateral"
    return "other"
