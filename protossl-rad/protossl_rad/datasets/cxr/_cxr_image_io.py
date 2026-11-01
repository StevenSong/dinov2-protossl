import numpy as np
from PIL import Image


def to_uint8_gray(im: Image.Image) -> Image.Image:
    if im.mode in ("I;16", "I;16L", "I;16B", "I"):
        # 16-bit sources we've seen (e.g. ReXGradient) are already display-windowed over the
        # full 16-bit range, so scale linearly rather than re-windowing per image
        a = np.asarray(im, dtype=np.float32)
        a = np.clip(np.rint(a / 257.0), 0, 255).astype(np.uint8)
        return Image.fromarray(a)
    if im.mode != "L":
        return im.convert("L")
    return im


def read_cxr_image(path: str) -> Image.Image:
    # reads an 8/16-bit grayscale (or RGB) image file as 8-bit grayscale, the mode CXRs are cached in
    with Image.open(path) as im:
        im.load()
    return to_uint8_gray(im)
