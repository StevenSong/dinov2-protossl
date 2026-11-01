import hashlib
import os
from collections.abc import Callable, Sequence
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..defines import CACHE_DIR, CACHE_SHORT_EDGE, CACHE_VERSION

# reads a source image given its path, returning a PIL image in the mode to be cached (e.g. 8-bit "L"
# for CXRs, see cxr/_cxr_image_io.py). must be a module-level function so it can be pickled into
# cache-building / dataloader workers
SourceReader = Callable[[str], Image.Image]


def downsample_short_edge(im: Image.Image, short_edge: int = CACHE_SHORT_EDGE) -> Image.Image:
    # only downsample, smaller images are left as is (training crops / eval resizes rescale anyway)
    w, h = im.size
    if min(w, h) <= short_edge:
        return im
    if w <= h:
        size = (short_edge, round(h * short_edge / w))
    else:
        size = (round(w * short_edge / h), short_edge)
    return im.resize(size, Image.Resampling.BICUBIC, reducing_gap=3.0)


def get_cache_root(name: str) -> str:
    return os.path.join(CACHE_DIR, name, f"se{CACHE_SHORT_EDGE}_v{CACHE_VERSION}")


def atomic_save_png(im: Image.Image, path: str):
    # write to a temp file in the same dir, then rename, so a concurrent reader never sees a partial file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.png"
    try:
        im.save(tmp, format="PNG")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _cache_one(args: tuple[str, str, SourceReader]) -> str | None:
    src, dst, reader = args
    try:
        atomic_save_png(downsample_short_edge(reader(src)), dst)
    except Exception as e:  # collect failures rather than killing the whole pool
        return f"{src}: {e!r}"
    return None


def build_image_cache(
    *,  # enforce kwargs
    src_paths: Sequence[str],
    cache_paths: Sequence[str],
    reader: SourceReader,
    num_workers: int,
):
    todo = [(s, d, reader) for s, d in zip(src_paths, cache_paths) if not os.path.exists(d)]
    print(f"Caching {len(todo)} images with {num_workers} workers")
    failures = []
    with Pool(num_workers) as pool:
        for err in tqdm(pool.imap_unordered(_cache_one, todo, chunksize=64), total=len(todo)):
            if err is not None:
                failures.append(err)
    if failures:
        msg = "\n".join(failures[:20])
        raise RuntimeError(f"Failed to cache {len(failures)} images, first few:\n{msg}")


class CachedImages:
    """Indexable collection of images, read from the preprocessed cache when available, otherwise
    read and preprocessed from source on the fly."""

    def __init__(
        self,
        *,  # enforce kwargs
        src_paths: Sequence[str],
        cache_paths: Sequence[str] | None,
        reader: SourceReader,
    ):
        # fixed-width numpy byte strings instead of python lists of str, so dataloader workers
        # don't slowly copy the list through refcount updates (copy-on-read with fork)
        self._src_paths = np.asarray([os.fsencode(p) for p in src_paths])
        self._cache_paths = None if cache_paths is None else np.asarray([os.fsencode(p) for p in cache_paths])
        self.reader = reader

    def __getitem__(self, i: int) -> Image.Image:
        if self._cache_paths is not None:
            with Image.open(os.fsdecode(self._cache_paths[i])) as im:
                im.load()
            return im
        return downsample_short_edge(self.reader(os.fsdecode(self._src_paths[i])))

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._src_paths),)

    def __len__(self) -> int:
        return self.shape[0]


def load_cached_images(
    *,  # enforce kwargs
    name: str,
    dataset_path: str,
    split: str,
    rel_paths: Sequence[str],
    reader: SourceReader,
    use_cache: bool = True,
    build_cache: bool = False,
    num_workers: int | None = None,
) -> CachedImages:
    src_paths = [os.path.join(dataset_path, p) for p in rel_paths]
    print("================load_cached_images=================")
    print(f"Dataset parameters: ({name}, {dataset_path}, {split}, {len(rel_paths)} images)")
    if not use_cache:
        print("Not using cache, images will be read and preprocessed from source on the fly")
        print("===================================================")
        return CachedImages(src_paths=src_paths, cache_paths=None, reader=reader)

    root = get_cache_root(name)
    cache_paths = [os.path.join(root, os.path.splitext(p)[0] + ".png") for p in rel_paths]

    # checking ~10^5 files on every init (and every rank) is slow, so record when a set of images is fully cached
    key = hashlib.md5("\n".join(sorted(rel_paths)).encode("utf-8")).hexdigest()[:16]
    marker = os.path.join(root, f"_complete_{key}")
    if not os.path.exists(marker):
        n_missing = sum(not os.path.exists(p) for p in cache_paths)
        if n_missing > 0:
            if not build_cache:
                raise FileNotFoundError(
                    f"{n_missing}/{len(cache_paths)} images of {name} are not cached under {root}, build the "
                    f"cache first (see protossl_rad.tools, e.g. cache_cxr_images) or pass build_cache=True"
                )
            build_image_cache(
                src_paths=src_paths,
                cache_paths=cache_paths,
                reader=reader,
                num_workers=num_workers or max(1, (os.cpu_count() or 2) // 2),
            )
        # the marker only speeds up later inits, so a read-only cache (e.g. a shared mount) is fine
        try:
            os.makedirs(root, exist_ok=True)
            with open(marker, "w") as f:
                f.write(f"{len(cache_paths)}\n")
        except OSError as e:
            print(f"Couldn't write cache marker ({e!r}), cache completeness will be rechecked on every init")
    print(f"Using cached images under {root}")
    print("===================================================")
    return CachedImages(src_paths=src_paths, cache_paths=cache_paths, reader=reader)
