import hashlib
import os
from collections.abc import Callable, Sequence

import pandas as pd

from ..defines import CACHE_DIR, CACHE_VERSION


def load_cached_metadata(
    *,  # enforce kwargs
    name: str,
    build_fn: Callable[[], pd.DataFrame],
    source_files: Sequence[str],
) -> pd.DataFrame:
    # parsing some source metadata is slow (e.g. large json), so cache the parsed table. keyed on the
    # source files' size/mtime so updated sources rebuild the cache
    fingerprint = "_".join(f"{f}:{os.path.getsize(f)}:{os.path.getmtime(f)}" for f in source_files)
    hashed = hashlib.md5(f"{name}_{CACHE_VERSION}_{fingerprint}".encode("utf-8")).hexdigest()[:8]
    cache_file = os.path.join(CACHE_DIR, name, f"metadata_{hashed}.csv")
    if not os.path.exists(cache_file):
        df = build_fn()
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        tmp = f"{cache_file}.{os.getpid()}.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, cache_file)
    # always read back from disk so a fresh build and a cache hit give identical frames
    return pd.read_csv(cache_file, dtype=str, keep_default_na=False)


def validate_label_subset(label_subset: list[str], labels: list[str]):
    _labels = set(labels)
    for label in label_subset:
        if label not in _labels:
            raise ValueError(f"'{label}' is not in the current label set")
