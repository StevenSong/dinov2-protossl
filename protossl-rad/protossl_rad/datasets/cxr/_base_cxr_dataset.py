from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ...defines import SPLIT_T, UNKNOWN_LABEL, VIEW_T, canonical_view
from .._image_cache import CachedImages, SourceReader, load_cached_images
from ._cxr_image_io import read_cxr_image

# columns every dataset's per-image metadata must have before calling `_setup`
REQUIRED_COLUMNS = ["image_id", "patient_id", "study_id", "split", "view_raw", "rel_path"]
# optional source image size columns (pixels, before cache downsampling), needed to filter by `min_short_edge`
SIZE_COLUMNS = ["width", "height"]


class BaseCXRDataset(Dataset, ABC):
    name: str  # cache subdirectory
    source_ids: torch.Tensor  # (N,), patient, N = n_images
    sample_ids: torch.Tensor  # (N,), image
    images: CachedImages  # N 8-bit grayscale images
    labels: torch.Tensor | None  # (N, C), C = n_labels
    _df: pd.DataFrame  # (N, ...) per-image metadata, row-aligned with the above

    @abstractmethod
    def __init__(
        self,
        *,  # enforce kwargs
        dataset_path: str,
        split: SPLIT_T,
        views: Sequence[VIEW_T] | None = None,
        min_short_edge: int | None = None,
        transform: Callable | None = None,
        as_rgb: bool = True,
        use_cache: bool = True,
        build_cache: bool = False,
    ):
        """
        :param dataset_path: path to the source dataset directory
        :param split: one of `"train"`, `"val"`, or `"test"`
        :param views: optional subset of canonical views (`"frontal"`, `"lateral"`, `"other"`) to keep
        :param min_short_edge: optionally drop images whose source short edge is smaller than this many pixels
        :param transform: optional transform applied to the PIL image
        :param as_rgb: replicate the grayscale image to 3 channels before `transform`
        :param use_cache: read preprocessed images from the cache, otherwise read from source on the fly
        :param build_cache: build any missing cache entries, otherwise raise if the cache is incomplete
        """
        pass

    def _setup(
        self,
        *,  # enforce kwargs
        full_df: pd.DataFrame,
        dataset_path: str,
        split: SPLIT_T,
        views: Sequence[VIEW_T] | None,
        min_short_edge: int | None,
        transform: Callable | None,
        as_rgb: bool,
        use_cache: bool,
        build_cache: bool,
        reader: SourceReader = read_cxr_image,
    ):
        # full_df covers every split, so ids are stable no matter which split/views are selected
        missing_cols = set(REQUIRED_COLUMNS) - set(full_df.columns)
        assert not missing_cols, f"metadata is missing columns: {missing_cols}"
        assert set(full_df["split"]) <= {"train", "val", "test"}, set(full_df["split"])
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if not full_df["image_id"].is_unique:
            raise ValueError(f"image_id is not unique in {self.name} metadata")

        full_df = full_df.sort_values("image_id").reset_index(drop=True)
        full_df["sample_id"] = np.arange(len(full_df))
        full_df["source_id"] = pd.factorize(full_df["patient_id"], sort=True)[0]
        full_df["view"] = full_df["view_raw"].map(canonical_view)
        has_size = set(SIZE_COLUMNS) <= set(full_df.columns)
        if has_size:
            full_df[SIZE_COLUMNS] = full_df[SIZE_COLUMNS].astype(int)

        mask = full_df["split"] == split
        if views is not None:
            mask &= full_df["view"].isin(views)
        if min_short_edge is not None:
            if not has_size:
                raise ValueError(f"{self.name} metadata has no image sizes, can't filter by min_short_edge")
            mask &= full_df[SIZE_COLUMNS].min(axis=1) >= min_short_edge
        df = full_df[mask].reset_index(drop=True)
        self._df = df

        self.source_ids = torch.tensor(df["source_id"].to_numpy())
        self.sample_ids = torch.tensor(df["sample_id"].to_numpy())
        self.transform = transform
        self.as_rgb = as_rgb
        self.images = load_cached_images(
            name=self.name,
            dataset_path=dataset_path,
            split=split,
            rel_paths=list(df["rel_path"]),
            reader=reader,
            use_cache=use_cache,
            build_cache=build_cache,
        )
        assert self.source_ids.shape[0] == len(self.images)
        assert self.source_ids.shape[0] == self.sample_ids.shape[0]

    def __getitem__(self, i: int) -> dict:
        image = self.images[i]
        if self.as_rgb:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        ret = {
            "image": image,
            "source_id": self.source_ids[i],  # (,)
            "sample_id": self.sample_ids[i],  # (,)
        }
        if self.labels is not None:
            ret["label"] = self.labels[i]  # (C,)
        return ret

    def __len__(self) -> int:
        return len(self.images)

    def get_label_weights(self) -> torch.Tensor:
        if self.labels is None:
            raise ValueError("This dataset does not have any labels!")
        # labels should always be one-hot encoded, unknown labels don't count
        n_known = (self.labels != UNKNOWN_LABEL).sum(dim=0)
        per_label_count = (self.labels == 1).sum(dim=0)
        return (n_known - per_label_count) / per_label_count
