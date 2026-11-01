from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from ...defines import SPLIT_T, VIEW_T
from ._base_cxr_dataset import BaseCXRDataset

SPLIT_MAPPING = {"train": "train", "validate": "val", "test": "test"}


class MimicCXRDataset(BaseCXRDataset):
    name = "mimic-cxr"

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
        _path = Path(dataset_path)
        df = pd.read_csv(_path / "mimic-cxr-2.0.0-split.csv", dtype=str)
        meta = pd.read_csv(
            _path / "mimic-cxr-2.0.0-metadata.csv",
            usecols=["dicom_id", "ViewPosition", "Rows", "Columns"],
            dtype=str,
        )
        df = df.merge(meta, on="dicom_id", how="left", validate="one_to_one")
        df = df.rename(
            columns={
                "dicom_id": "image_id",
                "subject_id": "patient_id",
                "ViewPosition": "view_raw",
                # the JPGs keep the DICOM resolution
                "Rows": "height",
                "Columns": "width",
            }
        )
        df["split"] = df["split"].map(SPLIT_MAPPING)
        # files/p10/p10000032/s50414267/<dicom_id>.jpg
        df["rel_path"] = (
            "files/p"
            + df["patient_id"].str[:2]
            + "/p"
            + df["patient_id"]
            + "/s"
            + df["study_id"]
            + "/"
            + df["image_id"]
            + ".jpg"
        )
        # TODO: labels (chexpert / negbio / chexbert are per-study)
        self.labels = None
        self._setup(
            full_df=df,
            dataset_path=dataset_path,
            split=split,
            views=views,
            min_short_edge=min_short_edge,
            transform=transform,
            as_rgb=as_rgb,
            use_cache=use_cache,
            build_cache=build_cache,
        )
