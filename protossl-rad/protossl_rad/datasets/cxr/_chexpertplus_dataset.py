from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from ...defines import SPLIT_T, VIEW_T
from .._utils import load_cached_metadata, read_image_sizes
from ._base_cxr_dataset import BaseCXRDataset

SPLIT_MAPPING = {"train": "train", "validate": "val", "test": "test"}
# patients from the original CheXpert validation set (the CheXpert+ test split) live under PNG/valid
FIRST_VALID_PATIENT = 64541


def get_chexpertplus_metadata(dataset_path: str) -> pd.DataFrame:
    # one row per image, cached since the image sizes are read from the PNG headers
    _path = Path(dataset_path)
    source_files = [str(_path / "split.csv"), str(_path / "metadata.csv")]

    def build_fn() -> pd.DataFrame:
        df = pd.read_csv(_path / "split.csv", dtype=str)
        meta = pd.read_csv(_path / "metadata.csv", dtype=str)
        df = df.merge(
            meta,
            on=["subject_id", "study_id", "dicom_id"],
            how="left",
            validate="one_to_one",
        )
        df = df.rename(columns={"dicom_id": "image_id", "subject_id": "patient_id"})
        df["split"] = df["split"].map(SPLIT_MAPPING)
        # ViewPosition is AP / PA / Lateral / LL / RL, fall back to frontal_lateral if it's ever missing
        df["view_raw"] = df["ViewPosition"].fillna(df["frontal_lateral"])
        # patient00001_study1_view1_frontal -> PNG/train/patient00001/study1/view1_frontal.png
        parts = df["image_id"].str.split("_", n=2, expand=True)
        folder = (df["patient_id"].str[len("patient") :].astype(int) >= FIRST_VALID_PATIENT).map(
            {True: "valid", False: "train"}
        )
        df["rel_path"] = "PNG/" + folder + "/" + parts[0] + "/" + parts[1] + "/" + parts[2] + ".png"
        df["width"], df["height"] = read_image_sizes(
            [str(_path / p) for p in df["rel_path"]],
            desc="Reading chexpertplus image sizes",
        )
        return df

    return load_cached_metadata(name="chexpertplus", build_fn=build_fn, source_files=source_files)


class ChexpertPlusDataset(BaseCXRDataset):
    name = "chexpertplus"

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
        df = get_chexpertplus_metadata(dataset_path)
        # TODO: labels (chexbert_labels.zip, report-level)
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
