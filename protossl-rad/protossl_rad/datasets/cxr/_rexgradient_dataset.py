import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from ...defines import SPLIT_T, VIEW_T
from .._utils import load_cached_metadata, read_image_sizes
from ._base_cxr_dataset import BaseCXRDataset

SPLIT_MAPPING = {"train": "train", "valid": "val", "test": "test"}


def get_rexgradient_metadata(dataset_path: str) -> pd.DataFrame:
    # one row per image, parsed from the per-split study-level json (the train json alone is ~250 MB)
    meta_dir = Path(dataset_path) / "metadata"
    source_files = [str(meta_dir / f"{s}_metadata_view_position.json") for s in SPLIT_MAPPING]

    def build_fn() -> pd.DataFrame:
        rows = []
        for source_split, f in zip(SPLIT_MAPPING, source_files):
            with open(f) as fh:
                studies = json.load(fh)
            for study_id, study in studies.items():
                for image_path, view in zip(study["ImagePath"], study["ImageViewPosition"], strict=True):
                    # ../deid_png/<patient>/<accession>/studies/<study uid>/series/<series uid>/instances/<uid>.png
                    rel_path = os.path.normpath(os.path.join("metadata", image_path))
                    rows.append(
                        {
                            "image_id": os.path.splitext(os.path.basename(rel_path))[0],
                            "patient_id": study["PatientID"],
                            "study_id": study_id,
                            "split": SPLIT_MAPPING[source_split],
                            "view_raw": view,
                            "rel_path": rel_path,
                        }
                    )
        df = pd.DataFrame(rows)
        # the json's ImageShape is the original DICOM size, the deid PNGs are downsampled so read their headers
        df["width"], df["height"] = read_image_sizes(
            [os.path.join(dataset_path, p) for p in df["rel_path"]],
            desc="Reading rexgradient image sizes",
        )
        return df

    return load_cached_metadata(name="rexgradient", build_fn=build_fn, source_files=source_files)


class RexGradientDataset(BaseCXRDataset):
    name = "rexgradient"

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
        df = get_rexgradient_metadata(dataset_path)
        # TODO: labels (metadata/*_chexbert_labels.csv, study-level, findings + impression)
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
