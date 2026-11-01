from collections.abc import Mapping
from typing import Type

from torch.utils.data import ConcatDataset

from ...defines import SPLIT_T
from ._base_cxr_dataset import BaseCXRDataset
from ._chexpertplus_dataset import ChexpertPlusDataset
from ._cxr_image_io import read_cxr_image, to_uint8_gray
from ._mimic_cxr_dataset import MimicCXRDataset
from ._rexgradient_dataset import RexGradientDataset

DATASETS: dict[str, Type[BaseCXRDataset]] = {
    "chexpertplus": ChexpertPlusDataset,
    "mimic-cxr": MimicCXRDataset,
    "rexgradient": RexGradientDataset,
}
# used for self-supervised pretraining, downstream datasets must never be added here
PRETRAIN_DATASETS = ["chexpertplus", "mimic-cxr", "rexgradient"]


def make_dataset(name: str, *, dataset_path: str, **kwargs) -> BaseCXRDataset:
    """
    :param name: dataset name, one of `DATASETS`
    :param dataset_path: full path to the root of the source dataset (e.g. `/path/to/mimic-cxr`)
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}, expected one of {list(DATASETS)}")
    return DATASETS[name](dataset_path=dataset_path, **kwargs)


def make_pretrain_dataset(
    *,  # enforce kwargs
    dataset_paths: Mapping[str, str],
    split: SPLIT_T = "train",
    **kwargs,
) -> ConcatDataset:
    """
    :param dataset_paths: pretraining dataset name -> full path to the root of that source dataset,
        the datasets used are exactly the keys given
    """
    if not dataset_paths:
        raise ValueError("dataset_paths must name at least one pretraining dataset")
    unknown = set(dataset_paths) - set(PRETRAIN_DATASETS)
    if unknown:
        raise ValueError(f"{unknown} are not pretraining datasets, expected a subset of {PRETRAIN_DATASETS}")
    # all views are kept by default, as in RAD-DINO
    return ConcatDataset(
        [make_dataset(name, dataset_path=path, split=split, **kwargs) for name, path in dataset_paths.items()]
    )
