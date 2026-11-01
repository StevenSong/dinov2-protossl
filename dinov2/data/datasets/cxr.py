# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from torch.utils.data import Dataset


class CXRPretrain(Dataset):
    """
    Union of the chest X-ray pretraining datasets from the protossl_rad package. Each source dataset is
    given as a <dataset name>=<full path to dataset root> token, and exactly those datasets are used, e.g.
        CXRPretrain:split=train:mimic-cxr=/path/to/mimic-cxr:chexpertplus=/path/to/chexpertplus
    where names are from protossl_rad.datasets.cxr.PRETRAIN_DATASETS. Uses the pretraining defaults of
    make_pretrain_dataset (frontal views only, source short edge >= 518px). The cache is under PROTOSSL_RAD_CACHE_DIR.
    """

    def __init__(
        self,
        *,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **dataset_paths: str,
    ) -> None:
        # imported here so dinov2 doesn't require protossl_rad unless CXR data is used
        from protossl_rad.datasets.cxr import make_pretrain_dataset

        self.dataset = make_pretrain_dataset(dataset_paths=dataset_paths, split=split)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        image = self.dataset[index]["image"]
        target = None
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self) -> int:
        return len(self.dataset)
