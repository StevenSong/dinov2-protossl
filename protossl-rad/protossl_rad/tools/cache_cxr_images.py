import argparse

from protossl_rad.datasets.cxr import DATASETS, make_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and cache CXR images for the given datasets")
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "PATH"),
        help=f"dataset name ({', '.join(DATASETS)}) and full path to its root, repeat for multiple datasets",
    )
    parser.add_argument("--split", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    for name, _ in args.dataset:
        if name not in DATASETS:
            parser.error(f"unknown dataset {name}, expected one of {list(DATASETS)}")

    for name, path in args.dataset:
        for split in args.split:
            # caches any images that aren't already cached
            ds = make_dataset(name, dataset_path=path, split=split, build_cache=True)
            print(f"{name} {split}: {len(ds)} images cached")
