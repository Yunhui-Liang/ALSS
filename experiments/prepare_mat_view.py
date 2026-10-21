"""Convert one local MATLAB multi-view dataset into the release NPZ format.

The script does not download or redistribute the source dataset. It selects a
single 1-based feature view and records the source hash and preprocessing rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, normalize


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path: Path, key: str) -> np.ndarray:
    keys = [key] if key != "auto" else ["y", "Y"]
    for candidate in keys:
        record = loadmat(path, variable_names=[candidate], squeeze_me=True)
        if candidate in record:
            labels = np.asarray(record[candidate]).reshape(-1)
            if labels.size:
                return labels
    raise KeyError(f"No nonempty label variable found; tried {keys}")


def load_view(path: Path, view_index: int, label_key: str) -> tuple[np.ndarray, np.ndarray]:
    if view_index < 1:
        raise ValueError("view-index is 1-based and must be positive")
    labels = load_labels(path, label_key)
    record = loadmat(path, variable_names=["X"], squeeze_me=True)
    if "X" not in record:
        raise KeyError("MAT-file has no variable named 'X'")
    values = record["X"]
    if isinstance(values, np.ndarray) and values.dtype == object:
        views = list(values.reshape(-1))
    else:
        views = [values]
    if view_index > len(views):
        raise IndexError(f"Requested view {view_index}, but the MAT-file contains {len(views)} views")
    value = views[view_index - 1]
    X = value.toarray() if sparse.issparse(value) else np.asarray(value)
    if X.ndim != 2:
        raise ValueError(f"Selected view must be a matrix, got {X.shape}")
    if X.shape[0] != labels.size and X.shape[1] == labels.size:
        X = X.T
    if X.shape[0] != labels.size:
        raise ValueError(f"View has {X.shape[0]} rows but labels have {labels.size} entries")
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Selected view contains non-finite values")
    return np.ascontiguousarray(X), labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mat", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="Stable dataset key used in result CSVs")
    parser.add_argument("--view-index", type=int, required=True, help="1-based view index")
    parser.add_argument("--preprocessing", choices=("standard", "row_l2", "none"), default="standard")
    parser.add_argument("--label-key", choices=("auto", "y", "Y"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_mat.resolve()
    X, y = load_view(source, args.view_index, args.label_key)
    if args.preprocessing == "standard":
        X = StandardScaler().fit_transform(X)
    elif args.preprocessing == "row_l2":
        X = normalize(X, norm="l2", axis=1, copy=True)
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y)
    metadata = {
        "dataset": args.dataset,
        "view": f"view_{args.view_index:02d}",
        "display_name": f"{args.dataset} view {args.view_index}",
        "k": int(np.unique(y).size),
        "preprocessing": "none",
        "preprocessing_applied_during_prepare": args.preprocessing,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "source_file_name": source.name,
        "source_sha256": sha256_file(source),
        "source_distribution": "local user-provided MAT-file; not bundled by this script",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()

