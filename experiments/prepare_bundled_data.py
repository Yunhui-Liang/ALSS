"""Prepare the bundled UCI and scikit-learn datasets for reproduction runs."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_iris, load_wine


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "data" / "upstream" / "multiple+features.zip"
BUNDLED = ROOT / "data" / "bundled"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def save_bundle(name: str, X: np.ndarray, y: np.ndarray, metadata: dict) -> dict:
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(np.asarray(y).reshape(-1))
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(f"{name}: X and y shapes do not agree")
    out = BUNDLED / f"{name}.npz"
    np.savez_compressed(
        out,
        X=X,
        y=y,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        **metadata,
        "file": out.relative_to(ROOT).as_posix(),
        "file_sha256": sha256_file(out),
        "X_shape": list(X.shape),
        "X_dtype": str(X.dtype),
        "X_sha256": sha256_array(X),
        "y_shape": list(y.shape),
        "y_dtype": str(y.dtype),
        "y_sha256": sha256_array(y),
    }


def load_mfeat_view6() -> tuple[np.ndarray, np.ndarray]:
    if not UPSTREAM.is_file():
        raise FileNotFoundError(f"Missing UCI source archive: {UPSTREAM}")
    with zipfile.ZipFile(UPSTREAM) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == "mfeat-mor"]
        if len(matches) != 1:
            raise ValueError(f"Expected one mfeat-mor file in UCI archive, found {matches}")
        X = np.loadtxt(io.BytesIO(archive.read(matches[0])), dtype=np.float64)
    if X.shape != (2000, 6):
        raise ValueError(f"Expected MFeat morphological view shape (2000, 6), got {X.shape}")
    # UCI stores 200 consecutive examples for each digit class 0 through 9.
    y = np.repeat(np.arange(10, dtype=np.int64), 200)
    return X, y


def main() -> None:
    BUNDLED.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    wine = load_wine()
    mfeat_X, mfeat_y = load_mfeat_view6()

    records = [
        save_bundle(
            "iris",
            iris.data,
            iris.target,
            {
                "dataset": "iris",
                "view": "view_01",
                "display_name": "Iris",
                "k": 3,
                "preprocessing": "standard",
                "source": "UCI Iris; loaded from scikit-learn 1.3.0 bundled copy",
                "source_url": "https://archive.ics.uci.edu/dataset/53/iris",
                "source_doi": "10.24432/C56C76",
                "license": "CC BY 4.0",
            },
        ),
        save_bundle(
            "wine",
            wine.data,
            wine.target,
            {
                "dataset": "wine",
                "view": "view_01",
                "display_name": "Wine",
                "k": 3,
                "preprocessing": "standard",
                "source": "UCI Wine; loaded from scikit-learn 1.3.0 bundled copy",
                "source_url": "https://archive.ics.uci.edu/dataset/109/wine",
                "source_doi": "10.24432/C5PC7J",
                "license": "CC BY 4.0",
            },
        ),
        save_bundle(
            "mfeat_view6",
            mfeat_X,
            mfeat_y,
            {
                "dataset": "mfeat",
                "view": "view_06",
                "display_name": "MFeat view 6 (mfeat-mor)",
                "k": 10,
                "preprocessing": "standard",
                "source": "UCI Multiple Features, morphological descriptor view",
                "source_url": "https://archive.ics.uci.edu/dataset/72/multiple+features",
                "source_doi": "10.24432/C5HC70",
                "source_archive": "data/upstream/multiple+features.zip",
                "source_archive_sha256": sha256_file(UPSTREAM),
                "source_member": "mfeat-mor",
                "license": "CC BY 4.0",
            },
        ),
    ]
    manifest = {
        "prepared_by": "experiments/prepare_bundled_data.py",
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "datasets": {record["dataset"] + ":" + record["view"]: record for record in records},
    }
    out = ROOT / "data" / "bundled_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

