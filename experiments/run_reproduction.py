"""Reproduce ALSS and BKM comparison rows on bundled or prepared datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.prepare_bundled_data import main as prepare_bundled_data
from src.bkm import BKM
from src.evaluation import evaluate_clustering


METHODS = {
    "BKM-Largest": ("largest", False),
    "BKM-Inertia": ("inertia", False),
    "BKM-MeanSSE": ("mean_sse", False),
    "BKM-Savaresi-J": ("savaresi", False),
    "BKM-Gain": ("gain", False),
    "ALSS": ("alss", False),
    "ALSS-BudgetAware": ("alss", True),
    "Flat-KMeans": ("flat", False),
}
DEFAULT_METHODS = list(METHODS)
RESULT_FIELDS = [
    "dataset", "view", "n_samples", "n_features", "k", "method", "seed",
    "lambda", "top_r", "n_init", "ACC", "NMI", "ARI", "SSE",
    "kmeans_fit_calls", "fit_seconds", "input_sha256", "label_sha256",
]


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


def read_bundle(path: Path, default_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"]).reshape(-1)
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))
        else:
            metadata = {}
    metadata.setdefault("dataset", default_name)
    metadata.setdefault("view", "view_01")
    metadata.setdefault("display_name", default_name)
    metadata.setdefault("preprocessing", "standard")
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(f"Invalid X/y dimensions in {path}")
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite feature values in {path}")
    return X, y, metadata


def load_data(item: str) -> tuple[np.ndarray, np.ndarray, dict, Path]:
    bundled = ROOT / "data" / "bundled" / f"{item}.npz"
    path = bundled if bundled.is_file() else Path(item).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset '{item}' is neither a bundled dataset nor an existing NPZ path. "
            "Run prepare_bundled_data.py or prepare a local paper view first."
        )
    X, y, metadata = read_bundle(path, Path(item).stem)
    preprocess = metadata.get("preprocessing", "standard")
    if preprocess == "standard":
        X = StandardScaler().fit_transform(X)
    elif preprocess == "row_l2":
        X = normalize(X, norm="l2", axis=1, copy=True)
    elif preprocess != "none":
        raise ValueError(f"Unsupported preprocessing in metadata: {preprocess}")
    return np.ascontiguousarray(X, dtype=np.float64), y, metadata, path


def run_one(
    X: np.ndarray,
    y: np.ndarray,
    dataset: str,
    view: str,
    method: str,
    seed: int,
    lambda_: float,
    top_r: int,
    n_init: int,
) -> dict:
    k = int(np.unique(y).size)
    strategy, horizon_aware = METHODS[method]
    started = perf_counter()
    with threadpool_limits(limits=1):
        if strategy == "flat":
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=n_init,
                random_state=seed,
                algorithm="lloyd",
            ).fit(X)
            labels = model.labels_
            fit_calls = 1
        else:
            model = BKM(
                n_clusters=k,
                strategy=strategy,
                lambda_=lambda_,
                top_r=top_r,
                n_init=n_init,
                random_state=seed,
                horizon_aware=horizon_aware,
            ).fit(X)
            labels = model.labels_
            fit_calls = int(model.kmeans_fit_calls_)
    fit_seconds = perf_counter() - started
    scores = evaluate_clustering(X, y, labels)
    return {
        "dataset": dataset,
        "view": view,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "k": k,
        "method": method,
        "seed": int(seed),
        "lambda": float(lambda_),
        "top_r": int(top_r),
        "n_init": int(n_init),
        "ACC": scores["acc"],
        "NMI": scores["nmi"],
        "ARI": scores["ari"],
        "SSE": scores["sse"],
        "kmeans_fit_calls": fit_calls,
        "fit_seconds": fit_seconds,
        "input_sha256": sha256_array(X),
        "label_sha256": sha256_array(np.asarray(y)),
    }


def verify_reference(rows: list[dict], reference_path: Path) -> dict:
    with reference_path.open(newline="", encoding="utf-8-sig") as stream:
        reference = list(csv.DictReader(stream))
    reference_by_key = {
        (r["dataset"], r["view"], r["method"], int(r["seed"])): r
        for r in reference
    }
    comparisons = []
    for row in rows:
        if row["dataset"] != "mfeat":
            continue
        key = (row["dataset"], row["view"], row["method"], int(row["seed"]))
        expected = reference_by_key.get(key)
        if expected is None:
            continue
        metric_errors = {}
        for output_name, reference_name in (("ACC", "ACC"), ("NMI", "NMI"), ("ARI", "ARI"), ("SSE", "SSE")):
            actual = float(row[output_name])
            target = float(expected[reference_name])
            atol = 1e-8 if output_name == "SSE" else 1e-12
            if not np.isclose(actual, target, rtol=1e-12, atol=atol):
                raise AssertionError(f"{key} {output_name}: {actual} != reference {target}")
            metric_errors[output_name] = actual - target
        comparisons.append({"key": list(key), "metric_errors": metric_errors})
    if not comparisons:
        raise AssertionError("No MFeat rows matched the frozen paper reference")
    return {
        "status": "PASS",
        "reference_csv": reference_path.name,
        "matched_rows": len(comparisons),
        "max_abs_metric_error": {
            metric: max(abs(item["metric_errors"][metric]) for item in comparisons)
            for metric in ("ACC", "NMI", "ARI", "SSE")
        },
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mfeat_view6", "iris", "wine"])
    parser.add_argument("--methods", nargs="+", choices=["all", *METHODS], default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--top-r", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-paper-reference", action="store_true")
    args = parser.parse_args()

    # Prepare the small, license-attributed input archive when bundles are absent.
    if any(name in {"iris", "wine", "mfeat_view6"} and not (ROOT / "data" / "bundled" / f"{name}.npz").is_file() for name in args.datasets):
        prepare_bundled_data()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (ROOT / "results" / "reproduced" / f"run_{stamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    methods = list(METHODS) if "all" in args.methods else args.methods
    all_rows = []
    source_records = []
    for item in args.datasets:
        X, y, metadata, path = load_data(item)
        dataset = metadata.get("dataset", Path(item).stem)
        view = metadata.get("view", "view_01")
        source_records.append({
            "argument": item,
            "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "file_sha256": sha256_file(path),
            "dataset": dataset,
            "view": view,
            "display_name": metadata.get("display_name", dataset),
            "preprocessing": metadata.get("preprocessing", "standard"),
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "k": int(np.unique(y).size),
            "X_sha256_after_preprocessing": sha256_array(X),
            "y_sha256": sha256_array(y),
        })
        print(f"{dataset}/{view}: n={X.shape[0]} d={X.shape[1]} k={np.unique(y).size}", flush=True)
        for seed in args.seeds:
            for method in methods:
                row = run_one(X, y, dataset, view, method, seed, args.lambda_, args.top_r, args.n_init)
                all_rows.append(row)
                print(
                    f"  seed={seed} {method}: ACC={row['ACC']:.6f} NMI={row['NMI']:.6f} "
                    f"ARI={row['ARI']:.6f} SSE={row['SSE']:.6f}",
                    flush=True,
                )

    result_path = output_dir / "per_seed_results.csv"
    with result_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    verify_report = None
    if args.verify_paper_reference:
        reference_path = ROOT / "results" / "reference" / "quality_7sets_8methods_5seeds.csv"
        verify_report = verify_reference(all_rows, reference_path)
        (output_dir / "paper_reference_validation.json").write_text(
            json.dumps(verify_report, indent=2) + "\n", encoding="utf-8"
        )

    metadata = {
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "seeds": args.seeds,
            "n_init": args.n_init,
            "init": "k-means++",
            "lambda": args.lambda_,
            "top_r": args.top_r,
            "thread_limit": 1,
            "preprocessing": "StandardScaler fit independently to each bundled/prepared view, unless local NPZ metadata specifies row_l2 or none",
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "source_sha256": {
            name: sha256_file(ROOT / "src" / name)
            for name in ("bkm.py", "split_rules.py", "evaluation.py", "parallel_candidate_eval.py")
        },
        "input_files": source_records,
        "rows": len(all_rows),
        "paper_reference_validation": verify_report,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved: {result_path}")
    if verify_report:
        print(f"Paper reference check: {verify_report['status']} ({verify_report['matched_rows']} matched rows)")


if __name__ == "__main__":
    main()

