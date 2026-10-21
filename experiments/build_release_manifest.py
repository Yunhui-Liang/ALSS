"""Verify frozen inputs and write a SHA-256 manifest for this release."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FROZEN = {
    "src/bkm.py": "765cd38e6493aa21ffc160c4749e8b19015901eadc5ee79b0d139e45f43efe31",
    "src/split_rules.py": "f625942027ae85377564d70f77c7c73a8bfd6f8d9c67c2da9fe294c91d53e80b",
    "src/evaluation.py": "e59d5c81720073349ea367cadd11665544653ccf40c8f12e210ded331aef6cb6",
}
EXPECTED_DATA = {
    "data/upstream/multiple+features.zip": "898a50a7637f1ed5a8cee2493aaa0e7f4d52795c412f59b0969ec7c3046ee7bd",
}
TRACKED_FILES = [
    "src/__init__.py", "src/bkm.py", "src/split_rules.py", "src/evaluation.py",
    "src/parallel.py", "src/parallel_candidate_eval.py",
    "experiments/prepare_bundled_data.py", "experiments/prepare_mat_view.py",
    "experiments/run_reproduction.py", "experiments/validate_parallel_equivalence.py",
    "experiments/build_release_manifest.py", "experiments/package_release.py",
    "requirements.txt", ".gitignore", ".gitattributes",
    "README.md", "DATASET_SOURCES.md",
    "REPRODUCTION_REPORT.md", "data/paper_views_local/README.md",
    "results/reference/quality_7sets_8methods_5seeds.csv",
    "data/upstream/multiple+features.zip", "data/bundled/iris.npz",
    "data/bundled/wine.npz", "data/bundled/mfeat_view6.npz",
    "data/bundled_manifest.json",
    "results/reproduced/reference_validation/per_seed_results.csv",
    "results/reproduced/reference_validation/paper_reference_validation.json",
    "results/reproduced/reference_validation/run_metadata.json",
    "results/reproduced/reference_validation/parallel_equivalence.json",
    "results/reproduced/iris_wine_smoke/per_seed_results.csv",
    "results/reproduced/iris_wine_smoke/run_metadata.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    frozen = {}
    for relative, expected in EXPECTED_FROZEN.items():
        actual = sha256(ROOT / relative)
        frozen[relative] = {"sha256": actual, "expected_sha256": expected, "match": actual == expected}
    data = {}
    for relative, expected in EXPECTED_DATA.items():
        actual = sha256(ROOT / relative)
        data[relative] = {"sha256": actual, "expected_sha256": expected, "match": actual == expected}

    reference = ROOT / "results/reference/quality_7sets_8methods_5seeds.csv"
    with reference.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    keys = [(r["dataset"], r["view"], r["method"], r["seed"]) for r in rows]
    reference_check = {
        "rows": len(rows),
        "unique_dataset_view_method_seed_keys": len(set(keys)),
        "complete_7x8x5_grid": len(rows) == 280 and len(set(keys)) == 280,
    }
    files = {relative: sha256(ROOT / relative) for relative in TRACKED_FILES}
    missing = [relative for relative in TRACKED_FILES if not (ROOT / relative).is_file()]
    manifest = {
        "format": 1,
        "frozen_source_hashes": frozen,
        "frozen_input_hashes": data,
        "reference_quality_table": reference_check,
        "release_file_hashes": files,
        "missing_files": missing,
        "status": "PASS" if not missing and all(x["match"] for x in [*frozen.values(), *data.values()]) and reference_check["complete_7x8x5_grid"] else "FAIL",
    }
    out = ROOT / "SOURCE_MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": out.name, "files_hashed": len(files), "reference_rows": len(rows)}, indent=2))
    if manifest["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
