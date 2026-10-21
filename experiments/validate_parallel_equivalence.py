"""Correctness-only check that the parallel BKM preserves serial outputs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_reproduction import load_data
from src.bkm import BKM
from src.parallel import ParallelBKM, ParallelBKMSession


def main() -> None:
    parser_msg = "Use an if __name__ == '__main__' guard on Windows; the parallel session uses spawn."
    # A tiny benchmark keeps the release check quick while exercising a paper dataset.
    X, y, metadata, _ = load_data("mfeat_view6")
    k = int(np.unique(y).size)
    cases = [("gain", False), ("alss", False), ("alss", True)]
    seed, n_init = 17, 3
    report = {"dataset": metadata["display_name"], "n": int(X.shape[0]), "d": int(X.shape[1]), "k": k, "checks": []}

    for strategy, horizon_aware in cases:
        serial = BKM(
            n_clusters=k, strategy=strategy, lambda_=1.0, top_r=3,
            n_init=n_init, random_state=seed, horizon_aware=horizon_aware,
        ).fit(X)
        for workers in (1, 2, 4, 8):
            with ParallelBKMSession(X, n_jobs=workers) as session:
                parallel = ParallelBKM(
                    n_clusters=k, strategy=strategy, lambda_=1.0, top_r=3,
                    n_init=n_init, random_state=seed, horizon_aware=horizon_aware,
                    n_jobs=workers,
                ).fit(X, session=session)
            if not np.array_equal(serial.labels_, parallel.labels_):
                raise AssertionError(f"Labels differ for {strategy}, horizon={horizon_aware}, p={workers}")
            if serial.kmeans_fit_calls_ != parallel.kmeans_fit_calls_:
                raise AssertionError(f"Fit calls differ for {strategy}, horizon={horizon_aware}, p={workers}")
            if not np.isclose(serial.sse_, parallel.sse_, rtol=1e-10, atol=1e-9):
                raise AssertionError(f"SSE differs for {strategy}, horizon={horizon_aware}, p={workers}")
            serial_trajectory = json.dumps(serial.trajectory_, sort_keys=True, default=_json_default)
            parallel_trajectory = json.dumps(parallel.trajectory_, sort_keys=True, default=_json_default)
            if serial_trajectory != parallel_trajectory:
                raise AssertionError(f"Split trajectory differs for {strategy}, horizon={horizon_aware}, p={workers}")
            report["checks"].append({
                "strategy": strategy,
                "horizon_aware": horizon_aware,
                "workers": workers,
                "labels_exact": True,
                "fit_calls_equal": True,
                "sse_equal": True,
                "trajectory_equal": True,
                "fit_calls": int(serial.kmeans_fit_calls_),
                "sse": float(serial.sse_),
            })

    report["status"] = "PASS"
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    out = ROOT / "results" / "reproduced" / "parallel_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(parser_msg)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()

