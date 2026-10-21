# ALSS: reproducible bisecting k-means experiments

This repository is a self-contained release for reproducing the paper's
clustering methods and selected quality results. It contains the frozen serial
implementation, an independent candidate-evaluation parallel implementation,
small redistributable datasets, the exact MFeat view used in the paper, and the
archived per-seed quality table used for verification.

## Quick start (Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m experiments.prepare_bundled_data
python -m experiments.run_reproduction --datasets mfeat_view6 --methods ALSS --seeds 0 1 2 3 4
```

Run every method in the frozen comparison on MFeat view 6 and check the
reproduced metrics against the archived paper rows:

```powershell
python -m experiments.run_reproduction --datasets mfeat_view6 --methods all --seeds 0 1 2 3 4 --verify-paper-reference
python -m experiments.build_release_manifest
python -m experiments.package_release
```

The check compares ACC, NMI, ARI, and SSE for every matching dataset/view,
method, and seed. Results and a machine-readable run manifest are written to a
new directory under `results/reproduced/`. Runtime is recorded for inspection;
it is not used as a cross-machine reproduction criterion.
The final command verifies the frozen source/input hashes and reference-table
shape, then refreshes `SOURCE_MANIFEST.json`.
The last command creates a CRC-checked `ALSS_GitHub_Release.zip` one directory
above this folder, excluding local timestamped scratch runs and Python caches.

## Methods and frozen protocol

The release includes BKM-Largest, BKM-Inertia, BKM-MeanSSE, BKM-Savaresi-J,
BKM-Gain, ALSS, ALSS-BudgetAware, and Flat K-Means. The paper protocol uses
K-Means++ initialization, 10 restarts per binary fit, seeds 0–4, and one BLAS
thread. ALSS uses `lambda=1` and `top_r=3`. The runner fits a separate
`StandardScaler` to each input view unless an imported NPZ records another
preprocessing rule.

The serialized source files `src/bkm.py`, `src/split_rules.py`, and
`src/evaluation.py` are copied from the frozen experiment code and their SHA-256
hashes are recorded in `SOURCE_MANIFEST.json`. The scripts do not change the
clustering algorithms.

## Data included

* `iris` and `wine`: small UCI datasets, loaded from the pinned scikit-learn
  package version by the preparation script.
* `mfeat_view6`: the morphological descriptor view (`mfeat-mor`, 2,000 rows,
  6 features, 10 classes) extracted from the included UCI archive. The complete
  Multiple Features archive is retained in `data/upstream/` so the input can be
  independently rebuilt and checked by hash.
* `results/reference/quality_7sets_8methods_5seeds.csv`: frozen numerical
  results for all seven selected paper views, eight methods, and five seeds.
  This is a comparison record, not a substitute for the original inputs.

The UCI dataset pages identify Multiple Features, Iris, and Wine as CC BY 4.0;
see `DATASET_SOURCES.md` for attribution and source details. Other paper views
are not bundled because their redistribution terms were not confirmed. They can
be converted from a locally obtained MATLAB `.mat` file using
`experiments/prepare_mat_view.py`; the converter records the source filename
and hash and never downloads data.

Example for a local MATLAB multi-view file:

```powershell
python experiments/prepare_mat_view.py --source-mat D:\data\dataset.mat --dataset dataset_key --view-index 2 --preprocessing standard --output data/paper_views_local/dataset_view_02.npz
python -m experiments.run_reproduction --datasets data/paper_views_local/dataset_view_02.npz --methods ALSS --seeds 0 1 2 3 4
```

For an already preprocessed view, pass `--preprocessing none`. Select the same
view and preprocessing reported by the paper; do not combine results across
different views or transformations.

## Parallel implementation

`src/parallel.py` exposes `ParallelBKM` and `ParallelBKMSession`. To run the
correctness check (not a speed benchmark):

```powershell
python -m experiments.validate_parallel_equivalence
```

It compares labels, split trajectory, fit-call count, and final SSE against the
serial implementation for Gain, ALSS, and ALSS-BudgetAware at 1, 2, 4, and 8
workers. Spawned workers are guarded for Windows. Timing from this check should
not be interpreted as a controlled performance result.

Dataset attribution and source links are in `DATASET_SOURCES.md`.
