# Reproduction and integrity report

Validation completed on 2026-10-21 with Python 3.11.5, NumPy 1.24.3, SciPy
1.11.1, scikit-learn 1.3.0, and threadpoolctl 2.2.0. These match the archived
experiment environment. The released serial core files also match the frozen
experiment SHA-256 values; see `SOURCE_MANIFEST.json`.

## Exact paper-row reproduction

The bundled MFeat morphological view (`mfeat-mor`, 2,000 samples, 6 features,
10 classes) was run using seeds 0–4, K-Means++ initialization, 10 restarts per
binary fit, `lambda=1`, `top_r=3`, and one numerical thread. All eight methods
completed, giving 40 dataset–method–seed rows. ACC, NMI, ARI, and SSE were
compared row by row against the matching frozen paper records. All 160 metric
comparisons passed with maximum absolute difference 0. The detailed CSV,
reference comparison, and run metadata are in
`results/reproduced/reference_validation/`.

## Additional execution checks

ALSS completed five seeds each on bundled Iris and Wine. Those runs validate the
data preparation and command-line path; they are not compared to archived paper
rows. The outputs and metadata are in `results/reproduced/iris_wine_smoke/`.

The parallel implementation passed 12 correctness comparisons on MFeat: Gain,
ALSS, and ALSS-BudgetAware at 1, 2, 4, and 8 workers. Labels and split
trajectories matched exactly; fit-call counts and final SSE matched. This is a
correctness check, not a speedup claim. See
`results/reproduced/reference_validation/parallel_equivalence.json`.

## Scope

This package reproduces all eight methods on the bundled MFeat paper view and
provides the frozen numeric table for the seven selected views.
