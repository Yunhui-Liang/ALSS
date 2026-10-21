# Dataset sources and attribution

| Dataset | Included representation | Source and attribution | License stated by source |
|---|---|---|---|
| Multiple Features, view 6 (`mfeat-mor`) | Included as the upstream ZIP; preparation script extracts the 2,000 × 6 morphological view and assigns the dataset's consecutive 200-example class labels | UCI Machine Learning Repository, Multiple Features, DOI [10.24432/C5HC70](https://doi.org/10.24432/C5HC70), [dataset page](https://archive.ics.uci.edu/dataset/72/multiple+features) | CC BY 4.0 |
| Iris | Rebuilt from scikit-learn 1.3.0's bundled copy of the UCI dataset | UCI Machine Learning Repository, Iris, DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76), [dataset page](https://archive.ics.uci.edu/dataset/53/iris) | CC BY 4.0 |
| Wine | Rebuilt from scikit-learn 1.3.0's bundled copy of the UCI dataset | UCI Machine Learning Repository, Wine, DOI [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J), [dataset page](https://archive.ics.uci.edu/dataset/109/wine) | CC BY 4.0 |

The preparation script writes `data/bundled_manifest.json`, including shapes,
array hashes, archive hash, source DOI, and license metadata. The bundled UCI
archive is retained so the exact MFeat input can be regenerated without a
network connection.

To evaluate another paper view, use the local MAT converter with its `.mat`
file and preserve the recorded hash and view index.
