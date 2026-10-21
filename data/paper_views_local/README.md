# Locally supplied paper views

To add a local `.mat` dataset, convert a single feature view with
`experiments/prepare_mat_view.py`.
That script records the source file hash, selected 1-based view, dimensions,
and preprocessing. Keep each view in its own NPZ and do not merge results from
different views or preprocessing pipelines.
