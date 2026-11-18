# Reserved eight-CSV test set

The eight CSVs are grouped by their verified **fault-family folder name**. Each file is a byte-for-byte copy of an original recording; `source_manifest.json` records its original path and SHA-256 of the six numeric vibration channels. `known_labels.json` contains the family and detailed subtype for the upload form.

Future training runs exclude these eight sensor signals by the `excluded_sensor_sha256` entries in `config/config.yaml`. This also excludes an identical copy of the Cocked Bearing recording still present under another Day 3 filename. Files placed in `data/unseen_test` are outside the training-day folders and are not scanned by the training data loader.

The active model was retrained after this reservation and excludes these eight sensor signals. This folder still comes from the same original three-day dataset, so its results do not establish accuracy on a new machine or recording day. The separate 603-CSV experiment model excluded these signals and scored 8/8 family and 8/8 subtype on them. Recheck that model with:

```powershell
.\.venv\Scripts\python.exe scripts/retest_unique_sensor_eight.py
```

The previous contents of `data/unseen_test` were archived under `reports/archives/unseen_test_before_20261103`.
