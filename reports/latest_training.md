# Latest training results

Run: `20261104T041252Z_164da514`
Raw CSVs: 670; valid paths: 661; distinct recordings: 603; exact copy paths: 58; invalid exclusions: 0.

| task      | day       | accuracy   | recording_accuracy   | macro_f1   |
|:----------|:----------|:-----------|:---------------------|:-----------|
| 6-family  | data_day1 | 85.47%     | 85.64%               | 82.71%     |
| 33-detail | data_day1 | 68.34%     | 68.21%               | 49.52%     |
| 6-family  | data_day2 | 60.42%     | 61.46%               | 49.81%     |
| 33-detail | data_day2 | 37.22%     | 37.50%               | 14.89%     |
| 6-family  | data_day3 | 91.10%     | 93.19%               | 88.70%     |
| 33-detail | data_day3 | 62.23%     | 63.87%               | 44.07%     |

Equal-day means:
| task      | accuracy   | macro_f1   | recording_accuracy   |
|:----------|:-----------|:-----------|:---------------------|
| 33-detail | 55.93%     | 36.16%     | 56.53%               |
| 6-family  | 79.00%     | 73.74%     | 80.10%               |

Both model families use up to 15 evenly spaced windows per file in training; the detailed Random Forest uses 180 trees in both evaluation and the final fit.
Model recipes: family `baseline`, detailed `rf_relative_amp`.
Validation uses all complete windows of unique held-out recordings. The original locked holdout and matching raw hashes are excluded from every fold training set.
Detailed predictions are chosen from the predicted family. The independent detailed-model candidate is retained to flag disagreement for review.

Artifacts: `artifacts/runs/20261104T041252Z_164da514/`.
These are retrospective day-held-out results; final models are then fitted on all valid recordings.
