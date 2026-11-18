# Simulator Fault Prediction

This local FastAPI application predicts a fault family and a detailed subtype
from six velocity channels sampled at 4 kHz. Start it with
[`launch_project.bat`](launch_project.bat) and open `http://127.0.0.1:8000/`.
Training is a separate action; opening the app does not retrain the models.

## Current model

The active model is run `20261104T041252Z_164da514`, trained on the previous
simulator CSV dataset. It recognizes six condition families, including Normal,
and 33 subtypes. Its earlier day-held-out mean recording accuracy was 80.10%
for families and 56.53% for subtypes. These are retrospective results, not
accuracy for a newly uploaded or live recording. The active model has no expiry
date and remains in use until a new model successfully publishes and the server
is restarted. The original model hashes are in
[`config/current_model_baseline.json`](config/current_model_baseline.json).

## Two prediction paths

- **Manual:** choose a saved example, upload a CSV/TSV/JSON file or a folder of
  CSVs, or paste readings. Each CSV receives a separate recording result.
  Manually checked readings, labels, and results stay in browser/server memory
  for the request and are not written to MongoDB or server files. Download
  creates a local JSON file only when the user clicks it. The Saved Recording
  choices are examples from the previous dataset; they are not an unseen test
  of either the current or a future model.
- **Live:** a separate sensor acquisition application sends timestamped
  six-channel batches to `POST /v1/streams/{stream_id}/samples`. The server
  stores each accepted batch in `ship_fault_monitoring.live_data`, reads it
  back for prediction, and saves predicted family, subtype, and review status
  to `live_data` and `live_faults`. The API does not decode sensor hardware
  packets. See [the live guide](docs/LIVE_MONGO_GUIDE.md).

The required channel order is `DEV (mm/s)`, `DE H (mm/s)`, `DE A (mm/s)`,
`NDE V (mm/s)`, `NDE H (mm/s)`, `NDE A (mm/s)`. At least 4,000 consecutive
samples are needed for one prediction window. The app cannot verify physical
sensor wiring from a CSV header. See [the web guide](docs/WEB_GUIDE.md) and
[label catalog](docs/LABEL_CATALOG.md).

The embedded 3D simulator highlights components for manual and live fault
candidates. It uses only vibration predictions. Its animation is illustrative,
and it does not show or use simulated temperature readings. The exact bearing
side cannot be inferred from a family/subtype label alone. See the
[web guide](docs/WEB_GUIDE.md) for the highlight and review rules.

## Future training, only when new data is ready

The training source is the separate collection
`ship_simulator_training.Sim_data_for_traning`. Configure the real source field
mapping in [`config/external_mongo_training.yaml`](config/external_mongo_training.yaml),
then run [`train_from_mongo.bat`](train_from_mongo.bat). This creates a fresh
workspace under `mongo_runs/` and uses **only that collection's labeled
captures**. The previous CSV dataset is never added to this run. The training
command requires all 33 known subtypes and at least two capture days; otherwise
it keeps the active model unchanged. A successful run verifies and publishes
both new models, then the prediction server must be restarted. See the
[MongoDB training guide](docs/MONGO_RETRAINING.md). The collection is currently
empty, so no new training has been performed.

`run_train.py` is an internal step of the Mongo training launcher and rejects
direct training in the main project. The old CSV training launchers have been
retired. Saved model pipelines still require their original Python module
paths, so some older transformer modules remain in `src/`.

## Setup and checks

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest httpx
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q tests
```

The current model's raw training data and archived research are separate from
future MongoDB training. The sibling `ship_fault_project_archive` folder holds
older backups and research material removed from the working application.
