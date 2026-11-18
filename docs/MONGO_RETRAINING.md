# Retrain from a separate MongoDB database

The simulator data collection application writes labeled vibration captures to
its own MongoDB database. The training launcher **reads** that collection;
it does not write training samples to it. The prediction application uses a separate database
(`ship_fault_monitoring` by default) for `live_data` and `live_faults`.
Predictions stored in those collections are not automatically treated as true
training labels.

## Configure the source once

The dedicated local source is `ship_simulator_training.Sim_data_for_traning`.
It is created by `scripts/init_training_mongo.py`. Edit
[`config/external_mongo_training.yaml`](../config/external_mongo_training.yaml)
if the other application uses different database or field names. The training
database must differ from the prediction database.
If the source uses another MongoDB server, set
`SHIP_FAULT_TRAINING_MONGO_URI` in the environment. Do not put credentials in
the YAML file. The prediction database still uses `SHIP_FAULT_MONGO_URI` and
`SHIP_FAULT_MONGO_DATABASE`.

The current adapter accepts documents with these values (the field names are
configurable):

```json
{
  "session_id": "simulator-run-001",
  "created_at": "2026-11-09T10:30:00Z",
  "verified_fault_subtype": "outer_race__high_severity__no_ub_specified",
  "fs": 4000,
  "channel_units": "mm/s",
  "timestamps": [0.0, 0.00025],
  "channels": {
    "DEV (mm/s)": [0.1, 0.2],
    "DE H (mm/s)": [0.1, 0.2],
    "DE A (mm/s)": [0.1, 0.2],
    "NDE V (mm/s)": [0.1, 0.2],
    "NDE H (mm/s)": [0.1, 0.2],
    "NDE A (mm/s)": [0.1, 0.2]
  }
}
```

The short arrays illustrate the schema; a complete training capture needs
32,000 consecutive samples (8 seconds). Multiple consecutive documents can
form one capture if their session ID, verified subtype, and capture date match.
Use a new session ID when the simulator condition changes. The subtype must
be one of the 33 IDs from `GET /v1/labels` or the label catalog. The label must
come from the simulator test setup, **not a model prediction**. The six named
channels must reflect the actual sensor axis mapping; the program cannot
verify physical wiring from a MongoDB field name.

If the external application stores binary samples, different fields, another
sample rate, or another unit, its adapter needs to be mapped or extended after
we inspect a redacted example document. Filling in the database name alone
does not make an unknown schema compatible.

## Run training

Close the prediction server, then double-click [`train_from_mongo.bat`](../train_from_mongo.bat).
This is the user-facing training command.

The command validates the source labels, six channels, sample spacing, and
signal quality. It exports complete captures into a new `mongo_runs/<run>/data/data_dayN`
workspace and runs `run_train.py` using **only that workspace**. The previous `data/data_dayN`
CSVs and their feature cache are not training inputs. Repeated imports of
identical captures do not create duplicate CSVs. The reserved `data/unseen_test`
signals are rejected. The importer stops without starting training when there
is no complete compatible capture.

The new source must contain all 33 known subtypes, including normal, and
recordings from at least two capture days before the existing six-family,
33-subtype model can be replaced. Without examples of a subtype, a new model
cannot learn to predict it. Day-held-out evaluation is run only on the new
source. The current model remains active if this validation or training fails.

The import inventory is in the run workspace's `reports/mongo_training_import.json`;
successful training results are copied to `reports/latest_training.md`.
Restart `launch_project.bat` after
training so the prediction server loads the new saved models. Subsequent live
predictions are saved with their predicted family and subtype in the
prediction database's `live_data` and `live_faults` collections.
