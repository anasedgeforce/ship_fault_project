# Live monitoring and dedicated MongoDB

The live service uses its own `ship_fault_monitoring` database. It creates exactly two collections on startup or with `python -m scripts.init_mongo`:

| Collection | Contents |
| --- | --- |
| `live_data` | Live six-channel readings at 4 kHz with their normal, fault, pending, invalid, or needs-review status. Window quality and model results are embedded in the same live-data document. |
| `live_faults` | One labeled result per valid prediction window: Normal predictions or non-normal fault candidates, with family, subtype, model scores, and review status. A persistent alert is marked only after three consecutive matching classified fault windows. |

Set `SHIP_FAULT_MONGO_URI` and `SHIP_FAULT_MONGO_DATABASE` to change the connection or database. `SHIP_FAULT_MONGO_ENABLED=0` disables storage for an offline test. The default database is separate from `admin.live_test` and `cbpm_database`.

## Send compatible readings

Each `POST /v1/streams/{stream_id}/samples` body has a nonempty `samples` array and may include `source` metadata. This application uses the fixed equipment ID `8c088678ea386840f36fd10c`; a different supplied ID is rejected.

Optional `source` metadata includes `cycle_key`, `live_label`, `record_type`, and `source_timestamp_ms`. These values are saved with the new live-data document; they are not used as fault features or as labels for future Mongo-only training. The service generates the new Mongo document ID and returns its database, collection, and ID with each prediction. Manual recording predictions are returned to the user and are never written to these collections.

For each accepted live batch, the service validates and writes the readings to
`live_data`, reads and verifies the saved timestamps and six binary channels,
then feeds those retrieved samples into the rolling prediction window. A window
that crosses batches is linked to, and updates, every batch that contributed
samples; history deduplicates the shared window. Each labeled prediction window
also creates a `live_faults` document with `condition_label`, `fault_family`,
`fault_subtype`, `decision`, and `review_reasons`. Normal predictions have
`event_type: normal_prediction`, `is_fault: false`, and `persistent_alert: false`.
Non-normal candidates have `event_type: fault_candidate` and `is_fault: true`;
`persistent_alert` identifies a sustained classified fault.
If MongoDB cannot save or read back a batch,
the service returns an error and does not run inference on that batch. A batch
saved before a readback failure remains in MongoDB as pending. Retrying the
same unchanged pending batch in the same stream session reuses it.

If inference fails after consuming part of a batch, the in-memory stream is
rolled back so the saved pending batch can be retried without duplicated samples.
If result or fault storage fails after successful inference, the response has
`persistence: result_save_failed`. The raw readings are already stored. Pending
results are retained in the stream and retried before accepting its next batch.
The page shows **Retry saving results**, which calls
`POST /v1/streams/{stream_id}/retry-storage`; this retries storage without
repeating inference. Reset and idle expiry cannot discard a stream's pending
result save. Polling the stream reports `persistence` and `pending_result_batch`.

New batches record their processing state and model/verification configuration
inside `live_data`. On startup the service replays each latest unclosed session
from MongoDB, rebuilding its rolling samples and completing unfinished predictions.
No extra collection is used. Window and fault writes use the original session and
window IDs, so replay does not create duplicate faults. A saved reset closes the
session; it will not be reopened after restart. Status includes `recovered_from_mongo`.

If MongoDB is unavailable during startup, watching the stream or sending its next
batch retries recovery. A model/configuration mismatch, damaged batch, or legacy
batch without recovery metadata blocks that stream with HTTP 503 instead of
silently mixing model versions; health reports the stream recovery error.
Finish/reset active sessions before changing models or equipment verification
settings. Recovery rebuilds the entire active session, so startup time grows with
session length. Run one API worker per database: distributed stream ownership and
checkpoint-based accelerated replay are not implemented. These changes recover
readings already saved to MongoDB; the acquisition sender must buffer readings
that were never acknowledged. After recovery, inspect the stream's last timestamp
and send only subsequent samples.

`start_timestamp_seconds` and `end_timestamp_seconds` preserve sensor time;
`recording_duration_seconds` reports the batch duration. `created_at` is the
server's ingestion time. `source_timestamp_ms` is an optional timestamp sent by
the source and is not automatically assumed to be the exact recording start.
The original recording-time audit is retained in the sibling project archive.

```json
{
  "source": {
    "equipment_id": "8c088678ea386840f36fd10c",
    "channel_names": ["DEV (mm/s)", "DE H (mm/s)", "DE A (mm/s)", "NDE V (mm/s)", "NDE H (mm/s)", "NDE A (mm/s)"],
    "channel_units": "mm/s",
    "sample_rate_hz": 4000,
    "record_type": "live_sensor"
  },
  "samples": [
    {"timestamp": 0.0, "channels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
  ]
}
```

The example shows the shape only. Send at least 4,000 consecutive samples for the first window. Sample timestamps use **seconds**; a Mongo `timestamp` stored as epoch milliseconds belongs in `source.source_timestamp_ms`, with per-sample times converted to seconds. For a compatible local CSV, run:

```powershell
python -m scripts.replay_csv_to_mongo "data/data_day1/Cocked Bearing with loader/1.csv" --stream-id test-motor
```

Do not use that training recording as an accuracy test. Replace it with new labeled recordings for validation.

The shown Fire Pump No 5 document in `admin.live_test` has 16 channels at 8 kHz. It cannot be fed directly to this six-channel, 4 kHz ship simulator model. A CSV header or declared `channel_names` checks order but cannot prove physical wiring, units, or whether the equipment resembles training conditions. The service rejects wrong channel count, nonfinite values, wrong rate, and timestamp gaps. Flat, duplicated, or possibly clipped channels block a fault prediction for that window. Clipping detection is only a signal heuristic; verify the sensor's actual full-scale range separately.

## Results and alerts

`GET /v1/streams/{stream_id}` returns recent window decisions and alerts. `GET /v1/streams/{stream_id}/diagnostics` returns bounded waveform, spectrum, and RMS trends. `GET /v1/streams/{stream_id}/history` reads window results embedded in `live_data`, plus separate `normal_predictions`, `fault_candidates`, and persistent `alerts` lists from `live_faults`. `GET /v1/storage` reports the database, live batch count, and separate normal and fault candidate counts. Every saved prediction event includes its source `live_data` document reference.

Window history is deduplicated by both `session_id` and `window_index`, so window
zero from a reset stream cannot overwrite window zero from its previous session.
When the main family prediction is Normal but the independent detailed model
suggests a fault, `live_faults` retains that independent candidate with
`candidate_source: independent_detailed_condition`, its subtype score, and
`needs_review`. The original family and detailed predictions are saved alongside
it. Its `family_score` is null because the top family score belongs to Normal.

`GET /health/ready` checks MongoDB with a fresh ping. It reports `degraded` and
`live_storage: unavailable` during a connection failure; manual inference can
still work. Connection attempts recover on later requests, including when MongoDB
was unavailable at startup. Credentials and connection strings are not returned.

To check this project's running service against its configured MongoDB, run:

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_live_connectivity
```

This writes UUID-tagged normal/fault test streams to the two existing collections,
checks raw readback, predictions, reset/history, timestamps and manual isolation,
then deletes only its own test records. Results are saved in
`reports/live_pipeline_review/connectivity.json`. This is a software connectivity
test using simulator CSVs, not a test of physical wiring or new-equipment accuracy.

The active models predict six family names and 33 detailed condition names
learned from the labeled simulator data. The displayed subtype is chosen within
the predicted family; disagreement from the independent subtype model is marked
for review.

`needs_review` retains the model's candidate family/subtype but does not issue a fault alert. It can be caused by low or inconsistent model scores, an outlier waveform, missing equipment validation, or operation beyond the short capture duration used in training. The IsolationForest outlier reference was fitted to existing training features; its unknown-fault detection rate is **not validated**. Model scores are also uncalibrated.

Only after checking the real six-axis wiring, units, acquisition rate, and a held-out recording set for an equipment ID should that ID be added to `SHIP_FAULT_VALIDATED_EQUIPMENT_IDS` (comma-separated). This enables classified results and persistent alerts within the model's validated short duration. Longer streams remain `needs_review` until a new continuous model is trained and validated.

## Continuous-stream validation

Prepare a CSV manifest with columns `path,fault_family,fault_subtype`. Each path points to a genuinely new labeled long run with the seven trained CSV columns. Run:

```powershell
python -m scripts.validate_continuous_stream --manifest path\to\long_runs.csv --output reports\continuous_validation.json
```

The report separates early and late window candidate accuracy and classified coverage. It does not establish unseen-equipment or unknown-fault accuracy by itself. The current project has no new labeled long runs, so continuous-operation accuracy is still unmeasured.

## Scope

The application provides current fault classification. Short-, medium-, and long-term forecasting sections have been removed from manual and live results.
