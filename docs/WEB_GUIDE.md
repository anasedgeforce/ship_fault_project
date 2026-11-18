# Local prediction page

Run `launch_project.bat` and open `http://127.0.0.1:8000/`. The page has two
prediction modes and no training button. A running server keeps using the model
it loaded at startup. Restart it after a future successful training run.

## Manual recording

Select a saved example, upload one CSV/TSV/JSON file, select a folder of CSVs,
or paste readings. The browser processes every selected CSV separately and
shows one aggregate fault-family and subtype result per recording. Files in
the same equipment subfolder can be grouped only when their classified
predictions agree. The optional verified-label JSON file measures accuracy
against labels the user supplies; without such labels, model scores are not
accuracy percentages.

Saved examples come from the previous simulator dataset and are demonstrations,
not new unseen tests. Uploads, pasted readings, supplied labels, and manual
predictions are processed in memory. The application does not automatically
save them in MongoDB, on the server filesystem, or in persistent browser
storage. A JSON file is saved locally only if the user clicks Download.

Input requires `Time` plus six velocity channels in this order:
`DEV (mm/s)`, `DE H (mm/s)`, `DE A (mm/s)`, `NDE V (mm/s)`,
`NDE H (mm/s)`, `NDE A (mm/s)`. Sampling must be 4 kHz, with at least
4,000 consecutive samples. Physical axis wiring must be checked separately.

## Digital twin highlights

The 3D view is an isolated copy of the supplied simulator HTML. The page sends
only exact model subtype IDs, preset IDs, and component names to it; labels are
never matched by display text. Inner and outer race faults highlight bearings,
cocked bearing highlights bearing housings, foundation looseness highlights the
rotor base, and unbalance highlights rotor discs. Combined bearing or foundation
conditions with unbalance highlight both regions. The view cannot identify the
exact physical bearing side. Camera controls remain available. Its animation
speed is illustrative, not a measured RPM. Simulated temperature, simulated
vibration readouts, manual fault selection, and thermal view are hidden.
The motor frame remains unmarked because none of the trained labels identifies
a motor-frame fault. The motor housing in the scene is not a fault location
inferred from these six vibration channels.
Small fault labels are linked by red leader lines to the highlighted component
regions and follow those locations as the camera rotates. Each region shows
one shortened fault name, a count when more faults affect that region, and a
review marker when needed. The list beside the twin retains the full subtype
name and one result per file. The bearing side remains unspecified, and normal
results show no labels.
The scene opens from a slightly elevated front angle and starts a slow camera
orbit automatically. This camera movement and the shaft animation are
illustrative; neither is a measured equipment RPM.

For a manual folder, the view combines regions from the separate CSV results
and lists each file's candidate. A single recording still has one primary
classification; mixed windows and uncertain results require review. In live
mode the animation keeps running while the latest non-normal candidate stays
red, including Review Required candidates. Three consecutive valid, classified
normal windows clear the red. Invalid or missing readings do not count as
normal. Changing or resetting a stream clears the previous result. This view
does not add any MongoDB writes or store manual predictions.
If the family model says Normal but the independent subtype model finds a known
fault, the twin shows that subtype as an independent candidate requiring review.
It does not count as a confirmed fault or as a normal window for clearing red.


## Live vibration

The sensor acquisition application sends batches to
`POST /v1/streams/{stream_id}/samples`. A batch contains a `samples` array of
objects with `timestamp` and six `channels` values. The first window needs
4,000 samples; subsequent windows complete every 2,000 samples. In the Live
Monitor tab, enter the same stream ID and click Watch Stream.

Live batches are saved to `ship_fault_monitoring.live_data`, read back for
inference, then updated with predictions. Labeled prediction events are also
saved in `live_faults`. See [the live MongoDB guide](LIVE_MONGO_GUIDE.md).

The current model learned from short recordings. Its accuracy during long
continuous operation has not yet been validated. The acquisition application
must handle sensor hardware protocols and send the documented API payload.

Future training uses the separate `ship_simulator_training.Sim_data_for_traning`
collection only when `train_from_mongo.bat` is run. See
[the MongoDB training guide](MONGO_RETRAINING.md).
