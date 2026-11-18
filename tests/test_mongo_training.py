"""Verified Mongo captures become labeled, idempotent training CSVs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest

from ship_fault.mongo_training import export_verified_batches
from ship_fault.settings import APP_EQUIPMENT_ID, CHANNELS
from src.labels import label_for


ROOT = Path(__file__).resolve().parents[1]
LABEL = 'outer_race__high_severity__no_ub_specified'


@pytest.fixture
def training_project():
    project = (ROOT / 'cache' / f'mt_{uuid.uuid4().hex[:8]}').resolve()
    assert project.is_relative_to((ROOT / 'cache').resolve())
    project.mkdir()
    yield project
    if project.is_relative_to((ROOT / 'cache').resolve()):
        shutil.rmtree(project)


def project_tree(project):
    (project / 'config').mkdir()
    (project / 'config/label_map.yaml').write_bytes((ROOT / 'config/label_map.yaml').read_bytes())
    for number in (1, 2, 3):
        (project / 'data' / f'data_day{number}').mkdir(parents=True)
    (project / 'reports').mkdir()
    return project


def capture_batches(label=LABEL, gap=False, csv_path=None):
    csv_path = csv_path or ROOT / 'data/data_day1/Outer Race Fault with loader/Outer Race Fault High Severity/1.csv'
    frame = pd.read_csv(csv_path, nrows=32_000)
    times = np.arange(32_000, dtype='<f8') / 4000
    if gap:
        times[16_000:] += 1
    values = frame[list(CHANNELS)].to_numpy(dtype='<f4')
    result = []
    for start in (0, 16_000):
        end = start + 16_000
        result.append({
            'record_type': 'verified_training', 'equipment_id': APP_EQUIPMENT_ID,
            'session_id': 'simulator-run-1', 'live_label': label,
            'created_at': datetime(2026, 11, 9, 4, tzinfo=timezone.utc),
            'fs': 4000, 'dtype': 'float32_le', 'total_samples': end - start,
            'model_input_order': list(CHANNELS), 'declared_channel_names': list(CHANNELS),
            'declared_channel_units': 'mm/s',
            'timestamps_bin': times[start:end].tobytes(),
            'channel_data_bin': {name: values[start:end, axis].tobytes()
                                 for axis, name in enumerate(CHANNELS)},
        })
    return result


def test_verified_capture_imports_once_and_preserves_label(training_project):
    root = project_tree(training_project)
    docs = capture_batches()
    first = export_verified_batches(root, docs)
    assert first['selected_batches'] == 2
    assert first['complete_captures'] == first['new_csvs'] == 1
    assert first['training_days'] == {'2026-11-09': 'data_day4'}
    path = root / 'data' / first['new_files'][0]
    assert label_for(path.parent.relative_to(root / 'data/data_day4').as_posix()) == LABEL
    imported = pd.read_csv(path)
    assert list(imported.columns) == ['Time', *CHANNELS]
    assert len(imported) == 32_000
    assert imported['Time'].iloc[-1] == pytest.approx(7.99975)
    assert imported[CHANNELS[0]].iloc[100] == pytest.approx(float(
        np.float32(pd.read_csv(ROOT / 'data/data_day1/Outer Race Fault with loader/Outer Race Fault High Severity/1.csv',
                               skiprows=range(1, 101), nrows=1)[CHANNELS[0]].iloc[0])), rel=1e-6)
    second = export_verified_batches(root, docs)
    assert second['complete_captures'] == 1
    assert second['new_csvs'] == 0


def test_unverified_or_incomplete_capture_is_not_training_data(training_project):
    root = project_tree(training_project)
    assert export_verified_batches(root, [{**doc, 'record_type': 'live_sensor'}
                                          for doc in capture_batches()])['new_csvs'] == 0
    assert export_verified_batches(root, capture_batches()[:1])['new_csvs'] == 0
    assert not list((root / 'data').glob('data_day4/**/*.csv'))
    with pytest.raises(ValueError, match='Unknown verified subtype'):
        export_verified_batches(root, capture_batches(label='model_guess'))
    assert not list((root / 'data').glob('data_day4/**/*.csv'))


def test_timestamp_gap_does_not_make_an_eight_second_recording(training_project):
    root = project_tree(training_project)
    report = export_verified_batches(root, capture_batches(gap=True))
    assert report['new_csvs'] == 0
    assert report['short_segments_skipped'] == 2


def test_reserved_sensor_replay_is_rejected(training_project):
    root = project_tree(training_project)
    reserved = json.loads((ROOT / 'data/unseen_test/source_manifest.json').read_text(encoding='utf-8'))
    # Copy the manifest and reserved files into this isolated test project.
    (root / 'data/unseen_test').mkdir(parents=True)
    (root / 'data/unseen_test/source_manifest.json').write_text(json.dumps(reserved), encoding='utf-8')
    row = reserved[0]
    original = ROOT / 'data/unseen_test' / row['file']
    target = root / 'data/unseen_test' / row['file']
    target.parent.mkdir(parents=True)
    target.write_bytes(original.read_bytes())
    # The guard requires every manifest file, so copy the remaining seven too.
    for row in reserved[1:]:
        original = ROOT / 'data/unseen_test' / row['file']
        target = root / 'data/unseen_test' / row['file']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match='Reserved unseen_test signal'):
        export_verified_batches(root, capture_batches(label='cocked_bearing__with_loader', csv_path=
            root / 'data/unseen_test' / reserved[0]['file']))
    assert not list((root / 'data').glob('data_day4/**/*.csv'))
