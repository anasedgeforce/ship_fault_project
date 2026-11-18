"""The localhost page uses saved models without exposing training actions."""
from __future__ import annotations

import json
import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ship_fault.catalog as catalog_module
from ship_fault.api import create_app
from ship_fault.settings import RuntimeSettings


def test_page_catalog_and_saved_prediction():
    with TestClient(create_app(RuntimeSettings())) as client:
        page = client.get('/')
        assert page.status_code == 200
        assert 'Saved Recording' in page.text
        assert 'Load Smooth Sample' in page.text
        assert 'Folder of CSV Files' in page.text
        assert 'Uploaded Dataset Summary' in page.text
        assert 'forecast-panel' not in page.text
        assert 'Fault Subtype' in page.text
        assert '<th>Assessment</th>' not in page.text
        assert 'id="upload-family-accuracy"' not in page.text
        assert 'id="upload-subtype-accuracy"' not in page.text
        assert 'Health indicator' not in page.text
        assert 'Train both models' not in page.text
        assert client.get('/assets/app.css').status_code == 200
        assert client.get('/assets/app.js').status_code == 200
        assert client.get('/assets/twin-map.js').status_code == 200
        assert client.get('/assets/twin-view.js').status_code == 200
        twin = client.get('/assets/digital-twin.html')
        assert twin.status_code == 200
        assert 'simulator-twin/ready' in twin.text
        assert 'simulator-twin/update' in twin.text
        assert 'id="digital-twin-frame"' in page.text
        assert 'sandbox="allow-scripts"' in page.text

        catalog = client.get('/v1/recordings')
        assert catalog.status_code == 200
        days = catalog.json()['days']
        displayed = {recording['id'] for day in days for condition in day['conditions']
                     for recording in condition['recordings']}
        manifest = json.loads(Path('reports/duplicate_inclusive/manifest.json').read_text(encoding='utf-8'))
        run_dir = Path(manifest['family_artifact'].replace('\\', '/')).parent
        index = run_dir / 'validation_files.csv'
        if not index.is_file():
            index = Path('evaluation_data/validation_files.csv')
        with index.open(encoding='utf-8', newline='') as stream:
            records = list(csv.DictReader(stream))
        root = catalog_module.configured_data_root(Path('.').resolve())
        eligible = []
        for row in records:
            path = root / row['file']
            if not path.is_file():
                continue
            stat = path.stat()
            if catalog_module._matches_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns, row['sha256']):
                eligible.append(row)
        expected = {row['file'] for row in eligible}
        assert {day['name'] for day in days} == {row['day'] for row in eligible}
        assert displayed == expected
        assert displayed
        selected = next(iter(sorted(displayed)))
        result = client.post('/v1/predict/saved', json={'id': selected})
        assert result.status_code == 200
        assert result.json()['window_count'] > 0
        assert result.json()['recording_family']['label']
        assert client.post('/v1/predict/saved', json={'id': '../config/config.yaml'}).status_code == 404
        locked = Path('reports/holdout_files.txt').read_text(encoding='utf-8').splitlines()[0]
        assert locked not in displayed
        assert client.post('/v1/predict/saved', json={'id': locked}).status_code == 404


def test_synthetic_sample_button_source_is_complete_and_outside_training_data():
    with TestClient(create_app(RuntimeSettings())) as client:
        sample = client.get('/v1/demo-inputs/synthetic_unseen_smooth.csv')
        assert sample.status_code == 200
        assert len(sample.text.splitlines()) == 8001
        assert sample.text.startswith('Time,DEV (mm/s),')
        prediction = client.post('/v1/predict/input', content=sample.content)
        assert prediction.status_code == 200
        assert prediction.json()['window_count'] == 3
        assert client.get('/v1/demo-inputs/config.yaml').status_code == 404


def test_evaluation_list_rejects_other_model_runs_and_changed_files(monkeypatch):
    root = Path('.').resolve()
    manifest = json.loads((root / 'reports/duplicate_inclusive/manifest.json').read_text(encoding='utf-8'))
    with pytest.raises(ValueError, match='does not match the loaded model run'):
        catalog_module.public_catalog(root, {**manifest, 'run_id': 'different-run'})
    _, original = catalog_module.public_catalog(root, manifest)
    key = sorted(original)[0]
    target = str(original[key][0])
    original_check = catalog_module._matches_hash
    monkeypatch.setattr(catalog_module, '_matches_hash',
                        lambda path, *args: False if path == target else original_check(path, *args))
    _, filtered = catalog_module.public_catalog(root, manifest)
    assert key not in filtered
    assert len(filtered) == len(original) - 1


def test_uploaded_and_pasted_readings_share_prediction_path():
    settings = RuntimeSettings()
    source = Path('data/data_day1/Cocked Bearing with loader/1.csv')
    header, *rows = source.read_text(encoding='utf-8-sig').splitlines()
    csv = '\n'.join([header, *rows[:4000], '']).encode()
    with TestClient(create_app(settings)) as client:
        uploaded = client.post('/v1/predict/input', content=csv)
        assert uploaded.status_code == 200
        from_csv = uploaded.json()
        assert from_csv['window_count'] == 1
        assert from_csv['valid_window_count'] == 1
        assert from_csv['recording_duration_seconds'] == 1.0
        assert from_csv['recording_detailed_condition']['label']
        assert 'health_score' not in from_csv
        assert from_csv['recording_decision'] in {'classified', 'needs_review'}
        assert abs(sum(from_csv['family_probability_mean'].values()) - 1) < 1e-6
        assert abs(sum(from_csv['detailed_probability_mean'].values()) - 1) < 1e-6
        values = [row.split(',') for row in rows[:4000]]
        payload = {'timestamps': [float(row[0]) for row in values],
                   'channels': [[float(value) for value in row[1:]] for row in values]}
        pasted = client.post('/v1/predict/input', content=json.dumps(payload))
        assert pasted.status_code == 200
        assert pasted.json()['recording_family']['label'] == from_csv['recording_family']['label']
        assert client.post('/v1/predict/input', content='incomplete').status_code == 422


def test_manual_prediction_does_not_write_to_mongo(monkeypatch):
    import ship_fault.mongo_store as mongo_module

    writes = []

    class TrackingStore:
        def __init__(self, *args):
            pass

        def ping(self):
            pass

        def active_stream_ids(self):
            return []

        def close(self):
            pass

        def store_batch(self, *args):
            writes.append('batch')

        def store_results(self, *args):
            writes.append('results')

    monkeypatch.setattr(mongo_module, 'MongoStore', TrackingStore)
    source = Path('data/data_day1/Cocked Bearing with loader/1.csv')
    raw = '\n'.join(source.read_text(encoding='utf-8-sig').splitlines()[:4001]).encode()
    with TestClient(create_app(RuntimeSettings(mongo_enabled=True))) as client:
        assert client.post('/v1/predict/input', content=raw).status_code == 200
        saved = client.get('/v1/recordings').json()['days'][0]['conditions'][0]['recordings'][0]['id']
        assert client.post('/v1/predict/saved', json={'id': saved}).status_code == 200
    assert writes == []
    script = Path('ship_fault/web/app.js').read_text(encoding='utf-8')
    assert all(name not in script for name in ('localStorage', 'sessionStorage', 'indexedDB'))
    assert "$('download').onclick" in script
