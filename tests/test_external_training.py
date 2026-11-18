"""External training input remains separate from live prediction storage."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import pytest

from ship_fault.external_training import load_source_config, normalize_external_document
from ship_fault.mongo_only_training import check_source_coverage, prepare_workspace, publish_models
from ship_fault.settings import APP_EQUIPMENT_ID, CHANNELS


FIELDS = {
    'label': 'truth.subtype', 'session': 'session', 'created_at': 'date',
    'timestamps': 'sensor.times', 'channels': 'sensor.axes',
    'sample_rate': 'sensor.fs', 'units': 'sensor.units',
}


def sample_document():
    return {
        'session': 'external-run-1', 'date': datetime(2026, 11, 9, tzinfo=timezone.utc),
        'truth': {'subtype': 'outer_race__high_severity__no_ub_specified'},
        'sensor': {'fs': 4000, 'units': 'mm/s', 'times': [0, 0.00025],
                   'axes': {name: [index + 0.1, index + 0.2]
                            for index, name in enumerate(CHANNELS)}},
        'window_results': [{'detailed_condition': {'label': 'normal__with_loader'}}],
    }


def test_external_adapter_uses_verified_truth_and_all_six_axes():
    result = normalize_external_document(sample_document(), FIELDS)
    assert result['live_label'] == 'outer_race__high_severity__no_ub_specified'
    assert result['equipment_id'] == APP_EQUIPMENT_ID
    assert result['total_samples'] == 2
    assert np.frombuffer(result['channel_data_bin'][CHANNELS[5]], dtype='<f4')[1] == pytest.approx(5.2)
    assert 'window_results' not in result


def test_external_adapter_rejects_missing_sensor_and_wrong_rate():
    document = sample_document()
    del document['sensor']['axes'][CHANNELS[4]]
    with pytest.raises(ValueError, match='six named sensor axes'):
        normalize_external_document(document, FIELDS)
    document = sample_document()
    document['sensor']['fs'] = 8000
    with pytest.raises(ValueError, match='4 kHz'):
        normalize_external_document(document, FIELDS)


def test_source_configuration_needs_distinct_database(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.delenv('SHIP_FAULT_TRAINING_MONGO_DATABASE', raising=False)
    monkeypatch.delenv('SHIP_FAULT_TRAINING_MONGO_COLLECTION', raising=False)
    configured = load_source_config(project)
    assert configured['database'] == 'ship_simulator_training'
    assert configured['collection'] == 'Sim_data_for_traning'
    monkeypatch.setenv('SHIP_FAULT_TRAINING_MONGO_DATABASE', 'ship_fault_monitoring')
    monkeypatch.setenv('SHIP_FAULT_TRAINING_MONGO_COLLECTION', 'captures')
    with pytest.raises(ValueError, match='separate database'):
        load_source_config(project)
    monkeypatch.setenv('SHIP_FAULT_TRAINING_MONGO_DATABASE', 'new_simulator_training')
    config = load_source_config(project)
    assert config['database'] == 'new_simulator_training'
    assert config['collection'] == 'captures'


def test_each_mongo_training_run_gets_an_empty_data_workspace():
    project = Path(__file__).resolve().parents[1]
    first = prepare_workspace(project)
    second = prepare_workspace(project)
    try:
        assert first != second
        assert first.is_relative_to(project / 'mongo_runs')
        assert list(first.glob('data/**/*.csv')) == []
        assert list(second.glob('data/**/*.csv')) == []
        with pytest.raises(ValueError, match='all 33 known subtypes'):
            check_source_coverage(project, first)
    finally:
        for path in (first, second):
            if path.resolve().is_relative_to((project / 'mongo_runs').resolve()):
                shutil.rmtree(path)


def test_model_publication_is_isolated_and_keeps_old_pointer_on_failure(tmp_path):
    import json
    from ship_fault.models import InferenceEngine
    from ship_fault.settings import RuntimeSettings

    source = Path(__file__).resolve().parents[1]
    old_run = json.loads((source / 'config/current_model_baseline.json').read_text())['run_id']
    project, workspace = tmp_path / 'project', tmp_path / 'new_mongo_run'
    (project / 'config').mkdir(parents=True)
    (project / 'reports/duplicate_inclusive').mkdir(parents=True)
    (project / 'artifacts/runs').mkdir(parents=True)
    (workspace / 'reports/duplicate_inclusive').mkdir(parents=True)
    (workspace / 'artifacts/runs').mkdir(parents=True)
    shutil.copy2(source / 'config/label_map.yaml', project / 'config/label_map.yaml')
    shutil.copytree(source / 'artifacts/runs' / old_run, workspace / 'artifacts/runs' / old_run)
    shutil.copy2(source / 'artifacts/runs' / old_run / 'manifest.json',
                 workspace / 'reports/duplicate_inclusive/manifest.json')
    pointer = project / 'reports/duplicate_inclusive/manifest.json'
    pointer.write_text('{"run_id":"previous"}', encoding='utf-8')
    (workspace / 'artifacts').mkdir(exist_ok=True)
    shutil.copy2(source / 'artifacts/novelty_reference.joblib',
                 workspace / 'artifacts/novelty_reference.joblib')

    with pytest.raises(ValueError, match='published together'):
        publish_models(project, workspace)
    assert json.loads(pointer.read_text())['run_id'] == 'previous'
    assert not (project / 'artifacts/runs' / old_run).exists()

    shutil.copy2(source / 'reports/novelty_reference.json',
                 workspace / 'reports/novelty_reference.json')
    assert publish_models(project, workspace) == old_run
    assert json.loads(pointer.read_text())['run_id'] == old_run
    engine = InferenceEngine(RuntimeSettings(project_root=project))
    assert engine.novelty_status == 'ready'


def test_empty_external_collection_does_not_change_active_model(monkeypatch):
    import hashlib
    import sys
    import scripts.train_from_mongo as trainer

    pointer = Path(__file__).resolve().parents[1] / 'reports/duplicate_inclusive/manifest.json'
    before = hashlib.sha256(pointer.read_bytes()).hexdigest()

    class EmptyCollection:
        def count_documents(self, query):
            return 0

    class Database:
        def list_collection_names(self):
            return ['Sim_data_for_traning']

        def __getitem__(self, name):
            return EmptyCollection()

    class Client:
        admin = None

        def __init__(self, *args, **kwargs):
            self.admin = self

        def command(self, name):
            return {'ok': 1}

        def __getitem__(self, name):
            return Database()

        def close(self):
            pass

    monkeypatch.setattr(trainer, 'MongoClient', Client)
    monkeypatch.setattr(sys, 'argv', ['train_from_mongo.py'])
    assert trainer.main() == 1
    assert hashlib.sha256(pointer.read_bytes()).hexdigest() == before
