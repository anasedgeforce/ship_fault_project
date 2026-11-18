"""Exercise save -> readback -> prediction using the real Mongo serialization code."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo.errors import DuplicateKeyError

from ship_fault.api import create_app
from ship_fault.mongo_store import MongoStore
from ship_fault.settings import APP_EQUIPMENT_ID, CHANNELS, RuntimeSettings

EQUIPMENT_ID = APP_EQUIPMENT_ID


class MemoryCursor(list):
    def sort(self, key, direction=None):
        pairs = key if isinstance(key, list) else [(key, direction)]
        for name, order in reversed(pairs):
            super().sort(key=lambda item: item.get(name), reverse=order == -1)
        return self


class MemoryCollection:
    def __init__(self):
        self.documents = []

    def insert_one(self, document):
        if any(item.get('session_id') == document.get('session_id')
               and item.get('start_timestamp_seconds') == document.get('start_timestamp_seconds')
               for item in self.documents):
            raise DuplicateKeyError('duplicate batch')
        saved = deepcopy(document)
        saved['_id'] = ObjectId()
        self.documents.append(saved)
        return SimpleNamespace(inserted_id=saved['_id'])

    def find(self, query, projection=None):
        return MemoryCursor(deepcopy(item) for item in self.documents
                            if all(item.get(key) == value for key, value in query.items()))

    def find_one(self, query, sort=None):
        items = self.find(query)
        if sort:
            items.sort(sort)
        return next(iter(items), None)

    def distinct(self, key):
        return list({item[key] for item in self.documents})

    def update_one(self, query, update):
        for item in self.documents:
            if all(item.get(key) == value for key, value in query.items()):
                item.update(deepcopy(update['$set']))
                return SimpleNamespace(matched_count=1)
        return SimpleNamespace(matched_count=0)

    def replace_one(self, query, document, upsert=False):
        for index, item in enumerate(self.documents):
            if all(item.get(key) == value for key, value in query.items()):
                self.documents[index] = deepcopy(document)
                return
        if upsert:
            self.documents.append(deepcopy(document))


class MemoryMongoStore(MongoStore):
    def __init__(self):
        self.database = 'ship_fault_monitoring'
        self.db = {self.LIVE_COLLECTION: MemoryCollection(),
                   self.FAULT_COLLECTION: MemoryCollection()}
        self.events = []

    def store_batch(self, *args):
        self.events.append('save')
        return super().store_batch(*args)

    def load_batch(self, *args):
        self.events.append('read')
        return super().load_batch(*args)

    def store_results(self, *args):
        self.events.append('results')
        return super().store_results(*args)

    def close(self):
        pass

    def ping(self):
        pass


class ExampleEngine:
    family_names = ['normal', 'inner_race']

    def __init__(self, settings, fault=True):
        self.settings = settings
        self.fault = fault
        self.manifest = {'training_windows_per_file': 15}
        self.model_info = {'family_model_sha256': 'a' * 64}
        self.predicted_windows = []

    def predict_probabilities(self, windows, elapsed_starts):
        self.predicted_windows.append(windows.copy())
        return np.array([[0.1, 0.9] if self.fault else [0.9, 0.1]]), np.array([[1.0]])

    def format_predictions(self, family, detail):
        label = 'inner_race' if self.fault else 'normal'
        subtype = 'inner_race__low__loader' if self.fault else 'normal__loader'
        return [{'family': {'label': label, 'score': 0.9},
                 'detailed_condition': {'label': subtype, 'score': 1.0}}]

    def rank_family(self, probabilities):
        return {'label': 'inner_race' if self.fault else 'normal', 'score': 0.9}


def make_samples(count=8000):
    times = np.arange(count) / 4000
    values = np.column_stack([np.sin(2 * np.pi * (41 + 7 * axis) * times)
                              for axis in range(6)])
    return [{'timestamp': float(time), 'channels': row.tolist()}
            for time, row in zip(times, values)]


def source():
    return {'equipment_id': EQUIPMENT_ID, 'channel_names': list(CHANNELS),
            'channel_units': 'mm/s', 'source_timestamp_ms': 1782816727116}


def test_single_app_equipment_id_is_applied_and_other_ids_are_rejected(monkeypatch):
    store = MemoryMongoStore()
    settings = RuntimeSettings(mongo_enabled=True, equipment_id=APP_EQUIPMENT_ID)
    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _settings: ExampleEngine(settings))
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', lambda *_args: store)
    with TestClient(create_app(settings)) as client:
        assert client.get('/health/ready').json()['equipment_id'] == APP_EQUIPMENT_ID
        wrong = client.post('/v1/streams/fixed/samples', json={
            'source': {**source(), 'equipment_id': '6a40f65d977804cc99984993'},
            'samples': make_samples(1)})
        assert wrong.status_code == 422
        assert 'equipment_id must be' in wrong.json()['detail']
        assert not store.db['live_data'].documents
        accepted = client.post('/v1/streams/fixed/samples', json={'samples': make_samples(4000)})
        assert accepted.status_code == 200
        assert store.db['live_data'].documents[0]['equipment_id'] == APP_EQUIPMENT_ID
        assert all(event['equipment_id'] == APP_EQUIPMENT_ID
                   for event in store.db['live_faults'].documents)


@pytest.mark.parametrize('fault', [True, False])
def test_saved_samples_are_read_for_inference_and_labels_are_preserved(monkeypatch, fault):
    store = MemoryMongoStore()
    settings = RuntimeSettings(mongo_enabled=True, validated_equipment_ids=(EQUIPMENT_ID,))
    engine = ExampleEngine(settings, fault=fault)
    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _settings: engine)
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', lambda *_args: store)
    samples = make_samples()
    with TestClient(create_app(settings)) as client:
        first = client.post('/v1/streams/test_sensor/samples', json={'source': source(), 'samples': samples[:2000]})
        second = client.post('/v1/streams/test_sensor/samples', json={'source': source(), 'samples': samples[2000:]})
        assert first.status_code == second.status_code == 200
        assert store.events == ['save', 'read', 'results', 'save', 'read', 'results']
        assert len(second.json()['predictions']) == 3
        assert 'health_score' not in second.json()['predictions'][0]
        assert engine.predicted_windows[0][0, 0, 0] == pytest.approx(samples[0]['channels'][0])
        status = client.get('/v1/streams/test_sensor').json()
        assert status['recording_time']['duration_seconds'] == pytest.approx(2.0)
        assert status['recording_time']['source_timestamp_ms'] == 1782816727116

    batches = store.db['live_data'].documents
    assert len(batches) == 2
    assert batches[0]['condition_status'] == ('fault' if fault else 'normal')
    assert len(batches[0]['window_results']) == 1
    assert batches[1]['condition_status'] == ('fault' if fault else 'normal')
    assert 'health_score' not in batches[1]
    assert batches[1]['window_results'][0]['fault_type'] == ('inner_race' if fault else 'normal')
    assert len(batches[1]['channel_data_bin']) == 6
    events = store.db['live_faults'].documents
    assert len(events) == 3
    if fault:
        assert events[0]['event_type'] == 'fault_candidate'
        assert events[0]['is_fault']
        assert events[0]['fault_type'] == 'inner_race'
        assert events[0]['fault_subtype'] == 'inner_race__low__loader'
        assert 'health_score' not in events[0]
        assert events[0]['decision'] == 'classified'
        assert not events[0]['persistent_alert']
        assert events[-1]['persistent_alert']
        assert len(events[-1]['mongo_references']) == 1
    else:
        assert all(event['event_type'] == 'normal_prediction' for event in events)
        assert all(event['is_fault'] is False for event in events)
        assert all(event['fault_family'] == 'normal' for event in events)
        assert all(event['fault_subtype'] == 'normal__loader' for event in events)
        assert all(event['condition_label'] == 'normal__loader' for event in events)
        assert all(event['persistent_alert'] is False for event in events)


def test_unvalidated_fault_candidates_are_saved_as_needing_review(monkeypatch):
    store = MemoryMongoStore()
    settings = RuntimeSettings(mongo_enabled=True)
    engine = ExampleEngine(settings)
    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _settings: engine)
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', lambda *_args: store)
    with TestClient(create_app(settings)) as client:
        response = client.post('/v1/streams/test_sensor/samples',
                               json={'source': source(), 'samples': make_samples(4000)})
        assert response.status_code == 200
        assert response.json()['predictions'][0]['decision'] == 'needs_review'
        assert not response.json()['alerts']
    candidates = store.db['live_faults'].documents
    assert len(candidates) == 1
    assert candidates[0]['decision'] == 'needs_review'
    assert candidates[0]['fault_family'] == 'inner_race'
    assert candidates[0]['fault_subtype'] == 'inner_race__low__loader'
    assert not candidates[0]['persistent_alert']


def test_subtype_fault_candidate_is_kept_when_family_model_says_normal(monkeypatch):
    store = MemoryMongoStore()
    settings = RuntimeSettings(mongo_enabled=True, validated_equipment_ids=(EQUIPMENT_ID,))
    engine = ExampleEngine(settings, fault=False)
    engine.format_predictions = lambda family, detail: [{
        'family': {'label': 'normal', 'score': 0.9},
        'detailed_condition': {'label': 'inner_race__low__loader', 'score': 0.9},
    }]
    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _settings: engine)
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', lambda *_args: store)
    with TestClient(create_app(settings)) as client:
        response = client.post('/v1/streams/test_sensor/samples',
                               json={'source': source(), 'samples': make_samples(4000)})
        assert response.status_code == 200
    candidate = store.db['live_faults'].documents[0]
    assert candidate['fault_family'] == 'normal'
    assert candidate['fault_subtype'] == 'inner_race__low__loader'
    assert candidate['decision'] == 'needs_review'
    assert 'family_detail_disagree' in candidate['review_reasons']


def test_mongo_readback_rejects_damaged_channel_data():
    store = MemoryMongoStore()
    times = np.arange(4) / 4000
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    reference = store.store_batch('session', 'sensor', times, values, None)
    saved_times, saved_values = store.load_batch(reference, 'session', 'sensor')
    np.testing.assert_array_equal(saved_times, times)
    np.testing.assert_array_equal(saved_values, values)
    store.db['live_data'].documents[0]['channel_data_bin'][CHANNELS[0]] = b'broken'
    with pytest.raises(ValueError, match='damaged'):
        store.load_batch(reference, 'session', 'sensor')


def test_failed_readback_does_not_predict_and_identical_pending_batch_can_retry(monkeypatch):
    store = MemoryMongoStore()
    real_read = store.load_batch
    attempts = 0

    def fail_once(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError('read unavailable')
        return real_read(*args)

    store.load_batch = fail_once
    settings = RuntimeSettings(mongo_enabled=True, validated_equipment_ids=(EQUIPMENT_ID,))
    engine = ExampleEngine(settings)
    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _settings: engine)
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', lambda *_args: store)
    payload = {'source': source(), 'samples': make_samples(4000)}
    with TestClient(create_app(settings)) as client:
        failed = client.post('/v1/streams/test_sensor/samples', json=payload)
        assert failed.status_code == 503
        assert not engine.predicted_windows
        assert len(store.db['live_data'].documents) == 1
        assert store.db['live_data'].documents[0]['condition_status'] == 'pending'
        retry = client.post('/v1/streams/test_sensor/samples', json=payload)
        assert retry.status_code == 200
        assert len(retry.json()['predictions']) == 1
        assert len(store.db['live_data'].documents) == 1


def test_result_save_failure_is_retained_and_retried_without_new_samples(monkeypatch):
    store = MemoryMongoStore()
    settings = RuntimeSettings(mongo_enabled=True)
    engine = ExampleEngine(settings)
    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _: engine)
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', lambda *_: store)
    original = store.store_results
    attempts = 0

    def fail_twice(*args):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise ConnectionError('write interrupted')
        return original(*args)

    store.store_results = fail_twice
    samples = make_samples(6000)
    with TestClient(create_app(settings)) as client:
        first = client.post('/v1/streams/retry/samples', json={'source': source(), 'samples': samples[:4000]})
        assert first.status_code == 200 and first.json()['persistence'] == 'result_save_failed'
        assert client.get('/v1/streams/retry').json()['pending_result_batch']
        assert client.delete('/v1/streams/retry').status_code == 409
        blocked = client.post('/v1/streams/retry/samples', json={'source': source(), 'samples': samples[4000:]})
        assert blocked.status_code == 503
        assert client.get('/v1/streams/retry').json()['total_samples'] == 4000
        assert len(store.db['live_data'].documents) == 1
        assert client.post('/v1/streams/retry/retry-storage').json()['persistence'] == 'saved'
        assert client.get('/v1/streams/retry').json()['pending_result_batch'] is None
        final = client.post('/v1/streams/retry/samples', json={'source': source(), 'samples': samples[4000:]})
        assert final.status_code == 200 and final.json()['total_samples'] == 6000
        assert len(store.db['live_data'].documents) == 2
        assert len(store.db['live_faults'].documents) == 2


def test_inference_failure_rolls_back_partial_stream_consumption(monkeypatch):
    store = MemoryMongoStore()
    settings = RuntimeSettings(mongo_enabled=True)
    engine = ExampleEngine(settings)
    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _: engine)
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', lambda *_: store)
    predict = engine.predict_probabilities
    attempts = 0

    def fail_second(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError('inference interrupted')
        return predict(*args)

    engine.predict_probabilities = fail_second
    payload = {'source': source(), 'samples': make_samples(6000)}
    with TestClient(create_app(settings)) as client:
        with pytest.raises(RuntimeError, match='inference interrupted'):
            client.post('/v1/streams/rollback/samples', json=payload)
        status = client.get('/v1/streams/rollback').json()
        assert status['total_samples'] == 0 and status['predictions'] == []
        assert len(store.db['live_data'].documents) == 1
        retry = client.post('/v1/streams/rollback/samples', json=payload)
        assert retry.status_code == 200 and retry.json()['total_samples'] == 6000
        assert [p['window_index'] for p in retry.json()['predictions']] == [0, 1]
        assert len(store.db['live_data'].documents) == 1


def test_mongo_readiness_checks_current_connection_and_recovers(monkeypatch):
    store = MemoryMongoStore()
    settings = RuntimeSettings(mongo_enabled=True)
    engine = ExampleEngine(settings)
    attempts = 0

    def connect(*_):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError('startup outage')
        return store

    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _: engine)
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', connect)
    with TestClient(create_app(settings)) as client:
        assert client.get('/health/ready').json()['live_storage'] == 'ready'
        store.ping = lambda: (_ for _ in ()).throw(ConnectionError('connection lost'))
        ready = client.get('/health/ready').json()
        assert ready['status'] == 'degraded' and ready['live_storage'] == 'unavailable'
        response = client.post('/v1/streams/offline/samples', json={'source': source(), 'samples': make_samples(2)})
        assert response.status_code == 503 and not store.db['live_data'].documents
        store.ping = lambda: None
        assert client.get('/health/ready').json()['live_storage'] == 'ready'


def test_independent_fault_survives_normal_family_gating(monkeypatch):
    store = MemoryMongoStore()
    settings = RuntimeSettings(mongo_enabled=True)
    engine = ExampleEngine(settings, fault=False)
    engine.format_predictions = lambda *_: [{
        'family': {'label': 'normal', 'score': .8},
        'detailed_condition': {'label': 'normal__with_loader', 'score': .3},
        'independent_detailed_condition': {'label': 'inner_race__low_severity__no_ub_specified', 'score': .6}}]
    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _: engine)
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', lambda *_: store)
    with TestClient(create_app(settings)) as client:
        response = client.post('/v1/streams/disagree/samples', json={'source': source(), 'samples': make_samples(4000)})
        assert response.status_code == 200
    event = store.db['live_faults'].documents[0]
    assert event['fault_family'] == 'inner_race'
    assert event['candidate_source'] == 'independent_detailed_condition'
    assert event['decision'] == 'needs_review' and not event['persistent_alert']
    assert event['primary_family']['label'] == 'normal' and event['family_score'] is None

@pytest.mark.parametrize('failure', ['partial_window', 'inference', 'results'])
def test_restart_recovers_raw_samples_and_pending_results(monkeypatch, failure):
    store = MemoryMongoStore()
    settings = RuntimeSettings(mongo_enabled=True)
    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _: ExampleEngine(settings))
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', lambda *_: store)
    samples = make_samples(8000)
    with TestClient(create_app(settings)) as api:
        original_predict = api.app.state.engine.predict_probabilities
        original_results = store.store_results
        if failure == 'inference':
            api.app.state.engine.predict_probabilities = lambda *_: (_ for _ in ()).throw(RuntimeError('crash'))
        elif failure == 'results':
            def save_then_fail(*args):
                original_results(*args)
                raise ConnectionError('ack lost after writes')
            store.store_results = save_then_fail
        count = 2000 if failure == 'partial_window' else 6000
        payload = {'source': source(), 'samples': samples[:count]}
        if failure == 'inference':
            with pytest.raises(RuntimeError, match='crash'):
                api.post('/v1/streams/restart/samples', json=payload)
        else:
            response = api.post('/v1/streams/restart/samples', json=payload)
            assert response.status_code == 200
        session_id = store.db['live_data'].documents[0]['session_id']
        store.store_results = original_results
        api.app.state.engine.predict_probabilities = original_predict
    # New app and new engine simulate a fresh server process; only Mongo survives.
    with TestClient(create_app(settings)) as api:
        status = api.get('/v1/streams/restart').json()
        assert status['recovered_from_mongo']
        assert status['session_id'] == session_id
        assert status['total_samples'] == count
        response = api.post('/v1/streams/restart/samples', json={'source': source(), 'samples': samples[count:]})
        assert response.status_code == 200, response.text
        assert response.json()['total_samples'] == 8000
        faults = store.db['live_faults'].documents
        assert len(faults) == 3
        assert sorted(p['window_index'] for p in faults) == [0, 1, 2]
        assert len(store.db['live_data'].documents) == 2
        assert all(p['processing_state'] == 'processed' for p in store.db['live_data'].documents)
        assert api.delete('/v1/streams/restart').status_code == 200
    with TestClient(create_app(settings)) as api:
        assert api.get('/v1/streams/restart').status_code == 404
        response = api.post('/v1/streams/restart/samples', json={'source': source(), 'samples': samples[:4000]})
        assert response.status_code == 200
        assert response.json()['session_id'] != session_id


def test_restart_blocks_model_change_before_overwriting_results(monkeypatch):
    store = MemoryMongoStore()
    settings = RuntimeSettings(mongo_enabled=True)
    monkeypatch.setattr('ship_fault.api.InferenceEngine', lambda _: ExampleEngine(settings))
    monkeypatch.setattr('ship_fault.mongo_store.MongoStore', lambda *_: store)
    with TestClient(create_app(settings)) as api:
        assert api.post('/v1/streams/pinned/samples', json={'source': source(), 'samples': make_samples(4000)}).status_code == 200
    before = deepcopy(store.db['live_data'].documents)
    def changed_engine(_):
        engine = ExampleEngine(settings)
        engine.model_info['family_model_sha256'] = 'b' * 64
        return engine
    monkeypatch.setattr('ship_fault.api.InferenceEngine', changed_engine)
    with TestClient(create_app(settings)) as api:
        assert api.get('/health/ready').json()['status'] == 'degraded'
        assert api.get('/v1/streams/pinned').status_code == 503
        assert api.post('/v1/streams/pinned/samples', json={'source': source(), 'samples': make_samples(4000)}).status_code == 503
    assert store.db['live_data'].documents == before
