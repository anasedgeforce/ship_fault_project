"""Optional integration check against an explicitly supplied MongoDB URI."""
from __future__ import annotations

import os
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient

from ship_fault.api import create_app
from ship_fault.settings import APP_EQUIPMENT_ID, CHANNELS, RuntimeSettings


@pytest.mark.skipif(not os.environ.get('SHIP_FAULT_TEST_MONGO_URI'),
                    reason='set SHIP_FAULT_TEST_MONGO_URI to test a MongoDB server')
def test_live_mongo_roundtrip_with_real_model():
    uri = os.environ['SHIP_FAULT_TEST_MONGO_URI']
    name = f"ship_fault_test_{uuid.uuid4().hex}"
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command('ping')
    assert name not in client.list_database_names()
    settings = RuntimeSettings(mongo_enabled=True, mongo_uri=uri, mongo_database=name)
    frame = pd.read_csv('data/data_day1/Cocked Bearing with loader/1.csv', nrows=4000)
    samples = [{'timestamp': float(row[0]), 'channels': row[1:].tolist()}
               for row in frame.to_numpy()]
    source = {'equipment_id': APP_EQUIPMENT_ID,
              'channel_names': list(CHANNELS), 'channel_units': 'mm/s'}
    try:
        with TestClient(create_app(settings)) as api:
            first = api.post('/v1/streams/verify/samples', json={'source': source, 'samples': samples[:2000]})
            assert first.status_code == 200
            response = api.post('/v1/streams/verify/samples', json={'source': source, 'samples': samples[2000:]})
            assert response.status_code == 200, response.text
            result = response.json()
            assert result['persistence'] == 'saved'
            assert len(result['predictions']) == 1
            assert 'health_score' not in result['predictions'][0]
            batches = list(client[name]['live_data'].find({'stream_id': 'verify'}).sort('start_timestamp_seconds', 1))
            assert len(batches) == 2
            for saved in batches:
                assert saved['total_samples'] == 2000
                assert saved['recording_duration_seconds'] == pytest.approx(0.5)
                assert saved['window_results'][0]['family']['label'] == result['predictions'][0]['family']['label']
                assert 'health_score' not in saved
            history = api.get('/v1/streams/verify/history').json()
            assert len(history['predictions']) == 1
            prediction = result['predictions'][0]
            family = prediction['family']['label']
            subtype = prediction['detailed_condition']['label']
            if family != 'normal' or subtype.split('__', 1)[0] != 'normal':
                candidate = client[name]['live_faults'].find_one({'stream_id': 'verify'})
                assert candidate['fault_family'] == family
                assert candidate['fault_subtype'] == subtype
                assert candidate['decision'] == prediction['decision']
                assert not candidate['persistent_alert']
                assert history['fault_candidates'][0]['fault_family'] == family
            original_session = result['session_id']
            assert api.delete('/v1/streams/verify').status_code == 200
            restarted = api.post('/v1/streams/verify/samples', json={'source': source, 'samples': samples}).json()
            assert restarted['session_id'] != original_session
            history = api.get('/v1/streams/verify/history').json()
            assert len(history['predictions']) == 2
            assert {p['session_id'] for p in history['predictions']} == {original_session, restarted['session_id']}
            assert [p['window_index'] for p in history['predictions']] == [0, 0]
    finally:
        # This name is generated in this test and checked for absence before use.
        client.drop_database(name)
        client.close()

@pytest.mark.skipif(not os.environ.get('SHIP_FAULT_TEST_MONGO_URI'), reason='real MongoDB URI required')
def test_real_mongo_restart_recovers_pending_results_and_continues():
    uri = os.environ['SHIP_FAULT_TEST_MONGO_URI']
    name = f'ship_fault_test_{uuid.uuid4().hex}'
    mongo = MongoClient(uri, serverSelectionTimeoutMS=3000)
    assert name not in mongo.list_database_names()
    settings = RuntimeSettings(mongo_enabled=True, mongo_uri=uri, mongo_database=name)
    frame = pd.read_csv('data/data_day1/Cocked Bearing with loader/1.csv', nrows=8000)
    samples = [{'timestamp': float(row[0]), 'channels': row[1:].tolist()} for row in frame.to_numpy()]
    source = {'equipment_id': APP_EQUIPMENT_ID, 'channel_names': list(CHANNELS), 'channel_units': 'mm/s'}
    try:
        with TestClient(create_app(settings)) as api:
            assert api.post('/v1/streams/restart/samples', json={'source': source, 'samples': samples[:2000]}).status_code == 200
            original_save = api.app.state.mongo.store_results
            def interrupted_save(*args):
                original_save(*args)
                raise ConnectionError('simulated process interruption after database writes')
            api.app.state.mongo.store_results = interrupted_save
            response = api.post('/v1/streams/restart/samples', json={'source': source, 'samples': samples[2000:6000]})
            assert response.status_code == 200 and response.json()['persistence'] == 'result_save_failed'
            old_session = response.json()['session_id']
            expected = api.app.state.engine.predict_recording(frame.Time, frame.iloc[:, 1:])
        with TestClient(create_app(settings)) as api:
            status = api.get('/v1/streams/restart').json()
            assert status['session_id'] == old_session and status['recovered_from_mongo']
            assert status['total_samples'] == 6000 and status['persistence'] == 'up_to_date'
            response = api.post('/v1/streams/restart/samples', json={'source': source, 'samples': samples[6000:]})
            assert response.status_code == 200, response.text
            status = api.get('/v1/streams/restart').json()
            assert status['total_samples'] == 8000
            assert len(status['predictions']) == 3
            for actual, original in zip(status['predictions'], expected['predictions']):
                assert actual['family'] == original['family']
                assert actual['detailed_condition'] == original['detailed_condition']
            assert mongo[name].live_data.count_documents({}) == 3
            assert mongo[name].live_faults.count_documents({}) == 3
            assert len(api.get('/v1/streams/restart/history').json()['predictions']) == 3
            assert api.delete('/v1/streams/restart').status_code == 200
        with TestClient(create_app(settings)) as api:
            assert api.get('/v1/streams/restart').status_code == 404
    finally:
        mongo.drop_database(name)
        mongo.close()
