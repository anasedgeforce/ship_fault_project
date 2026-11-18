"""Verify the running live API and configured MongoDB using tagged CSV replays.

Writes only tagged test streams to the two existing live collections, verifies
their readings/results, then removes only records belonging to this audit.
Does not measure new-equipment accuracy or operate a physical sensor.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import uuid

import numpy as np
import pandas as pd
from pymongo import MongoClient

from ship_fault.settings import APP_EQUIPMENT_ID, CHANNELS, RuntimeSettings


def run(base_url='http://127.0.0.1:8000'):
    settings = RuntimeSettings.from_env()
    root = settings.project_root
    tag = 'pipeline_audit_' + uuid.uuid4().hex[:16]
    streams = [tag + '_normal', tag + '_fault']
    report = dict(audit_id=tag, started_at_utc=datetime.now(timezone.utc).isoformat(),
                  database=settings.mongo_database, checks={}, streams=[])
    headers = {'Content-Type': 'application/json'}
    if settings.api_key:
        headers['X-API-Key'] = settings.api_key

    def request(path, body=None, method='GET', content_type='application/json', expected=200):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
        req = Request(base_url.rstrip('/') + path, data=payload, method=method,
                      headers={**headers, 'Content-Type': content_type})
        try:
            with urlopen(req, timeout=30) as response:
                status, data = response.status, json.load(response)
        except HTTPError as exc:
            status, data = exc.code, json.load(exc)
        if status != expected:
            raise AssertionError(f'{path}: expected HTTP {expected}, got {status}: {data}')
        return data

    mongo = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=3000, socketTimeoutMS=5000)
    db = mongo[settings.mongo_database]
    try:
        assert mongo.admin.command('ping')['ok'] == 1
        storage = request('/v1/storage')
        assert storage['status'] == 'ready' and storage['database'] == settings.mongo_database
        assert sorted(storage['collections']) == ['live_data', 'live_faults']
        assert not db.live_data.count_documents({'stream_id': {'$in': streams}})
        before_collections = sorted(db.list_collection_names())
        report['checks']['mongo_ping_and_api_connection'] = True
        ready = request('/health/ready')
        assert ready['live_storage'] == 'ready'
        report['models'] = {key: ready['models'][key] for key in ('family_model_sha256', 'detail_model_sha256')}
        manifest = json.loads((root / 'reports/duplicate_inclusive/manifest.json').read_text())
        assert ready['models']['detail_model_sha256'] == manifest['detail_sha256']
        index = pd.read_csv((root / manifest['family_artifact']).parent / 'validation_files.csv')
        for stream, label in zip(streams, ['normal__with_loader', 'cocked_bearing__with_loader']):
            source_path = index.loc[index.label.eq(label)].sort_values('file').iloc[0]['file']
            frame = pd.read_csv(root / 'data' / source_path, nrows=8000)
            times = frame.iloc[:, 0].to_numpy()
            values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
            samples = [{'timestamp': float(t), 'channels': v.tolist()} for t, v in zip(times, values)]
            source = dict(equipment_id=APP_EQUIPMENT_ID, channel_names=list(CHANNELS), channel_units='mm/s',
                          sample_rate_hz=4000, record_type='pipeline_connectivity_check', live_label=tag,
                          source_timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000))
            counts_before = [db[name].count_documents({}) for name in ('live_data', 'live_faults')]
            manual = request('/v1/predict/csv', frame.to_csv(index=False).encode(), 'POST', 'text/csv')
            assert counts_before == [db[name].count_documents({}) for name in ('live_data', 'live_faults')]
            responses = [request(f'/v1/streams/{stream}/samples',
                         {'source': source, 'samples': samples[a:b]}, 'POST') for a, b in [(0, 2000), (2000, 8000)]]
            assert all(r['persistence'] == 'saved' for r in responses)
            assert responses[0]['predictions'] == []
            result = responses[-1]
            predictions = result['predictions']
            assert len(predictions) == 3
            for live, uploaded in zip(predictions, manual['predictions']):
                assert live['family']['label'] == uploaded['family']['label']
                assert live['detailed_condition']['label'] == uploaded['detailed_condition']['label']
                assert np.isclose(live['family']['score'], uploaded['family']['score'])
                assert np.isclose(live['detailed_condition']['score'], uploaded['detailed_condition']['score'])
            documents = list(db.live_data.find({'session_id': result['session_id']}).sort('start_timestamp_seconds', 1))
            assert len(documents) == 2
            offset = 0
            for doc in documents:
                count = doc['total_samples']
                np.testing.assert_array_equal(np.frombuffer(doc['timestamps_bin'], dtype='<f8'), times[offset:offset+count])
                restored = np.column_stack([np.frombuffer(doc['channel_data_bin'][axis], dtype='<f4') for axis in CHANNELS])
                np.testing.assert_array_equal(restored, values[offset:offset+count])
                assert doc['model_input_order'] == list(CHANNELS) and doc['fs'] == 4000
                assert doc['source_timestamp_ms'] == source['source_timestamp_ms']
                assert np.isclose(doc['recording_duration_seconds'], count / 4000)
                assert doc['window_results'] and 'health_score' not in doc
                offset += count
            assert len(documents[0]['window_results']) == 1  # first window crosses both batches
            events = list(db.live_faults.find({'session_id': result['session_id']}))
            expected_faults = sum(p['family']['label'] != 'normal'
                                 or p['detailed_condition']['label'].split('__', 1)[0] != 'normal'
                                 or p.get('independent_detailed_condition', {}).get('label', 'normal').split('__', 1)[0] != 'normal'
                                 for p in predictions)
            faults = [event for event in events if event['event_type'] == 'fault_candidate']
            normals = [event for event in events if event['event_type'] == 'normal_prediction']
            assert len(events) == len(predictions)
            assert len(faults) == expected_faults
            assert len(normals) == len(predictions) - expected_faults
            for fault in faults:
                assert fault['is_fault'] and fault['decision'] in {'classified', 'needs_review'} and fault['fault_subtype']
                assert fault['mongo_references']
                assert fault['detail_model_sha256'] == manifest['detail_sha256']
            for normal in normals:
                assert normal['is_fault'] is False and normal['fault_family'] == 'normal'
                assert normal['fault_subtype'].startswith('normal__')
                assert normal['condition_label'] == normal['fault_subtype']
                assert normal['mongo_references']
            history = request(f'/v1/streams/{stream}/history')
            assert len(history['predictions']) == 3 and len(history['fault_candidates']) == expected_faults
            assert len(history['normal_predictions']) == len(normals)
            status = request(f'/v1/streams/{stream}')
            assert np.isclose(status['recording_time']['duration_seconds'], 2)
            assert status['pending_result_batch'] is None
            diagnostics = request(f'/v1/streams/{stream}/diagnostics')
            assert diagnostics['waveform'] and diagnostics['spectrum'] and diagnostics['trends']
            bad = {'source': source, 'samples': [{'timestamp': float(times[-1] + .01), 'channels': values[-1].tolist()}]}
            request(f'/v1/streams/{stream}/samples', bad, 'POST', expected=422)
            assert db.live_data.count_documents({'session_id': result['session_id']}) == 2
            request(f'/v1/streams/{stream}', method='DELETE')
            restarted = request(f'/v1/streams/{stream}/samples', {'source': source, 'samples': samples[:4000]}, 'POST')
            assert restarted['session_id'] != result['session_id']
            history = request(f'/v1/streams/{stream}/history')
            assert len(history['predictions']) == 4
            assert {p['session_id'] for p in history['predictions']} == {result['session_id'], restarted['session_id']}
            report['streams'].append(dict(source_csv=source_path, test_stream=stream,
                samples=8000, windows=3, raw_batches=2, fault_candidates=expected_faults,
                family=predictions[0]['family']['label'], child=predictions[0]['detailed_condition']['label'],
                reset_preserved_both_sessions=True))
        report['checks'].update(raw_binary_roundtrip=True, save_read_predict=True,
            manual_prediction_not_stored=True, live_manual_predictions_match=True,
            recording_time_and_channels=True, normal_and_fault_storage=True,
            invalid_gap_rejected_before_storage=True, history_and_diagnostics=True,
            only_existing_collections=sorted(db.list_collection_names()) == before_collections)
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # Exact UUID-tagged sessions only; never delete unrelated live records.
        query = {'stream_id': {'$in': streams}, 'record_type': 'pipeline_connectivity_check', 'live_label': tag}
        session_ids = list(db.live_data.distinct('session_id', query))
        deleted_faults = db.live_faults.delete_many({'session_id': {'$in': session_ids}, 'stream_id': {'$in': streams}}).deleted_count
        deleted_batches = db.live_data.delete_many(query).deleted_count
        for stream in streams:
            try:
                request(f'/v1/streams/{stream}', method='DELETE')
            except Exception:
                pass
        report['cleanup'] = dict(deleted_test_batches=deleted_batches, deleted_test_faults=deleted_faults,
            remaining_test_batches=db.live_data.count_documents(query),
            remaining_test_faults=db.live_faults.count_documents({'session_id': {'$in': session_ids}}))
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        destination = root / 'reports/live_pipeline_review'
        destination.mkdir(exist_ok=True)
        (destination / 'connectivity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        mongo.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    run(parser.parse_args().url)
