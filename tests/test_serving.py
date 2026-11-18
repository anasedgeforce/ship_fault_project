"""Manual and live input parity for the deployable serving boundary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ship_fault.api import create_app
from ship_fault.models import InferenceEngine, _verified_artifact
from ship_fault.settings import RuntimeSettings
from ship_fault.signal import parse_csv_recording, recording_windows, validate_samples
from ship_fault.stream import StreamSession
from src.features import append_elapsed_time_feature, extract_features
from src.round5_order_features import order_features
from src.train import load_cached


@pytest.fixture(scope='module')
def settings():
    return RuntimeSettings()


@pytest.fixture(scope='module')
def engine(settings):
    loaded = InferenceEngine(settings)
    assert loaded.model_info['primary_task'] == '6-family'
    assert loaded.model_info['primary_output'] == 'predictions[].family'
    assert 0 < loaded.model_info['primary_validation_accuracy'] <= 1
    return loaded


@pytest.fixture(scope='module')
def recording():
    return pd.read_csv('data/data_day1/Cocked Bearing with loader/1.csv', nrows=6000)


def test_manual_csv_and_numeric_values_agree(engine, settings, recording):
    raw = recording.to_csv(index=False).encode('utf-8')
    times, channels = parse_csv_recording(raw, settings)
    from_csv = engine.predict_recording(times, channels)
    from_arrays = engine.predict_recording(recording.Time.to_numpy(), recording.iloc[:, 1:].to_numpy())
    assert from_csv['window_count'] == from_arrays['window_count'] == 2
    for left, right in zip(from_csv['predictions'], from_arrays['predictions']):
        for key in ('family', 'detailed_condition'):
            assert left[key]['label'] == right[key]['label']
            assert left[key]['score'] == pytest.approx(right[key]['score'])
    assert from_csv['recording_family'] == from_arrays['recording_family']


def test_training_and_serving_features_match_for_original_csv(engine, settings):
    project = settings.project_root
    cached, metadata, names = load_cached(project)
    path = metadata.file.iloc[0]
    times, channels = parse_csv_recording((project / 'data' / path).read_bytes(), settings)
    time, windows, starts = recording_windows(times, channels, settings)
    original, feature_names = extract_features(windows, settings.sample_rate_hz,
                                               engine.speed_hz, engine.bearing_geometry)
    original, feature_names = append_elapsed_time_feature(original, feature_names,
                                                            time[starts] - time[0])
    order, order_names = order_features(windows, settings.sample_rate_hz)
    with np.load(project / 'cache' / 'round5_order_features.npz') as data:
        cached_order = data['X'][metadata.file.eq(path).to_numpy()]
    assert feature_names == names
    assert order_names == engine.family['order_features']
    np.testing.assert_allclose(original, cached[metadata.file.eq(path).to_numpy()], atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(order, cached_order, atol=1e-5, rtol=1e-6)


def test_recording_flags_disagreement_between_aggregate_family_and_subtype(engine, recording, monkeypatch):
    monkeypatch.setitem(engine.family, 'family_names', ['inner_race', 'outer_race'])
    monkeypatch.setitem(engine.detail, 'label_names', ['inner_race__low', 'outer_race__high'])
    monkeypatch.setattr(engine, 'predict_probabilities', lambda *_args: (
        np.array([[0.9, 0.1], [0.4, 0.6]]), np.array([[0.51, 0.49], [0.0, 1.0]])))
    monkeypatch.setattr(engine, 'novelty_score', lambda windows: [
        {'outside_reference': False} for _ in windows])
    result = engine.predict_recording(recording.Time, recording.iloc[:, 1:])
    assert all(window['decision'] == 'needs_review' for window in result['predictions'])
    assert all('physical_axis_mapping_unverified' in window['review_reasons']
               and 'equipment_domain_unvalidated' in window['review_reasons']
               for window in result['predictions'])
    assert result['recording_family']['label'] == 'inner_race'
    assert result['recording_detailed_condition']['label'] == 'inner_race__low'
    assert result['independent_recording_detailed_condition']['label'] == 'outer_race__high'
    assert result['recording_decision'] == 'needs_review'
    assert set(result['recording_review_reasons']) == {
        'family_detail_disagree', 'physical_axis_mapping_unverified', 'equipment_domain_unvalidated'}
    assert result['input_verification']['physical_axis_mapping'] == 'unverified'


def test_child_prediction_stays_with_family_without_hiding_disagreement(engine, monkeypatch):
    monkeypatch.setitem(engine.family, 'family_names', ['inner_race', 'outer_race'])
    monkeypatch.setitem(engine.detail, 'label_names',
                        ['inner_race__low', 'outer_race__high', 'inner_race__high'])
    result = engine.format_predictions(np.array([[0.8, 0.2]]),
                                       np.array([[0.3, 0.6, 0.1]]))[0]
    assert result['family']['label'] == 'inner_race'
    assert result['detailed_condition']['label'] == 'inner_race__low'
    assert result['detailed_condition']['score'] == pytest.approx(0.3)
    assert result['independent_detailed_condition']['label'] == 'outer_race__high'


def test_stream_batches_match_manual_windows(engine, recording):
    batch = engine.predict_recording(recording.Time, recording.iloc[:, 1:])['predictions']
    session = StreamSession(engine)
    first = session.push_batch(recording.Time.iloc[:3000], recording.iloc[:3000, 1:])
    second = session.push_batch(recording.Time.iloc[3000:], recording.iloc[3000:, 1:])
    assert first['predictions'] == []
    assert second['total_samples'] == 6000
    assert len(second['predictions']) == 2
    for actual, expected in zip(second['predictions'], batch):
        assert actual['family']['label'] == expected['family']['label']
        assert actual['detailed_condition']['label'] == expected['detailed_condition']['label']
        assert actual['family']['score'] == pytest.approx(expected['family']['score'])
        assert actual['start_seconds'] == expected['start_seconds']


def test_live_five_window_family_uses_only_prior_windows(engine, settings):
    recording = pd.read_csv('data/data_day1/Cocked Bearing with loader/1.csv', nrows=12000)
    _, windows, _ = recording_windows(
        recording.Time, recording.iloc[:, 1:], settings)
    family_probabilities, _ = engine.predict_probabilities(windows)
    session = StreamSession(engine)
    emitted = []
    for start in range(0, len(recording), 4000):
        emitted.extend(session.push_batch(recording.Time.iloc[start:start + 4000],
                                          recording.iloc[start:start + 4000, 1:])['predictions'])
    assert len(emitted) == 5
    for index, item in enumerate(emitted):
        expected = engine.rank_family(family_probabilities[max(0, index - 4):index + 1].mean(axis=0))
        assert item['smoothed_family']['label'] == expected['label']
        assert item['smoothed_family']['score'] == pytest.approx(expected['score'])


def test_live_status_returns_all_windows_after_cursor(engine, recording):
    session = StreamSession(engine)
    session.push_batch(recording.Time, recording.iloc[:, 1:])
    first = session.status(after=-1)
    assert [item['window_index'] for item in first['predictions']] == [0, 1]
    assert [item['window_index'] for item in session.status(after=0)['predictions']] == [1]
    assert session.status(after=1)['predictions'] == []
    assert not first['history_truncated']


def test_live_reports_elapsed_time_for_absolute_sensor_timestamps(engine, recording):
    session = StreamSession(engine)
    shifted = recording.Time.to_numpy() + 1_000.0
    first = session.push_batch(shifted, recording.iloc[:, 1:])['predictions'][0]
    assert first['start_seconds'] == pytest.approx(1_000.0)
    assert first['elapsed_start_seconds'] == pytest.approx(0.0)


def test_live_history_reports_when_old_windows_were_dropped(engine):
    session = StreamSession(engine)
    for index in range(260):
        session.recent_predictions.append({'window_index': index})
    result = session.status(after=0)
    assert result['oldest_available_window_index'] == 4
    assert result['history_truncated']
    assert len(result['predictions']) == 256


def test_long_live_stream_is_marked_beyond_training_window_budget(engine, settings, recording):
    samples = np.tile(recording.iloc[:4000, 1:].to_numpy(), (9, 1))
    timestamps = np.arange(len(samples)) / settings.sample_rate_hz
    predictions = StreamSession(engine).push_batch(timestamps, samples)['predictions']
    assert predictions[14]['beyond_training_window_budget'] is False
    assert predictions[15]['beyond_training_window_budget'] is True


def test_bad_sensor_input_rejected_without_advancing_stream(engine, settings, recording):
    with pytest.raises(ValueError, match='six channels'):
        validate_samples([0.0], [[1.0, 2.0]], settings)
    with pytest.raises(ValueError, match='4000 Hz'):
        validate_samples(np.arange(4000) / 2000, np.ones((4000, 6)), settings)
    session = StreamSession(engine)
    session.push_batch(recording.Time.iloc[:100], recording.iloc[:100, 1:])
    with pytest.raises(ValueError, match='strictly increase'):
        session.push_batch([0.0], [[0.0] * 6])
    assert session.sample_count == 100
    malformed = recording.iloc[:4000].to_csv(index=False)
    malformed += ','.join(recording.columns) + '\n'
    with pytest.raises(ValueError, match='numeric|finite'):
        parse_csv_recording(malformed.encode(), settings)


def test_http_manual_and_live_routes(settings, recording):
    protected = RuntimeSettings(project_root=settings.project_root, api_key='test-secret')
    with TestClient(create_app(protected)) as client:
        ready = client.get('/health/ready')
        assert ready.status_code == 200
        assert ready.json()['models']['family_classes'] == 6
        assert client.get('/v1/models').status_code == 401
        headers = {'X-API-Key': 'test-secret'}
        csv_bytes = recording.iloc[:4000].to_csv(index=False).encode()
        offline = client.post('/v1/predict/csv', content=csv_bytes, headers=headers)
        assert offline.status_code == 200
        payload = {'timestamps': recording.Time.iloc[:4000].tolist(),
                   'channels': recording.iloc[:4000, 1:].values.tolist()}
        manual = client.post('/v1/predict/recording', json=payload, headers=headers)
        assert manual.status_code == 200
        assert manual.json()['predictions'][0]['family']['label'] == offline.json()['predictions'][0]['family']['label']
        samples = [{'timestamp': t, 'channels': values} for t, values in
                   zip(payload['timestamps'], payload['channels'])]
        partial = client.post('/v1/streams/motor_1/samples', json={'samples': samples[:2000]}, headers=headers)
        complete = client.post('/v1/streams/motor_1/samples', json={'samples': samples[2000:]}, headers=headers)
        assert partial.status_code == complete.status_code == 200
        assert not partial.json()['predictions']
        assert len(complete.json()['predictions']) == 1
        assert client.delete('/v1/streams/motor_1', headers=headers).status_code == 200


def test_http_live_poll_returns_each_new_window(settings, recording):
    samples = [{'timestamp': t, 'channels': values} for t, values in
               zip(recording.Time.tolist(), recording.iloc[:, 1:].values.tolist())]
    with TestClient(create_app(settings)) as client:
        posted = client.post('/v1/streams/test_poll/samples', json={'samples': samples})
        assert posted.status_code == 200
        assert len(posted.json()['predictions']) == 2
        all_new = client.get('/v1/streams/test_poll?after=-1')
        assert [item['window_index'] for item in all_new.json()['predictions']] == [0, 1]
        next_only = client.get('/v1/streams/test_poll?after=0')
        assert [item['window_index'] for item in next_only.json()['predictions']] == [1]
        assert client.get('/v1/streams/test_poll?after=-2').status_code == 422


def test_artifact_hash_checked_before_loading(settings):
    with pytest.raises(ValueError, match='hash differs'):
        _verified_artifact(settings.project_root, {
            'family_artifact': 'artifacts/runs/20261104T041252Z_164da514/family.joblib',
            'family_sha256': '0' * 64,
        }, 'family')
