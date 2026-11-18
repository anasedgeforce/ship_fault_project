"""Live quality, provenance, review, and sustained fault behavior."""
from __future__ import annotations

import numpy as np
import pytest

from ship_fault.live_monitor import inspect_window
from ship_fault.settings import APP_EQUIPMENT_ID, CHANNELS, RuntimeSettings
from ship_fault.signal import validate_samples
from ship_fault.source import normalize_source
from ship_fault.stream import StreamSession

EQUIPMENT_ID = APP_EQUIPMENT_ID


class FixedFaultEngine:
    def __init__(self, validated=False):
        self.settings = RuntimeSettings(validated_equipment_ids=(EQUIPMENT_ID,) if validated else ())
        self.manifest = {'training_windows_per_file': 15}
        self.model_info = {'family_model_sha256': 'a' * 64}

    def predict_probabilities(self, windows, elapsed_starts):
        return np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])

    def format_predictions(self, family, detail):
        return [{'family': {'label': 'inner_race', 'score': 1.0},
                 'detailed_condition': {'label': 'inner_race__low__loader', 'score': 1.0}}]

    def rank_family(self, probabilities):
        return {'label': 'inner_race', 'score': 1.0}


def signal_samples(length=8000):
    times = np.arange(length) / 4000
    values = np.column_stack([np.sin(2 * np.pi * (41 + axis * 7) * times)
                              for axis in range(6)])
    return times, values


def source():
    return normalize_source({'equipment_id': EQUIPMENT_ID, 'channel_names': list(CHANNELS),
                             'channel_units': 'mm/s', 'sample_rate_hz': 4000}, 4000)


def test_flat_or_identical_channel_blocks_fault_prediction():
    times, values = signal_samples(4000)
    values[:, 0] = 0
    values[:, 1] = values[:, 2]
    quality = inspect_window(values, 4000)
    assert quality['status'] == 'invalid'
    assert {item['code'] for item in quality['issues']} >= {'flat_channel', 'identical_channels'}
    prediction = StreamSession(FixedFaultEngine(validated=True)).push_batch(times, values, source())['predictions'][0]
    assert prediction['decision'] == 'invalid_input'
    assert prediction['family'] is None
    assert prediction['fault_type'] is None


def test_timestamp_gap_and_wrong_axis_declaration_rejected():
    times, values = signal_samples(4000)
    times[100] += 1 / 4000
    with pytest.raises(ValueError, match='timestamp gap|strictly increase'):
        validate_samples(times, values, RuntimeSettings())
    with pytest.raises(ValueError, match='exact order'):
        normalize_source({'channel_names': list(reversed(CHANNELS))}, 4000)
    with pytest.raises(ValueError, match='4000 Hz'):
        normalize_source({'sample_rate_hz': 8000}, 4000)
    with pytest.raises(ValueError, match='4000 Hz'):
        normalize_source({'sample_rate_hz': float('nan')}, 4000)


def test_verified_training_requires_ground_truth_label_and_declared_axes():
    base = {'record_type': 'verified_training', 'channel_names': list(CHANNELS),
            'channel_units': 'mm/s'}
    with pytest.raises(ValueError, match='known detailed condition'):
        normalize_source(base, 4000)
    with pytest.raises(ValueError, match='known detailed condition'):
        normalize_source({**base, 'live_label': 'model_guess'}, 4000)
    with pytest.raises(ValueError, match='declared six-axis'):
        normalize_source({'record_type': 'verified_training',
                          'live_label': 'outer_race__high_severity__no_ub_specified'}, 4000)
    result = normalize_source({**base,
        'live_label': 'outer_race__high_severity__no_ub_specified'}, 4000)
    assert result['live_label'] == 'outer_race__high_severity__no_ub_specified'


def test_unvalidated_equipment_needs_review_and_does_not_alert():
    times, values = signal_samples()
    session = StreamSession(FixedFaultEngine(), 'pump')
    response = session.push_batch(times, values, source())
    assert len(response['predictions']) == 3
    assert all(item['decision'] == 'needs_review' for item in response['predictions'])
    assert not response['alerts']


def test_stream_does_not_mix_equipment_or_drop_metadata_after_start():
    times, values = signal_samples(200)
    session = StreamSession(FixedFaultEngine(), 'pump')
    session.push_batch(times[:100], values[:100], source())
    with pytest.raises(ValueError, match='equipment_id missing or changed'):
        session.push_batch(times[100:], values[100:])
    assert session.sample_count == 100
    other = {**source(), 'equipment_id': 'a' * 24}
    with pytest.raises(ValueError, match='equipment_id missing or changed'):
        session.push_batch(times[100:], values[100:], other)
    assert session.sample_count == 100


def test_validated_source_alerts_after_three_matching_windows_with_mongo_reference():
    times, values = signal_samples()
    session = StreamSession(FixedFaultEngine(validated=True), 'pump')
    reference = {'database': 'ship_fault_monitoring', 'collection': 'live_data',
                 'document_id': 'b' * 24}
    response = session.push_batch(times, values, source(), lambda *_: reference)
    assert len(response['alerts']) == 1
    alert = response['alerts'][0]
    assert alert['window_index'] == 2
    assert alert['fault_type'] == 'inner_race'
    assert alert['fault_subtype'] == 'inner_race__low__loader'
    assert alert['equipment_id'] == EQUIPMENT_ID
    assert alert['mongo_references'] == [reference]
    assert session.status(after=1)['alerts'] == [alert]
    assert session.diagnostics()['waveform']
    assert session.diagnostics()['spectrum']
    assert len(session.diagnostics()['trends']) == 3
