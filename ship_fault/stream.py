"""Bounded, per-device rolling windows for direct vibration samples."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager

import numpy as np

from .live_monitor import inspect_window, review_reasons
from .models import InferenceEngine
from .signal import validate_samples


class StreamSession:
    def __init__(self, engine: InferenceEngine, stream_id: str | None = None):
        self.engine = engine
        self.stream_id = stream_id
        self.session_id = uuid.uuid4().hex
        self.settings = engine.settings
        self.training_window_budget = int(engine.manifest.get('training_windows_per_file', 0))
        self.timestamps = deque(maxlen=self.settings.window_samples)
        self.channels = deque(maxlen=self.settings.window_samples)
        self.sources = deque(maxlen=self.settings.window_samples)
        self.equipment_id = None
        self.sample_count = 0
        self.last_timestamp = None
        self.first_timestamp = None
        self.latest_source = None
        self.latest = None
        self.recent_predictions = deque(maxlen=256)
        self.alerts = deque(maxlen=128)
        self.family_history = deque(maxlen=5)
        self.fault_streak = 0
        self.last_fault_key = None
        self.last_active = time.monotonic()
        self.pending_persistence = None
        self.recovered_from_mongo = False
        self.recovery_model = {
            'policy_version': 1,
            'family': engine.model_info.get('family_model_sha256'),
            'detail': engine.model_info.get('detail_model_sha256'),
            'feature_config': engine.model_info.get('feature_config_sha256'),
            'novelty_reference': engine.model_info.get('novelty_reference_sha256'),
            'sample_rate_hz': self.settings.sample_rate_hz,
            'window_samples': self.settings.window_samples,
            'hop_samples': self.settings.hop_samples,
            'validated_equipment_ids': list(self.settings.validated_equipment_ids),
            'alert_consecutive_windows': self.settings.alert_consecutive_windows,
        }
        self.lock = threading.RLock()

    @contextmanager
    def _rollback_on_failure(self):
        # A saved raw batch stays available for retry. The rolling window must
        # consume it exactly once even if inference fails partway through it.
        fields = ('timestamps', 'channels', 'sources', 'equipment_id', 'sample_count',
                  'last_timestamp', 'first_timestamp', 'latest_source', 'latest',
                  'recent_predictions', 'alerts', 'family_history', 'fault_streak',
                  'last_fault_key', 'last_active')
        snapshot = {name: (getattr(self, name).copy() if isinstance(getattr(self, name), deque)
                           else getattr(self, name)) for name in fields}
        try:
            yield
        except Exception:
            for name, value in snapshot.items():
                setattr(self, name, value)
            raise

    def push_batch(self, timestamps, channels, source=None, persist=None, fetch=None):
        with self.lock, self._rollback_on_failure():
            times, values = validate_samples(
                timestamps, channels, self.settings, minimum=1,
                maximum=self.settings.max_batch_samples,
                previous_timestamp=self.last_timestamp)
            incoming_equipment = source.get('equipment_id') if source else None
            if self.equipment_id and self.equipment_id != incoming_equipment:
                raise ValueError('equipment_id missing or changed within this stream; reset or use a new stream_id')
            if self.sample_count and not self.equipment_id and incoming_equipment:
                raise ValueError('equipment_id cannot be assigned after untagged samples; reset the stream')
            reference = None
            if persist is not None:
                reference = persist(times, values, source)
                if fetch is not None:
                    times, values = fetch(reference, times, values)
                source = {**(source or {}), 'mongo': reference}
            self.latest_source = source
            if incoming_equipment and not self.equipment_id:
                self.equipment_id = incoming_equipment
            emitted = []
            emitted_alerts = []
            self.last_active = time.monotonic()
            for timestamp, sample in zip(times, values):
                if self.first_timestamp is None:
                    self.first_timestamp = float(timestamp)
                self.timestamps.append(float(timestamp))
                self.channels.append(sample)
                self.sources.append(source)
                self.sample_count += 1
                if (self.sample_count >= self.settings.window_samples and
                        (self.sample_count - self.settings.window_samples) % self.settings.hop_samples == 0):
                    window = np.asarray(self.channels, dtype=np.float32)
                    elapsed_start = self.timestamps[0] - self.first_timestamp
                    window_index = ((self.sample_count - self.settings.window_samples)
                                    // self.settings.hop_samples)
                    beyond_budget = self.training_window_budget > 0 and window_index >= self.training_window_budget
                    quality = inspect_window(window, self.settings.sample_rate_hz)
                    references = []
                    seen = set()
                    for item in self.sources:
                        ref = item.get('mongo') if item else None
                        if ref:
                            key = (ref['database'], ref['collection'], ref['document_id'])
                            if key not in seen:
                                seen.add(key)
                                references.append(ref)
                    source_info = next((item for item in reversed(self.sources) if item), None)
                    mapping_declared = all(bool(item and item.get('channel_names')
                                                and item.get('channel_units') == 'mm/s')
                                           for item in self.sources)
                    equipment_validated = bool(self.equipment_id and
                                               self.equipment_id in self.settings.validated_equipment_ids)
                    if quality['status'] != 'ok':
                        result = {'family': None, 'detailed_condition': None,
                                  'smoothed_family': None, 'decision': 'invalid_input',
                                  'review_reasons': [item['code'] for item in quality['issues']]}
                        self.family_history.clear()
                    else:
                        family_probs, detail_probs = self.engine.predict_probabilities(
                            window[None, :, :], np.array([elapsed_start]))
                        result = self.engine.format_predictions(family_probs, detail_probs)[0]
                        self.family_history.append(family_probs[0])
                        result['smoothed_family'] = self.engine.rank_family(
                            np.mean(self.family_history, axis=0))
                        reasons = review_reasons(result['family'], result['detailed_condition'],
                                                 independent_detail=result.get('independent_detailed_condition'),
                                                 beyond_budget=beyond_budget,
                                                 mapping_verified=mapping_declared and equipment_validated,
                                                 equipment_validated=equipment_validated)
                        novelty = (self.engine.novelty_score(window[None, :, :])[0]
                                   if hasattr(self.engine, 'novelty_score') else None)
                        result['novelty'] = novelty
                        if novelty is None and hasattr(self.engine, 'novelty_score'):
                            reasons.append('novelty_reference_unavailable')
                        if novelty and novelty['outside_reference']:
                            reasons.append('outside_training_feature_reference')
                        result['decision'] = 'needs_review' if reasons else 'classified'
                        result['review_reasons'] = reasons
                    result['window_index'] = window_index
                    result['session_id'] = self.session_id
                    result['beyond_training_window_budget'] = beyond_budget
                    result['start_seconds'] = self.timestamps[0]
                    result['end_seconds'] = self.timestamps[-1]
                    result['elapsed_start_seconds'] = elapsed_start
                    result['quality'] = quality
                    result['equipment_id'] = self.equipment_id
                    result['family_model_sha256'] = self.engine.model_info.get('family_model_sha256')
                    result['detail_model_sha256'] = self.engine.model_info.get('detail_model_sha256')
                    result['source'] = source_info
                    result['mongo_references'] = references
                    result['fault_type'] = result['family']['label'] if result['decision'] == 'classified' else None
                    result['fault_subtype'] = (result['detailed_condition']['label']
                                               if result['decision'] == 'classified' else None)
                    fault_key = (result['fault_type'], result['fault_subtype'])
                    if result['decision'] == 'classified' and result['fault_type'] != 'normal':
                        self.fault_streak = self.fault_streak + 1 if fault_key == self.last_fault_key else 1
                        self.last_fault_key = fault_key
                        if self.fault_streak == self.settings.alert_consecutive_windows:
                            alert = {'stream_id': self.stream_id, 'equipment_id': self.equipment_id,
                                     'session_id': self.session_id,
                                     'window_index': window_index, 'timestamp_seconds': result['end_seconds'],
                                     'fault_type': result['fault_type'], 'fault_subtype': result['fault_subtype'],
                                     'family_score': result['family']['score'],
                                     'subtype_score': result['detailed_condition']['score'],
                                     'consecutive_windows': self.fault_streak,
                                     'mongo_references': references,
                                     'family_model_sha256': self.engine.model_info.get('family_model_sha256'),
                                     'detail_model_sha256': self.engine.model_info.get('detail_model_sha256')}
                            self.alerts.append(alert)
                            emitted_alerts.append(alert)
                    else:
                        self.fault_streak = 0
                        self.last_fault_key = None
                    self.latest = result
                    self.recent_predictions.append(result)
                    emitted.append(result)
            self.last_timestamp = float(times[-1])
            return {'session_id': self.session_id, 'batch_reference': reference,
                    'accepted_samples': len(times), 'total_samples': self.sample_count,
                    'ready': len(self.channels) == self.settings.window_samples,
                    'predictions': emitted, 'alerts': emitted_alerts,
                    'equipment_id': self.equipment_id}

    def status(self, after: int = -1) -> dict:
        """Return every retained prediction after a browser's last window index."""
        with self.lock:
            oldest = self.recent_predictions[0]['window_index'] if self.recent_predictions else None
            return {
                'session_id': self.session_id,
                'persistence': 'result_save_failed' if self.pending_persistence is not None else 'up_to_date',
                'recovered_from_mongo': self.recovered_from_mongo,
                'pending_result_batch': (self.pending_persistence['batch_reference']
                                         if self.pending_persistence is not None else None),
                'total_samples': self.sample_count,
                'ready': self.sample_count >= self.settings.window_samples,
                'latest': self.latest,
                'predictions': [item for item in self.recent_predictions
                                if item['window_index'] > after],
                'alerts': [item for item in self.alerts if item['window_index'] > after],
                'equipment_id': self.equipment_id,
                'recording_time': {
                    'sensor_start_seconds': self.first_timestamp,
                    'sensor_end_seconds': self.last_timestamp,
                    'duration_seconds': (self.last_timestamp - self.first_timestamp +
                                         1 / self.settings.sample_rate_hz)
                        if self.first_timestamp is not None and self.last_timestamp is not None else 0,
                    'source_timestamp_ms': (self.latest_source or {}).get('source_timestamp_ms'),
                    'time_basis': 'sensor_seconds',
                },
                'oldest_available_window_index': oldest,
                'history_truncated': oldest is not None and after < oldest - 1,
            }

    def diagnostics(self) -> dict:
        """Small bounded waveform and spectrum payload for the live dashboard."""
        with self.lock:
            if not self.channels:
                return {'waveform': [], 'spectrum': [], 'trends': []}
            values = np.asarray(self.channels, dtype=np.float32)
            times = np.asarray(self.timestamps, dtype=np.float64)
            step = max(1, int(np.ceil(len(values) / 250)))
            waveform = [{'timestamp_seconds': float(t), 'channels': row.tolist()}
                        for t, row in zip(times[::step], values[::step])]
            spectrum = []
            if len(values) >= self.settings.window_samples:
                centered = values - values.mean(axis=0)
                magnitudes = np.mean(np.abs(np.fft.rfft(centered, axis=0)), axis=1)
                frequencies = np.fft.rfftfreq(len(values), 1 / self.settings.sample_rate_hz)
                spectrum = [{'hz': float(frequencies[i]), 'magnitude': float(magnitudes[i])}
                            for i in range(0, min(len(magnitudes), 1001), 4)]
            trends = [{'window_index': item['window_index'],
                       'elapsed_start_seconds': item['elapsed_start_seconds'],
                       'decision': item['decision'],
                       'fault_type': item['fault_type'], 'fault_subtype': item['fault_subtype'],
                       'rms_mm_s': [axis['rms_mm_s'] for axis in item['quality']['channels']]}
                      for item in list(self.recent_predictions)[-100:]
                      if 'quality' in item]
            return {'waveform': waveform, 'spectrum': spectrum, 'trends': trends}


class StreamManager:
    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.sessions: dict[str, StreamSession] = {}
        self.lock = threading.RLock()
        self.restore = None
        self.close_session = None

    def get_or_create(self, stream_id: str) -> StreamSession:
        with self.lock:
            now = time.monotonic()
            for key, session in list(self.sessions.items()):
                if (session.pending_persistence is None
                        and now - session.last_active > self.engine.settings.stream_idle_seconds):
                    del self.sessions[key]
            if stream_id not in self.sessions:
                if len(self.sessions) >= self.engine.settings.max_streams:
                    raise ValueError('maximum active vibration streams reached')
                session = StreamSession(self.engine, stream_id)
                if self.restore is not None:
                    self.restore(session)
                self.sessions[stream_id] = session
            return self.sessions[stream_id]

    def get(self, stream_id: str) -> StreamSession | None:
        with self.lock:
            if stream_id not in self.sessions and self.restore is not None:
                if len(self.sessions) >= self.engine.settings.max_streams:
                    raise ValueError('maximum active vibration streams reached')
                session = StreamSession(self.engine, stream_id)
                if self.restore(session):
                    self.sessions[stream_id] = session
            return self.sessions.get(stream_id)

    def reset(self, stream_id: str) -> bool:
        with self.lock:
            session = self.sessions.get(stream_id)
            if session is None:
                return False
            with session.lock:
                if session.pending_persistence is not None:
                    raise ValueError('Save the pending results before resetting this stream')
                if self.close_session is not None:
                    self.close_session(session)
                self.sessions.pop(stream_id)
                return True
