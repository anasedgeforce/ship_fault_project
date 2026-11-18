"""Load verified model artifacts and apply their saved feature schemas."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src.features import extract_features, append_elapsed_time_feature, TIME_FEATURE_NAME
from src.round5_order_features import order_features

from .settings import CHANNELS, RuntimeSettings
from .presentation import label_catalog
from .signal import recording_windows


def _verified_artifact(root: Path, manifest: dict, task: str):
    relative = Path(str(manifest[f'{task}_artifact']).replace('\\', '/'))
    path = (root / relative).resolve()
    if not path.is_relative_to((root / 'artifacts').resolve()) or not path.is_file():
        raise ValueError(f'{task} model must exist inside the project artifacts directory')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest[f'{task}_sha256']:
        raise ValueError(f'{task} model hash differs from the training manifest')
    # joblib uses pickle: load only the trusted artifacts verified above.
    return joblib.load(path), actual


def _aligned_probabilities(model, features, classes: int):
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    output = np.zeros((len(features), classes), dtype=np.float64)
    output[:, np.asarray(model.classes_, dtype=int)] = probabilities
    if not np.isfinite(output).all():
        raise ValueError('model returned non-finite probabilities')
    return output


def _ranked(probabilities, labels):
    ranked = np.argsort(-probabilities)[:3]
    return {
        'label': str(labels[int(ranked[0])]),
        'score': float(probabilities[ranked[0]]),
        'top3': [{'label': str(labels[int(index)]), 'score': float(probabilities[index])}
                 for index in ranked],
    }


def _ranked_child(probabilities, labels, family_label):
    eligible = [index for index, label in enumerate(labels)
                if str(label).split('__', 1)[0] == family_label]
    if not eligible:
        raise ValueError(f'no detailed conditions exist for family {family_label}')
    ranked = sorted(eligible, key=lambda index: -probabilities[index])[:3]
    return {'label': str(labels[ranked[0]]), 'score': float(probabilities[ranked[0]]),
            'top3': [{'label': str(labels[index]), 'score': float(probabilities[index])}
                     for index in ranked]}


class InferenceEngine:
    """One feature path for CSV, manual arrays, and streaming samples."""

    def __init__(self, settings: RuntimeSettings, manifest_path: Path | None = None,
                 load_novelty: bool = True):
        self.settings = settings
        root = settings.project_root
        manifest_path = Path(manifest_path) if manifest_path is not None else root / 'reports' / 'duplicate_inclusive' / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (not str(self.manifest.get('primary_task','')).endswith('-family')
                or self.manifest.get('primary_output') != 'predictions[].family'):
            raise ValueError('serving manifest must select the family per-window model')
        self.family, self.family_hash = _verified_artifact(root, self.manifest, 'family')
        self.detail, self.detail_hash = _verified_artifact(root, self.manifest, 'detail')
        config_relative=Path(str(self.manifest.get('feature_config_artifact','artifacts/config_snapshot.json')).replace('\\','/'))
        config_path=(root/config_relative).resolve()
        if not config_path.is_relative_to((root/'artifacts').resolve()):
            raise ValueError('feature configuration must be inside the artifacts directory')
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != self.manifest['config_snapshot_sha256']:
            raise ValueError('feature configuration hash differs from the training manifest')
        snapshot = json.loads(config_path.read_text(encoding='utf-8'))
        if (self.manifest.get('speed_source') != 'configured_reference'
                or snapshot.get('speed_source') != 'configured_reference'
                or float(snapshot['estimated_speed_hz']) != float(snapshot['run_speed_hz'])):
            raise ValueError('serving requires the fixed training harmonic reference')
        if self.manifest.get('time_feature') != TIME_FEATURE_NAME or snapshot.get('time_feature') != TIME_FEATURE_NAME:
            raise ValueError('serving requires the trained Time-column feature')
        if self.manifest['expected_channels'] != list(CHANNELS):
            raise ValueError('runtime channel order differs from the model training setting')
        for key, actual in (('sample_rate_hz', settings.sample_rate_hz),
                            ('window_seconds', settings.window_seconds),
                            ('hop_seconds', settings.hop_seconds)):
            if abs(float(self.manifest[key]) - actual) > 1e-9:
                raise ValueError(f'runtime {key} differs from the model training setting')
        if abs(float(snapshot['window_seconds']) - settings.window_seconds) > 1e-9:
            raise ValueError('runtime window duration differs from the model training setting')
        self.speed_hz = float(snapshot['estimated_speed_hz'])
        self.bearing_geometry = snapshot.get('bearing_geometry')
        family_count = self.manifest.get('family_classes', self.manifest.get('classes_family', self.manifest.get('classes_8')))
        if (family_count is None or len(self.family['family_names']) != int(family_count)
                or len(self.detail['label_names'])!=int(self.manifest['detailed_classes'])):
            raise ValueError('loaded artifact class count differs from the training manifest')
        self.model_info = {
            'label_catalog': label_catalog(self.family['family_names'], self.detail['label_names']),
            'primary_task': self.manifest['primary_task'],
            'primary_output': self.manifest['primary_output'],
            'primary_validation_accuracy': self.manifest['mean_accuracy'][self.manifest['primary_task']],
            'detailed_validation_accuracy': self.manifest['mean_accuracy'].get(f"{len(self.detail['label_names'])}-detail"),
            'primary_recording_validation_accuracy': self.manifest.get('mean_recording_accuracy', {}).get(self.manifest['primary_task']),
            'detailed_recording_validation_accuracy': self.manifest.get('mean_recording_accuracy', {}).get(f"{len(self.detail['label_names'])}-detail"),
            'detailed_inference': self.manifest.get('detailed_inference', 'independent'),
            'family_recipe': self.manifest.get('family_recipe', 'baseline'),
            'detail_recipe': self.manifest.get('detail_recipe', 'baseline'),
            'family_classes': len(self.family['family_names']),
            'detailed_classes': len(self.detail['label_names']),
            'trained_file_paths': self.manifest['valid_files'],
            'distinct_recordings': self.family['n_unique_sensor_recordings'],
            'duplicate_copies': self.manifest['duplicate_copies_included'],
            'sample_rate_hz': settings.sample_rate_hz,
            'window_seconds': settings.window_seconds,
            'hop_seconds': settings.hop_seconds,
            'family_model_sha256': self.family_hash,
            'detail_model_sha256': self.detail_hash,
            'feature_config_sha256': self.manifest['config_snapshot_sha256'],
            'scores_calibrated': False,
        }
        self.novelty = None
        self.model_info['novelty_reference_sha256'] = None
        self.novelty_status = 'missing'
        run_dir = root / 'artifacts' / 'runs' / str(self.manifest['run_id'])
        versioned_report = run_dir / 'novelty_reference.json'
        if versioned_report.is_file():
            novelty_report_path = versioned_report
            novelty_path = run_dir / 'novelty_reference.joblib'
        else:
            novelty_report_path = root / 'reports' / 'novelty_reference.json'
            novelty_path = root / 'artifacts' / 'novelty_reference.joblib'
        if load_novelty and novelty_report_path.is_file() and novelty_path.is_file():
            novelty_report = json.loads(novelty_report_path.read_text(encoding='utf-8'))
            if hashlib.sha256(novelty_path.read_bytes()).hexdigest() != novelty_report['artifact_sha256']:
                raise ValueError('novelty reference hash differs from its report')
            if novelty_report['family_model_sha256'] != self.family_hash:
                self.novelty_status = 'stale_after_retraining'
            else:
                self.novelty = joblib.load(novelty_path)
                if (self.novelty['family_model_sha256'] != self.family_hash
                        or self.novelty['feature_names'] != list(self.family['original_features'])[:-1]):
                    raise ValueError('novelty feature schema differs from the active model')
                self.novelty_status = 'ready'
                self.model_info['novelty_reference_sha256'] = novelty_report['artifact_sha256']
        self.model_info['novelty_reference_available'] = self.novelty is not None
        self.model_info['novelty_status'] = self.novelty_status
        self.model_info['unknown_fault_detection_validated'] = False

    def novelty_score(self, windows) -> list[dict | None]:
        if self.novelty is None:
            return [None] * len(windows)
        original, names = extract_features(np.asarray(windows, dtype=np.float32),
                                           self.settings.sample_rate_hz,
                                           self.speed_hz, self.bearing_geometry)
        if list(names) != self.novelty['feature_names']:
            raise ValueError('novelty feature schema changed')
        scores = self.novelty['model'].decision_function(original)
        threshold = float(self.novelty['threshold'])
        return [{'score': float(score), 'threshold': threshold,
                 'outside_reference': bool(score < threshold),
                 'unknown_fault_detection_validated': False} for score in scores]

    def predict_probabilities(self, windows, elapsed_starts=None):
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim != 3 or windows.shape[1:] != (self.settings.window_samples, 6):
            raise ValueError(f'expected vibration windows of shape (N, {self.settings.window_samples}, 6)')
        original, names = extract_features(windows, self.settings.sample_rate_hz,
                                           self.speed_hz, self.bearing_geometry)
        if elapsed_starts is None:
            # Direct window-array callers provide consecutive windows from one recording.
            elapsed_starts = np.arange(len(windows)) * self.settings.hop_seconds
        original, names = append_elapsed_time_feature(original, names, elapsed_starts)
        extra, order_names = order_features(windows, self.settings.sample_rate_hz)
        if (list(names) != list(self.family['original_features'])
                or list(order_names) != list(self.family['order_features'])
                or list(names) != list(self.detail['feature_names'])):
            raise ValueError('live feature schema differs from the trained model schema')
        p_family = np.zeros((len(windows), len(self.family['family_names'])), dtype=np.float64)
        joined = np.column_stack((original, extra))
        for weight, model, features in zip(self.family['weights'], self.family['family_models'],
                                           (original, joined)):
            p_family += float(weight) * _aligned_probabilities(
                model, features, len(self.family['family_names']))
        p_detail = _aligned_probabilities(
            self.detail['model'], original, len(self.detail['label_names']))
        return p_family, p_detail

    def format_predictions(self, p_family, p_detail):
        output = []
        for index in range(len(p_family)):
            family = _ranked(p_family[index], self.family['family_names'])
            output.append({
                'family': family,
                'detailed_condition': _ranked_child(
                    p_detail[index], self.detail['label_names'], family['label']),
                'independent_detailed_condition': _ranked(
                    p_detail[index], self.detail['label_names']),
            })
        return output

    def rank_family(self, probabilities):
        return _ranked(probabilities, self.family['family_names'])

    def predict_windows(self, windows):
        return self.format_predictions(*self.predict_probabilities(windows))

    def predict_recording(self, timestamps, channels):
        times, windows, starts = recording_windows(timestamps, channels, self.settings)
        from .live_monitor import inspect_window, review_reasons
        quality = [inspect_window(window, self.settings.sample_rate_hz) for window in windows]
        valid = [index for index, item in enumerate(quality) if item['status'] == 'ok']
        predictions = [{'family': None, 'detailed_condition': None,
                        'decision': 'invalid_input',
                        'review_reasons': [issue['code'] for issue in quality[index]['issues']]}
                       for index in range(len(windows))]
        p_family = None
        p_detail = None
        if valid:
            p_family, p_detail = self.predict_probabilities(
                windows[valid], times[starts[valid]] - times[0])
            novelty = self.novelty_score(windows[valid])
            for position, index in enumerate(valid):
                item = self.format_predictions(p_family[position:position + 1],
                                               p_detail[position:position + 1])[0]
                beyond = (self.manifest.get('training_windows_per_file', 0) > 0
                          and index >= self.manifest['training_windows_per_file'])
                reasons = review_reasons(item['family'], item['detailed_condition'],
                                         independent_detail=item['independent_detailed_condition'],
                                         beyond_budget=beyond, mapping_verified=False,
                                         equipment_validated=False)
                item['novelty'] = novelty[position]
                if novelty[position] is None:
                    reasons.append('novelty_reference_unavailable')
                if novelty[position] and novelty[position]['outside_reference']:
                    reasons.append('outside_training_feature_reference')
                item['decision'] = 'needs_review' if reasons else 'classified'
                item['review_reasons'] = reasons
                predictions[index] = item
        for index, (prediction, start) in enumerate(zip(predictions, starts)):
            prediction['window_index'] = index
            prediction['start_seconds'] = float(times[start])
            prediction['end_seconds'] = float(times[start + self.settings.window_samples - 1])
            prediction['quality'] = quality[index]
        recording_family = self.rank_family(p_family.mean(axis=0)) if p_family is not None else None
        independent_recording_detail = (_ranked(p_detail.mean(axis=0), self.detail['label_names'])
                                        if p_detail is not None else None)
        recording_detail = (_ranked_child(p_detail.mean(axis=0), self.detail['label_names'],
                                          recording_family['label']) if p_detail is not None else None)
        aggregate_disagreement = bool(recording_family and independent_recording_detail and
                                      independent_recording_detail['label'].split('__', 1)[0]
                                      != recording_family['label'])
        recording_decision = ('invalid_input' if not valid else
                              'needs_review' if aggregate_disagreement or
                              any(item['decision'] != 'classified' for item in predictions)
                              else 'classified')
        return {
            'window_count': len(predictions),
            'valid_window_count': len(valid),
            'recording_duration_seconds': float(times[-1] - times[0] +
                                                1 / self.settings.sample_rate_hz),
            'recording_time_basis': 'sensor_seconds_clock_not_verified',
            'predictions': predictions,
            'recording_decision': recording_decision,
            'recording_review_reasons': sorted(
                {reason for item in predictions for reason in item.get('review_reasons', [])}
                | ({'family_detail_disagree'} if aggregate_disagreement else set())),
            'input_verification': {'physical_axis_mapping': 'unverified',
                                   'equipment_domain': 'unvalidated'},
            'recording_family': recording_family,
            'recording_detailed_condition': recording_detail,
            'independent_recording_detailed_condition': independent_recording_detail,
            'family_probability_mean': (
                {name: float(score) for name, score in zip(
                    self.family['family_names'], p_family.mean(axis=0))}
                if p_family is not None else {}),
            'detailed_probability_mean': (
                {name: float(score) for name, score in zip(
                    self.detail['label_names'], p_detail.mean(axis=0))}
                if p_detail is not None else {}),
            'score_note': 'Model scores are uncalibrated; review flags are conservative heuristics.',
        }
