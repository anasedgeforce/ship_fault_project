"""Fit an auxiliary waveform outlier reference for the current saved models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ship_fault.models import InferenceEngine
from ship_fault.settings import RuntimeSettings


def fit_novelty_reference(root: Path):
    settings = RuntimeSettings(project_root=root)
    root = settings.project_root
    engine = InferenceEngine(settings, load_novelty=False)
    feature_path = root / 'cache' / 'window_features.npz'
    metadata_path = root / 'cache' / 'window_metadata.csv'
    with np.load(feature_path) as source:
        names = source['names'].tolist()
        features = source['X']
    if names != list(engine.family['original_features']):
        raise ValueError('feature cache does not match the current model schema')
    if names[-1] != 'elapsed_window_start_s':
        raise ValueError('expected the elapsed-time feature last')
    values = np.asarray(features[:, :-1], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('outlier reference needs finite features')
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != len(values):
        raise ValueError('feature and metadata row counts differ')
    day_scores = {}
    held_out_scores = {}
    for day in sorted(metadata.day.unique()):
        held = (metadata.day == day).to_numpy()
        model = IsolationForest(n_estimators=80, max_samples=512,
                                random_state=29, n_jobs=1)
        model.fit(values[~held])
        scores = model.decision_function(values[held])
        held_out_scores[str(day)] = scores
        day_scores[str(day)] = {
            'windows': int(held.sum()),
            'held_out_score_p01': float(np.quantile(scores, 0.01)),
            'held_out_score_median': float(np.median(scores)),
        }
    threshold = min(item['held_out_score_p01'] for item in day_scores.values())
    for day, scores in held_out_scores.items():
        day_scores[day]['known_condition_review_fraction_at_threshold'] = float(np.mean(scores < threshold))
    final = IsolationForest(n_estimators=100, max_samples=512,
                            random_state=29, n_jobs=1)
    final.fit(values)
    final_scores = final.decision_function(values)
    artifact = {'model': final, 'threshold': threshold, 'feature_names': names[:-1],
                'family_model_sha256': engine.family_hash,
                'cache_sha256': hashlib.sha256(feature_path.read_bytes()).hexdigest()}
    model_path = root / 'artifacts' / 'novelty_reference.joblib'
    joblib.dump(artifact, model_path, compress=3)
    report = {
        'method': 'IsolationForest on 175 waveform features; elapsed time excluded',
        'threshold': threshold,
        'threshold_rule': 'minimum first percentile of known-condition held-out-day scores',
        'known_training_window_flag_fraction': float(np.mean(final_scores < threshold)),
        'held_out_day_score_summaries': day_scores,
        'unknown_fault_detection_validated': False,
        'family_model_sha256': engine.family_hash,
        'artifact_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'cache_sha256': artifact['cache_sha256'],
    }
    report_path = root / 'reports' / 'novelty_reference.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    raise SystemExit('Run train_from_mongo.bat to train and publish both models with a matching novelty reference.')


if __name__ == '__main__':
    main()
