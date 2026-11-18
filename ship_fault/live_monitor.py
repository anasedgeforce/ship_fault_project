"""Conservative signal checks and stream diagnostics for live inference."""
from __future__ import annotations

import numpy as np

from .settings import CHANNELS


def inspect_window(window: np.ndarray, sample_rate_hz: float) -> dict:
    """Inspect a one-second, six-axis velocity window before classification.

    Clipping is only suspected from repeated exact extrema. A sensor's actual
    voltage/velocity range is needed to prove clipping or wiring correctness.
    """
    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CHANNELS):
        raise ValueError('quality check requires six velocity channels')
    issues = []
    metrics = []
    for index, name in enumerate(CHANNELS):
        signal = values[:, index]
        rms = float(np.sqrt(np.mean(signal * signal)))
        peak = float(np.max(np.abs(signal)))
        std = float(np.std(signal))
        high, low = float(np.max(signal)), float(np.min(signal))
        extreme_fraction = float(max(np.mean(signal == high), np.mean(signal == low)))
        if std < 1e-6:
            issues.append({'code': 'flat_channel', 'channel': name,
                           'message': f'{name} is flat or nearly flat'})
        elif extreme_fraction >= 0.02 and peak > 0:
            issues.append({'code': 'possible_clipping', 'channel': name,
                           'message': f'{name} repeatedly reaches an exact extreme; check sensor range'})
        spectrum = np.abs(np.fft.rfft(signal - np.mean(signal)))
        dominant_bin = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0
        metrics.append({'channel': name, 'rms_mm_s': rms, 'peak_mm_s': peak,
                        'std_mm_s': std, 'dominant_hz': float(dominant_bin * sample_rate_hz / len(signal))})
    for first in range(len(CHANNELS)):
        for second in range(first + 1, len(CHANNELS)):
            if np.array_equal(values[:, first], values[:, second]):
                issues.append({'code': 'identical_channels', 'channel': CHANNELS[second],
                               'message': f'{CHANNELS[first]} and {CHANNELS[second]} are identical; check wiring'})
    return {'status': 'invalid' if issues else 'ok', 'issues': issues, 'channels': metrics}


def review_reasons(family: dict, detail: dict, *, independent_detail: dict | None = None, beyond_budget: bool,
                   mapping_verified: bool, equipment_validated: bool) -> list[str]:
    """Review flags are heuristics; tree probabilities are not calibrated."""
    reasons = []
    if beyond_budget:
        reasons.append('continuous_duration_unvalidated')
    if not mapping_verified:
        reasons.append('physical_axis_mapping_unverified')
    if not equipment_validated:
        reasons.append('equipment_domain_unvalidated')
    if (independent_detail or detail)['label'].split('__', 1)[0] != family['label']:
        reasons.append('family_detail_disagree')
    if family['score'] < 0.55 or detail['score'] < 0.40:
        reasons.append('low_model_score')
    return reasons
