"""Validate and window velocity signals before model inference."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

from src.preprocessing import window_signal

from .settings import CSV_COLUMNS, RuntimeSettings


def validate_samples(timestamps, channels, settings: RuntimeSettings, *, minimum=1, maximum=None,
                     previous_timestamp=None):
    """Return finite float arrays in the trained channel order and at 4 kHz."""
    try:
        times = np.asarray(timestamps, dtype=np.float64)
        values = np.asarray(channels, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('timestamps and six vibration channels must be numeric') from exc
    upper = settings.max_recording_samples if maximum is None else maximum
    if times.ndim != 1 or not minimum <= len(times) <= upper:
        raise ValueError(f'expected {minimum} to {upper} timestamped samples')
    if values.shape != (len(times), 6):
        raise ValueError('each sample must have exactly six channels in the documented order')
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError('timestamps and sensor readings must be finite numbers')
    intervals = np.diff(times)
    if previous_timestamp is not None:
        intervals = np.concatenate(([times[0] - previous_timestamp], intervals))
    if np.any(intervals <= 0):
        raise ValueError('timestamps must strictly increase')
    if len(intervals):
        expected = 1.0 / settings.sample_rate_hz
        if abs(np.median(intervals) - expected) / expected > settings.sample_rate_tolerance:
            raise ValueError(f'expected {settings.sample_rate_hz:g} Hz velocity samples')
        if np.any(intervals > expected * 1.5):
            raise ValueError('timestamp gap detected; check missing samples before prediction')
        irregular = np.abs(intervals - expected) > expected * settings.interval_tolerance
        if float(np.mean(irregular)) > settings.max_irregular_fraction:
            raise ValueError('too many irregular timestamp intervals')
    converted = values.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError('sensor readings exceed the supported float32 range')
    return times, converted


def parse_csv_recording(content: bytes, settings: RuntimeSettings):
    if not content or len(content) > settings.max_request_bytes:
        raise ValueError('CSV is empty or exceeds the configured request size')
    try:
        frame = pd.read_csv(BytesIO(content), encoding='utf-8-sig')
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ValueError('CSV could not be parsed') from exc
    if tuple(frame.columns) != CSV_COLUMNS:
        raise ValueError('CSV must contain Time and the six velocity channels in the documented order')
    return validate_samples(frame.iloc[:, 0], frame.iloc[:, 1:], settings,
                            minimum=settings.window_samples)


def recording_windows(timestamps, channels, settings: RuntimeSettings):
    times, values = validate_samples(timestamps, channels, settings,
                                     minimum=settings.window_samples)
    windows = window_signal(values, settings.sample_rate_hz, settings.window_seconds,
                            settings.hop_seconds / settings.window_seconds)
    starts = np.arange(len(windows)) * settings.hop_samples
    return times, windows, starts
