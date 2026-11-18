"""Validate acquisition metadata without claiming to verify physical wiring."""
from __future__ import annotations

import re
import math

from .settings import CHANNELS
from .presentation import LABELS

_OBJECT_ID = re.compile(r'^[0-9a-fA-F]{24}$')
_NAME = re.compile(r'^[A-Za-z0-9_\-]{1,80}$')


def normalize_source(raw, expected_rate_hz: float) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError('source must be an object')
    allowed = {'equipment_id', 'channel_names', 'channel_units', 'sample_rate_hz',
               'cycle_key', 'live_label', 'record_type', 'source_timestamp_ms'}
    if set(raw) - allowed:
        raise ValueError(f'unknown source fields: {sorted(set(raw) - allowed)}')
    equipment_id = raw.get('equipment_id')
    if equipment_id is not None and (not isinstance(equipment_id, str)
                                     or not _OBJECT_ID.fullmatch(equipment_id)):
        raise ValueError('equipment_id must be a 24-character MongoDB ObjectId')
    names = raw.get('channel_names')
    if names is not None and names != list(CHANNELS):
        raise ValueError('channel_names must list the six trained axes in exact order; physical wiring still needs verification')
    units = raw.get('channel_units')
    if units is not None and units != 'mm/s':
        raise ValueError('velocity channel_units must be mm/s')
    sample_rate = raw.get('sample_rate_hz', expected_rate_hz)
    if (isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float))
            or not math.isfinite(sample_rate) or abs(sample_rate - expected_rate_hz) > 1e-6):
        raise ValueError(f'live samples must be {expected_rate_hz:g} Hz after any resampling')
    for key in ('live_label', 'record_type'):
        if raw.get(key) is not None and (not isinstance(raw[key], str) or not _NAME.fullmatch(raw[key])):
            raise ValueError(f'{key} must be a short identifier')
    for key in ('cycle_key', 'source_timestamp_ms'):
        if raw.get(key) is not None and (isinstance(raw[key], bool)
                                         or not isinstance(raw[key], int)
                                         or not -(2 ** 63) <= raw[key] < 2 ** 63):
            raise ValueError(f'{key} must be a signed 64-bit integer')
    if raw.get('record_type') == 'verified_training':
        if raw.get('live_label') not in LABELS or '__' not in raw['live_label']:
            raise ValueError('verified_training needs a known detailed condition in live_label')
        if names != list(CHANNELS) or units != 'mm/s':
            raise ValueError('verified_training needs declared six-axis names and mm/s units')
    return {'equipment_id': equipment_id.lower() if equipment_id else None, 'channel_names': names,
            'channel_units': units, 'sample_rate_hz': expected_rate_hz,
            'mapping_status': 'declared_axes_unverified_wiring' if names and units else 'unverified',
            'mongo': None, 'cycle_key': raw.get('cycle_key'),
            'live_label': raw.get('live_label'), 'record_type': raw.get('record_type'),
            'source_timestamp_ms': raw.get('source_timestamp_ms')}
