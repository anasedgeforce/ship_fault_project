"""Read-only adapter for labeled captures collected by another application."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import yaml

from .settings import APP_EQUIPMENT_ID, CHANNELS
from .mongo_training import TRAINING_RECORD_TYPE


def load_source_config(project: Path) -> dict:
    path = Path(project) / 'config/external_mongo_training.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError(f'Invalid external MongoDB configuration: {path}')
    database = os.environ.get('SHIP_FAULT_TRAINING_MONGO_DATABASE') or config.get('database')
    collection = os.environ.get('SHIP_FAULT_TRAINING_MONGO_COLLECTION') or config.get('collection')
    if not isinstance(database, str) or not database.strip() or not isinstance(collection, str) or not collection.strip():
        raise ValueError('Set database and collection in config/external_mongo_training.yaml')
    if database == (os.environ.get('SHIP_FAULT_MONGO_DATABASE') or 'ship_fault_monitoring'):
        raise ValueError('Training source must be a separate database from live prediction storage')
    fields = config.get('fields')
    if not isinstance(fields, dict) or any(not isinstance(fields.get(name), str) or not fields[name]
                                           for name in ('label', 'session', 'created_at', 'timestamps',
                                                        'channels', 'sample_rate', 'units')):
        raise ValueError('External MongoDB field mapping is incomplete')
    return {'database': database.strip(), 'collection': collection.strip(), 'fields': fields,
            'uri': os.environ.get('SHIP_FAULT_TRAINING_MONGO_URI') or
                   os.environ.get('SHIP_FAULT_MONGO_URI') or 'mongodb://127.0.0.1:27017'}


def _field(document: dict, path: str):
    result = document
    for part in path.split('.'):
        if not isinstance(result, dict) or part not in result:
            raise ValueError(f'External training document missing field {path!r}')
        result = result[part]
    return result


def normalize_external_document(document: dict, fields: dict) -> dict:
    """Convert documented array-based captures to the validated export format.

    `label` is the known simulator test condition, never a model prediction.
    Physical axes must already be mapped into the six named velocity channels.
    """
    label = _field(document, fields['label'])
    session = _field(document, fields['session'])
    created = _field(document, fields['created_at'])
    raw_times = _field(document, fields['timestamps'])
    raw_channels = _field(document, fields['channels'])
    sample_rate = _field(document, fields['sample_rate'])
    units = _field(document, fields['units'])
    if isinstance(created, str):
        try:
            created = datetime.fromisoformat(created.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('External created_at must be an ISO timestamp or MongoDB date') from exc
    if not isinstance(created, datetime):
        raise ValueError('External created_at must be an ISO timestamp or MongoDB date')
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    if not isinstance(label, str) or not isinstance(session, str) or not session:
        raise ValueError('External capture needs a known subtype label and session ID')
    if sample_rate != 4000 or units != 'mm/s':
        raise ValueError('External capture must declare 4 kHz velocity readings in mm/s')
    if not isinstance(raw_channels, dict) or set(raw_channels) != set(CHANNELS):
        raise ValueError('External capture needs the six named sensor axes')
    try:
        times = np.asarray(raw_times, dtype='<f8')
        values = [np.asarray(raw_channels[name], dtype='<f4') for name in CHANNELS]
    except (TypeError, ValueError) as exc:
        raise ValueError('External capture has nonnumeric timestamps or sensor readings') from exc
    if times.ndim != 1 or not len(times) or any(axis.ndim != 1 or len(axis) != len(times) for axis in values):
        raise ValueError('External capture has inconsistent sample lengths')
    return {
        'record_type': TRAINING_RECORD_TYPE,
        'equipment_id': APP_EQUIPMENT_ID,
        'session_id': session,
        'live_label': label,
        'created_at': created,
        'fs': 4000,
        'dtype': 'float32_le',
        'total_samples': len(times),
        'model_input_order': list(CHANNELS),
        'declared_channel_names': list(CHANNELS),
        'declared_channel_units': 'mm/s',
        'timestamps_bin': times.tobytes(),
        'channel_data_bin': {name: axis.tobytes() for name, axis in zip(CHANNELS, values)},
    }
