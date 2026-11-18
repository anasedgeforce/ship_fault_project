"""Export normalized, explicitly labeled simulator captures for training."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
import pandas as pd
import yaml

from .settings import APP_EQUIPMENT_ID, CHANNELS
from src.labels import label_for


TRAINING_RECORD_TYPE = 'verified_training'
SAMPLES_PER_CAPTURE = 32_000  # Match the original eight-second recordings.
LOCAL_TIME = timezone(timedelta(hours=5, minutes=30))


def _condition_folders(project: Path) -> dict[str, str]:
    mapping = yaml.safe_load((project / 'config/label_map.yaml').read_text(encoding='utf-8'))
    if not isinstance(mapping, dict):
        raise ValueError('Training label map is missing or invalid')
    folders = {}
    for folder, label in sorted(mapping.items()):
        if label_for(folder) != label:
            raise ValueError(f'Condition folder does not match label: {folder}')
        folders.setdefault(label, folder)
    return folders


def _capture_day(document: dict) -> str:
    created = document.get('created_at')
    if not isinstance(created, datetime):
        raise ValueError('Verified training batch has no MongoDB created_at timestamp')
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.astimezone(LOCAL_TIME).date().isoformat()


def _decode(document: dict) -> tuple[np.ndarray, np.ndarray]:
    count = document.get('total_samples')
    if (not isinstance(count, int) or isinstance(count, bool) or count < 1
            or document.get('fs') != 4000 or document.get('dtype') != 'float32_le'
            or document.get('model_input_order') != list(CHANNELS)
            or document.get('declared_channel_names') != list(CHANNELS)
            or document.get('declared_channel_units') != 'mm/s'):
        raise ValueError('Verified training batch must have declared six-axis mm/s data at 4 kHz')
    raw_times = document.get('timestamps_bin')
    raw_channels = document.get('channel_data_bin')
    if not isinstance(raw_channels, dict) or raw_times is None or len(raw_times) != count * 8:
        raise ValueError('Verified training batch has damaged timestamps or channels')
    times = np.frombuffer(raw_times, dtype='<f8')
    axes = []
    for channel in CHANNELS:
        raw = raw_channels.get(channel)
        if raw is None or len(raw) != count * 4:
            raise ValueError(f'Verified training batch has damaged {channel} samples')
        axes.append(np.frombuffer(raw, dtype='<f4'))
    values = np.column_stack(axes)
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError('Verified training batch contains nonfinite readings')
    steps = np.diff(times)
    if len(steps) and (steps.min() <= 0 or np.max(np.abs(steps - .00025)) > .000005):
        raise ValueError('Verified training batch has timestamp gaps or the wrong sampling rate')
    return times, values


def _day_numbers(project: Path) -> dict[str, str]:
    path = project / 'reports/mongo_training_days.json'
    existing = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    if not isinstance(existing, dict) or any(not re.fullmatch(r'data_day[1-9][0-9]*', value)
                                             for value in existing.values()):
        raise ValueError('Mongo training day map is invalid')
    return existing


def _reserved_signal_hashes(project: Path) -> set[str]:
    manifest = project / 'data/unseen_test/source_manifest.json'
    if not manifest.is_file():
        return set()
    hashes = set()
    for row in json.loads(manifest.read_text(encoding='utf-8')):
        path = project / 'data/unseen_test' / row['file']
        if not path.is_file():
            raise FileNotFoundError(f'Reserved test recording missing: {path}')
        values = pd.read_csv(path, usecols=list(CHANNELS))[list(CHANNELS)].to_numpy(dtype='<f4')
        hashes.add(hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest())
    return hashes


def export_verified_batches(project: Path, documents, source_info: dict | None = None,
                            reserved_project: Path | None = None) -> dict:
    """Stage complete captures and publish them under new recording-day folders.

    ``live_label`` is an adapter-internal field containing verified source truth.
    Model predictions and ``condition_status`` are never training labels.
    """
    project = Path(project).resolve()
    data_root = project / 'data'
    folders = _condition_folders(project)
    reserved = _reserved_signal_hashes(Path(reserved_project) if reserved_project else project)
    day_map = _day_numbers(project)
    used_numbers = [int(p.name[8:]) for p in data_root.glob('data_day*')
                    if p.is_dir() and re.fullmatch(r'data_day[1-9][0-9]*', p.name)]
    next_number = max([*used_numbers, *(int(day[8:]) for day in day_map.values())], default=0) + 1
    pending = []
    staged_destinations = set()
    selected_batches = 0
    short_segments = 0
    stage_parent = project / 'cache'
    stage_parent.mkdir(exist_ok=True)
    stage = (stage_parent / f'mongo_training_{uuid.uuid4().hex}').resolve()
    if not stage.is_relative_to(stage_parent.resolve()):
        raise ValueError('Unsafe Mongo training stage path')
    stage.mkdir()
    buffer_times = np.empty(0, dtype='<f8')
    buffer_values = np.empty((0, 6), dtype='<f4')
    current = None
    seen_signals = {}

    def add_capture(times: np.ndarray, values: np.ndarray, key: tuple) -> None:
        nonlocal next_number
        session, label, date = key
        if np.any(np.std(values, axis=0) < 1e-6):
            raise ValueError(f'Flat sensor axis in verified capture: {session}')
        signal_hash = hashlib.sha256(np.ascontiguousarray(values, dtype='<f4').tobytes()).hexdigest()
        if signal_hash in reserved:
            raise ValueError('Reserved unseen_test signal cannot be imported for training')
        if signal_hash in seen_signals and seen_signals[signal_hash] != label:
            raise ValueError('Identical sensor capture has conflicting verified labels')
        seen_signals[signal_hash] = label
        if date not in day_map:
            day_map[date] = f'data_day{next_number}'
            next_number += 1
        # Keep paths below the Windows legacy path limit in this workspace.
        relative = Path(day_map[date]) / folders[label] / f'm_{signal_hash[:16]}.csv'
        if relative.as_posix() in staged_destinations:
            return
        staged_destinations.add(relative.as_posix())
        frame = pd.DataFrame(np.column_stack((times - times[0], values)),
                             columns=['Time', *CHANNELS])
        content = frame.to_csv(index=False, float_format='%.9g').encode('utf-8')
        staged = stage / f'{len(pending):08d}.csv'
        staged.write_bytes(content)
        pending.append((staged, data_root / relative, relative.as_posix(), label, signal_hash))

    try:
        for document in documents:
            if (document.get('record_type') != TRAINING_RECORD_TYPE
                    or document.get('equipment_id') != APP_EQUIPMENT_ID):
                continue
            label = document.get('live_label')
            if label not in folders:
                raise ValueError(f'Unknown verified subtype label: {label!r}')
            session = document.get('session_id')
            if not isinstance(session, str) or not session:
                raise ValueError('Verified training batch has no session_id')
            key = (session, label, _capture_day(document))
            times, values = _decode(document)
            selected_batches += 1
            if current != key or (len(buffer_times) and abs(times[0] - buffer_times[-1] - .00025) > .000005):
                if len(buffer_times):
                    short_segments += 1
                buffer_times = np.empty(0, dtype='<f8')
                buffer_values = np.empty((0, 6), dtype='<f4')
                current = key
            buffer_times = np.concatenate((buffer_times, times))
            buffer_values = np.concatenate((buffer_values, values))
            while len(buffer_times) >= SAMPLES_PER_CAPTURE:
                add_capture(buffer_times[:SAMPLES_PER_CAPTURE],
                            buffer_values[:SAMPLES_PER_CAPTURE], key)
                buffer_times = buffer_times[SAMPLES_PER_CAPTURE:]
                buffer_values = buffer_values[SAMPLES_PER_CAPTURE:]
        if len(buffer_times):
            short_segments += 1
        # Do not change the training tree if any selected batch fails validation.
        created = []
        for staged, destination, relative, label, signal_hash in pending:
            if destination.is_file():
                if destination.read_bytes() != staged.read_bytes():
                    raise ValueError(f'Existing Mongo training CSV changed: {relative}')
        for staged, destination, relative, label, signal_hash in pending:
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(destination)
            created.append(relative)
        if pending:
            (project / 'reports/mongo_training_days.json').write_text(
                json.dumps(day_map, indent=2) + '\n', encoding='utf-8')
        report = {'selected_batches': selected_batches, 'complete_captures': len(pending),
                  'new_csvs': len(created), 'short_segments_skipped': short_segments,
                  'training_days': day_map, 'new_files': created,
                  'labels_from': 'verified source subtype, never a model prediction'}
        if source_info:
            report['source'] = source_info
        (project / 'reports/mongo_training_import.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return report
    finally:
        if stage.is_relative_to(stage_parent.resolve()):
            shutil.rmtree(stage)
