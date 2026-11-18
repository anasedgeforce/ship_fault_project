"""Read-only catalog of recordings available for local prediction."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from src.labels import label_for
from .presentation import LABELS


def configured_data_root(project: Path) -> Path:
    from src.settings import load_config

    path = Path(load_config(project)['data_root'])
    return (path if path.is_absolute() else project / path).resolve()


def manual_recording_choices(project: Path) -> dict[str, dict[str, list[Path]]]:
    """Group available CSVs by day and condition for selectors."""
    project = Path(project).resolve()
    catalog: dict[str, dict[str, list[Path]]] = {}
    examples = sorted((project / 'examples').glob('*.csv')) if (project / 'examples').is_dir() else []
    if examples:
        catalog['Examples'] = {'Sample recordings': examples}
    root = configured_data_root(project)
    if root.is_dir():
        for day in sorted(root.glob('data_day*')):
            if not day.is_dir():
                continue
            conditions: dict[str, list[Path]] = {}
            for csv in sorted(day.rglob('*.csv')):
                condition = csv.parent.relative_to(day).as_posix()
                conditions.setdefault(condition, []).append(csv)
            if conditions:
                catalog[day.name] = conditions
    return catalog


@lru_cache(maxsize=4096)
def _matches_hash(path: str, size: int, mtime_ns: int, expected: str) -> bool:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == expected


def public_catalog(project: Path, manifest: dict) -> tuple[list[dict], dict[str, tuple[Path, str]]]:
    """List unchanged reference recordings, independent of future model runs."""
    project = Path(project).resolve()
    loaded_run = manifest.get('run_id')
    loaded_manifest_path = project / 'artifacts/runs' / str(loaded_run) / 'manifest.json'
    if (not loaded_manifest_path.is_file()
            or json.loads(loaded_manifest_path.read_text(encoding='utf-8')) != manifest):
        raise ValueError('evaluation list does not match the loaded model run')
    reference_path = project / 'config/saved_recordings.json'
    if reference_path.is_file():
        source_run_id = json.loads(reference_path.read_text(encoding='utf-8'))['source_run_id']
        reference_manifest = json.loads((project / 'artifacts/runs' / source_run_id / 'manifest.json').read_text(encoding='utf-8'))
    else:
        reference_manifest = manifest
    run_id = reference_manifest['run_id']
    relative_artifact = Path(str(reference_manifest['family_artifact']).replace('\\', '/'))
    run_dir = (project / relative_artifact).resolve().parent
    if not run_dir.is_relative_to((project / 'artifacts').resolve()):
        raise ValueError('evaluation index must be under the model artifacts directory')
    index_path = run_dir / 'validation_files.csv'
    if not index_path.is_file():
        # Earlier model runs exported this same inventory separately.
        index_path = project / 'evaluation_data' / 'validation_files.csv'
    if not index_path.is_file():
        raise ValueError('evaluation recording list is unavailable for this model run')
    with index_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row.get('run_id') != run_id for row in rows):
        raise ValueError('evaluation recording list does not match the loaded model run')
    with (run_dir / 'unseen_day_folds.csv').open(encoding='utf-8-sig', newline='') as stream:
        folds = list(csv.DictReader(stream))
    expected_counts = {row['day']: int(row['n_valid_files']) for row in folds
                       if row['task'] == reference_manifest['primary_task']}
    keys = [row['file'] for row in rows]
    if (not expected_counts or len(keys) != len(set(keys))
            or Counter(row['day'] for row in rows) != expected_counts):
        raise ValueError('evaluation recording list does not match the model fold counts')
    root = configured_data_root(project)
    choices: dict[str, dict[str, list[dict]]] = {}
    allowed: dict[str, tuple[Path, str]] = {}
    for row in rows:
        key = row['file'].replace('\\', '/')
        parts = key.split('/')
        expected = row['sha256']
        path = (root / key).resolve()
        if (len(parts) < 3 or parts[0] != row['day'] or not path.is_relative_to(root)
                or not re.fullmatch(r'[0-9a-f]{64}', expected)):
            raise ValueError('invalid recording entry in evaluation inventory')
        try:
            stat = path.stat()
            if not path.is_file() or not _matches_hash(str(path), stat.st_size, stat.st_mtime_ns, expected):
                continue
        except OSError:
            continue
        condition = '/'.join(parts[1:-1])
        choices.setdefault(parts[0], {}).setdefault(condition, []).append({'id': key, 'name': parts[-1]})
        allowed[key] = (path, expected)
    days = [{'name': day, 'display_name': re.sub(r'^data_day', 'Day ', day), 'conditions': [
        {'name': condition, 'display_name': LABELS[label_for(condition)], 'recordings': recordings}
        for condition, recordings in sorted(conditions.items())]}
        for day, conditions in sorted(choices.items())]
    return days, allowed
