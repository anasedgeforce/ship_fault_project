"""Metadata for invalidating feature caches when data root or key window settings change."""
import hashlib
import json
from pathlib import Path

import numpy as np

def cache_key(config):
    root=Path(config['data_root']).resolve()
    files=sorted(root.glob('data_day*/**/*.csv'))
    fingerprint=hashlib.sha256()
    for path in files:
        stat=path.stat()
        fingerprint.update(f'{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n'.encode())
    return dict(data_root=str(root),file_count=len(files),file_metadata_sha256=fingerprint.hexdigest(),
                window_seconds=float(config['window_seconds']),hop_fraction=float(config['hop_fraction']),
                run_speed_hz=config.get('run_speed_hz'),bearing_geometry=config.get('bearing_geometry'),
                time_feature=config.get('time_feature'),expected_channels=config.get('expected_channels'),
                max_nan_fraction=config.get('max_nan_fraction'),
                max_timestamp_irregular_fraction=config.get('max_timestamp_irregular_fraction'),
                required_sample_rate_hz=config.get('required_sample_rate_hz'),
                excluded_files=sorted(str(item).replace('\\','/') for item in config.get('excluded_files',[])),
                excluded_sensor_sha256=sorted(config.get('excluded_sensor_sha256',[])))

def fresh_cache(project,config):
    project=Path(project);p=project/'cache'/'cache_manifest.json'
    required=['cache/window_features.npz','cache/round5_order_features.npz','cache/window_metadata.csv',
              'cache/data_settings.json','reports/file_hashes.csv','reports/duplicate_recordings.csv','reports/data_audit.csv']
    if not p.exists() or any(not (project/path).is_file() for path in required):return False
    try:
        if json.loads(p.read_text()) != cache_key(config):return False
        with np.load(project/'cache'/'window_features.npz') as base, np.load(project/'cache'/'round5_order_features.npz') as order:
            return len(base['X'])==len(order['X']) and base['X'].shape[1]==len(base['names'])
    except (OSError,ValueError,KeyError):return False

def mark_cache(project,config):
    p=Path(project)/'cache'/'cache_manifest.json'
    p.write_text(json.dumps(cache_key(config),indent=2))
