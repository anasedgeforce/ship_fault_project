"""Small, explicit runtime contract for manual and live vibration inputs."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CHANNELS = (
    'DEV (mm/s)', 'DE H (mm/s)', 'DE A (mm/s)',
    'NDE V (mm/s)', 'NDE H (mm/s)', 'NDE A (mm/s)',
)
CSV_COLUMNS = ('Time', *CHANNELS)
APP_EQUIPMENT_ID = '8c088678ea386840f36fd10c'


@dataclass(frozen=True)
class RuntimeSettings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    sample_rate_hz: float = 4000.0
    sample_rate_tolerance: float = 0.01
    interval_tolerance: float = 0.02
    max_irregular_fraction: float = 0.01
    window_seconds: float = 1.0
    hop_seconds: float = 0.5
    max_recording_samples: int = 640_000
    max_batch_samples: int = 40_000
    max_request_bytes: int = 32_000_000
    max_streams: int = 32
    stream_idle_seconds: int = 1800
    api_key: str | None = None
    input_mode: str = 'both'
    validated_equipment_ids: tuple[str, ...] = ()
    equipment_id: str | None = None
    alert_consecutive_windows: int = 3
    mongo_enabled: bool = False
    mongo_uri: str = 'mongodb://127.0.0.1:27017'
    mongo_database: str = 'ship_fault_monitoring'

    def __post_init__(self):
        object.__setattr__(self, 'project_root', Path(self.project_root).resolve())
        if self.sample_rate_hz <= 0 or self.window_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError('sampling, window, and hop settings must be positive')
        if self.max_batch_samples <= 0 or self.max_recording_samples < self.window_samples:
            raise ValueError('invalid sample limits')
        if self.input_mode not in ('both', 'manual', 'live'):
            raise ValueError('input_mode must be both, manual, or live')
        if self.alert_consecutive_windows < 2:
            raise ValueError('alert_consecutive_windows must be at least two')

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.window_seconds))

    @property
    def hop_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.hop_seconds))

    @classmethod
    def from_env(cls) -> 'RuntimeSettings':
        root = os.environ.get('SHIP_FAULT_PROJECT_DIR')
        return cls(project_root=Path(root) if root else Path(__file__).resolve().parents[1],
                   api_key=os.environ.get('SHIP_FAULT_API_KEY') or None,
                   input_mode=os.environ.get('SHIP_FAULT_INPUT_MODE', 'both'),
                   mongo_enabled=os.environ.get('SHIP_FAULT_MONGO_ENABLED', '1') == '1',
                   mongo_uri=os.environ.get('SHIP_FAULT_MONGO_URI', 'mongodb://127.0.0.1:27017'),
                   mongo_database=os.environ.get('SHIP_FAULT_MONGO_DATABASE', 'ship_fault_monitoring'),
                   equipment_id=APP_EQUIPMENT_ID,
                   validated_equipment_ids=tuple(filter(None, (
                       item.strip() for item in os.environ.get(
                           'SHIP_FAULT_VALIDATED_EQUIPMENT_IDS', '').split(',')))))
