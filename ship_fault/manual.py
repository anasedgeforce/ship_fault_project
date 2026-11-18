"""Parse a complete manually supplied vibration recording."""
from __future__ import annotations

from io import BytesIO
import json

import pandas as pd

from .settings import CSV_COLUMNS, RuntimeSettings
from .signal import parse_csv_recording, validate_samples


def parse_manual_payload(payload: bytes | str, settings: RuntimeSettings):
    raw = payload.encode('utf-8') if isinstance(payload, str) else payload
    if not raw or len(raw) > settings.max_request_bytes:
        raise ValueError('Recording is empty or exceeds the configured input limit.')
    first = raw.lstrip(b'\xef\xbb\xbf \t\r\n')
    if first.startswith((b'{', b'[')):
        try:
            value = json.loads(raw.decode('utf-8-sig'))
            if isinstance(value, dict) and 'samples' in value:
                rows = value['samples']
                timestamps = [item['timestamp'] for item in rows]
                channels = [item['channels'] for item in rows]
            elif isinstance(value, dict):
                timestamps, channels = value['timestamps'], value['channels']
            elif isinstance(value, list):
                timestamps = [row[0] for row in value]
                channels = [row[1:] for row in value]
            else:
                raise ValueError('JSON needs timestamps and channels, samples, or rows of seven numbers.')
        except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
            raise ValueError('JSON needs timestamps and six-channel readings.') from exc
        return validate_samples(timestamps, channels, settings, minimum=settings.window_samples)
    first_line = first.splitlines()[0] if first.splitlines() else b''
    if b'\t' in first_line and b',' not in first_line:
        try:
            frame = pd.read_csv(BytesIO(raw), sep='\t', encoding='utf-8-sig')
        except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
            raise ValueError('Could not read pasted spreadsheet rows.') from exc
        if tuple(frame.columns) != CSV_COLUMNS:
            raise ValueError('Pasted rows need Time followed by the six documented velocity channels.')
        return validate_samples(frame.iloc[:, 0], frame.iloc[:, 1:], settings, minimum=settings.window_samples)
    return parse_csv_recording(raw, settings)
