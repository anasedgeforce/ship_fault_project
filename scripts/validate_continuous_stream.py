"""Replay separately labeled long recordings and report early/late window behavior.

Manifest columns: path,fault_family,fault_subtype. Paths are relative to the
manifest. These recordings must be genuinely new runs, not windows cut from
the model's training captures.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from ship_fault.models import InferenceEngine
from ship_fault.settings import CHANNELS, CSV_COLUMNS, RuntimeSettings
from ship_fault.stream import StreamSession


def summarize(rows: list[dict]) -> dict:
    return {
        'windows': len(rows),
        'candidate_family_accuracy': (sum(row['family_correct'] for row in rows) / len(rows)
                                      if rows else None),
        'candidate_subtype_accuracy': (sum(row['subtype_correct'] for row in rows) / len(rows)
                                       if rows else None),
        'classified_coverage': (sum(row['decision'] == 'classified' for row in rows) / len(rows)
                                if rows else None),
    }


def replay(manifest: Path) -> dict:
    settings = RuntimeSettings()
    engine = InferenceEngine(settings)
    all_rows = []
    recordings = []
    with manifest.open(newline='', encoding='utf-8-sig') as handle:
        entries = list(csv.DictReader(handle))
    if not entries or not {'path', 'fault_family', 'fault_subtype'} <= set(entries[0]):
        raise ValueError('manifest needs path,fault_family,fault_subtype columns and at least one row')
    for entry in entries:
        path = (manifest.parent / entry['path']).resolve()
        session = StreamSession(engine, path.stem)
        source = {'equipment_id': None, 'channel_names': list(CHANNELS),
                  'channel_units': 'mm/s', 'mapping_status': 'declared_axes_unverified_wiring',
                  'mongo': None}
        count = 0
        for chunk in pd.read_csv(path, chunksize=4000, encoding='utf-8-sig'):
            if tuple(chunk.columns) != CSV_COLUMNS:
                raise ValueError(f'{path}: expected Time and six trained channels in order')
            count += len(chunk)
            result = session.push_batch(chunk.iloc[:, 0], chunk.iloc[:, 1:], source)
            for prediction in result['predictions']:
                all_rows.append({
                    'path': str(path), 'window_index': prediction['window_index'],
                    'late': prediction['beyond_training_window_budget'],
                    'decision': prediction['decision'],
                    'family_correct': bool(prediction['family'] and
                                           prediction['family']['label'] == entry['fault_family']),
                    'subtype_correct': bool(prediction['detailed_condition'] and
                                            prediction['detailed_condition']['label'] == entry['fault_subtype']),
                })
        recordings.append({'path': str(path), 'samples': count,
                           'windows': session.latest['window_index'] + 1 if session.latest else 0})
    early = [row for row in all_rows if not row['late']]
    late = [row for row in all_rows if row['late']]
    return {'status': 'descriptive_replay_only' if late else 'insufficient_long_recordings',
            'note': 'Overlapping windows are correlated. Use new labeled runs from unseen days to establish continuous-operation accuracy.',
            'recordings': recordings, 'early': summarize(early), 'late': summarize(late),
            'model_sha256': engine.model_info['family_model_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = replay(args.manifest.resolve())
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
