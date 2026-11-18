"""Send a compatible seven-column CSV through the live API into this project's MongoDB."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from ship_fault.settings import APP_EQUIPMENT_ID, CHANNELS, RuntimeSettings
from ship_fault.presentation import LABELS
from ship_fault.signal import parse_csv_recording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--equipment-id', default=APP_EQUIPMENT_ID,
                        help='fixed application equipment ID')
    parser.add_argument('--stream-id', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--batch-size', type=int, default=2000)
    parser.add_argument('--record-type', default='csv_replay')
    parser.add_argument('--verified-label', help='true simulator subtype ID, for verified training captures only')
    args = parser.parse_args()
    if args.equipment_id != APP_EQUIPMENT_ID:
        parser.error(f'this application uses equipment ID {APP_EQUIPMENT_ID}')
    if (args.record_type == 'verified_training') != bool(args.verified_label):
        parser.error('--record-type verified_training and --verified-label must be supplied together')
    if args.verified_label and (args.verified_label not in LABELS or '__' not in args.verified_label):
        parser.error('--verified-label must be one of the 33 detailed condition IDs')
    settings = RuntimeSettings()
    if not 1 <= args.batch_size <= settings.max_batch_samples:
        parser.error(f'--batch-size must be 1-{settings.max_batch_samples}')
    times, values = parse_csv_recording(args.csv.read_bytes(), settings)
    source = {'equipment_id': args.equipment_id, 'channel_names': list(CHANNELS),
              'channel_units': 'mm/s', 'sample_rate_hz': settings.sample_rate_hz,
              'record_type': args.record_type}
    if args.verified_label:
        source['live_label'] = args.verified_label
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('SHIP_FAULT_API_KEY'):
        headers['X-API-Key'] = os.environ['SHIP_FAULT_API_KEY']
    endpoint = f"{args.url.rstrip('/')}/v1/streams/{args.stream_id}/samples"
    for start in range(0, len(times), args.batch_size):
        end = start + args.batch_size
        samples = [{'timestamp': float(t), 'channels': row.tolist()}
                   for t, row in zip(times[start:end], values[start:end])]
        request = Request(endpoint, data=json.dumps({'source': source, 'samples': samples}).encode(),
                          headers=headers, method='POST')
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
        last = result['predictions'][-1] if result['predictions'] else None
        reference = last['mongo_references'][-1] if last and last['mongo_references'] else None
        print(json.dumps({'accepted_samples': result['accepted_samples'],
                          'total_samples': result['total_samples'],
                          'persistence': result.get('persistence'),
                          'last_decision': last['decision'] if last else None,
                          'last_mongo_reference': reference,
                          'new_alerts': result['alerts']}, indent=2))


if __name__ == '__main__':
    main()
