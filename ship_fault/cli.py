"""Local manual inference, CSV replay, and single-worker HTTP serving."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen

from .models import InferenceEngine
from .settings import RuntimeSettings
from .signal import parse_csv_recording
from .stream import StreamSession


def _read_limited(path: Path, maximum: int) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > maximum:
        raise ValueError(f'input exceeds {maximum} bytes')
    return path.read_bytes()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=RuntimeSettings.from_env().project_root)
    commands = parser.add_subparsers(dest='command', required=True)
    csv_command = commands.add_parser('predict-csv', help='classify a local seven-column CSV recording')
    csv_command.add_argument('path', type=Path)
    json_command = commands.add_parser('predict-json', help='classify manually supplied timestamp and channel arrays')
    json_command.add_argument('path', type=Path)
    replay_command = commands.add_parser('replay-csv', help='send a CSV to the same rolling stream used by live sensors')
    replay_command.add_argument('path', type=Path)
    replay_command.add_argument('--batch-size', type=int, default=2000)
    server_command = commands.add_parser('serve', help='start the local HTTP service')
    server_command.add_argument('--host', default='127.0.0.1')
    server_command.add_argument('--port', type=int, default=8000)
    server_command.add_argument('--input-mode', choices=('both', 'manual', 'live'), default='both',
                                help='restrict the server to one prediction input mode')
    server_command.add_argument('--no-access-log', action='store_true', help='hide routine HTTP request logs')
    server_command.add_argument('--open-browser', action='store_true', help='open the local prediction page when ready')
    args = parser.parse_args(argv)
    settings = RuntimeSettings(project_root=args.project,
                               api_key=os.environ.get('SHIP_FAULT_API_KEY') or None)
    if args.command == 'serve':
        os.environ['SHIP_FAULT_PROJECT_DIR'] = str(settings.project_root)
        os.environ['SHIP_FAULT_INPUT_MODE'] = args.input_mode
        if args.open_browser:
            def open_when_ready():
                address = f'http://127.0.0.1:{args.port}/'
                for _ in range(120):
                    try:
                        with urlopen(address + 'health/ready', timeout=1):
                            webbrowser.open(address)
                            return
                    except (OSError, URLError):
                        time.sleep(.25)
            threading.Thread(target=open_when_ready, daemon=True).start()
        import uvicorn
        uvicorn.run('ship_fault.api:app', host=args.host, port=args.port, workers=1,
                    access_log=not args.no_access_log)
        return
    engine = InferenceEngine(settings)
    if args.command in ('predict-csv', 'replay-csv'):
        timestamps, values = parse_csv_recording(_read_limited(args.path, settings.max_request_bytes), settings)
        if args.command == 'predict-csv':
            result = engine.predict_recording(timestamps, values)
        else:
            if not 1 <= args.batch_size <= settings.max_batch_samples:
                parser.error(f'--batch-size must be 1 to {settings.max_batch_samples}')
            session = StreamSession(engine)
            predictions = []
            for start in range(0, len(timestamps), args.batch_size):
                end = start + args.batch_size
                predictions.extend(session.push_batch(timestamps[start:end], values[start:end])['predictions'])
            result = {'accepted_samples': len(timestamps), 'window_count': len(predictions),
                      'predictions': predictions}
    else:
        payload = json.loads(_read_limited(args.path, settings.max_request_bytes).decode('utf-8'))
        result = engine.predict_recording(payload['timestamps'], payload['channels'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
