"""Import labeled captures from an external MongoDB, then retrain both models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ship_fault.external_training import load_source_config, normalize_external_document
from ship_fault.mongo_training import export_verified_batches
from ship_fault.mongo_only_training import prepare_workspace, check_source_coverage, publish_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--import-only', action='store_true', help='import CSVs without retraining')
    args = parser.parse_args()
    project = args.project.resolve()
    try:
        source = load_source_config(project)
        client = MongoClient(source['uri'], serverSelectionTimeoutMS=3000)
        try:
            client.admin.command('ping')
            collection = client[source['database']][source['collection']]
            if source['collection'] not in client[source['database']].list_collection_names():
                raise ValueError(f"Training collection {source['database']}.{source['collection']} does not exist")
            if collection.count_documents({}) == 0:
                print(f"No training captures in {source['database']}.{source['collection']}; active models unchanged.")
                return 1
            workspace = prepare_workspace(project)
            fields = source['fields']
            documents = collection.find({}).sort(
                [(fields['session'], ASCENDING), (fields['created_at'], ASCENDING)])
            normalized = (normalize_external_document(document, fields) for document in documents)
            report = export_verified_batches(workspace, normalized, source_info={
                'database': source['database'], 'collection': source['collection']},
                reserved_project=project)
        finally:
            client.close()
    except (ValueError, FileNotFoundError, PyMongoError) as exc:
        print(f'External MongoDB training input is not ready: {exc}', file=sys.stderr, flush=True)
        return 2
    print(json.dumps(report, indent=2), flush=True)
    if not report['complete_captures']:
        print('No complete labeled eight-second captures found. Existing models were not changed.', flush=True)
        return 1
    if args.import_only:
        return 0
    try:
        print('MONGO-ONLY DATA:', json.dumps(check_source_coverage(project, workspace)), flush=True)
    except ValueError as exc:
        print(f'Training stopped: {exc}', file=sys.stderr, flush=True)
        return 2
    result = subprocess.call([sys.executable, '-u', str(project / 'run_train.py'),
                              '--project', str(workspace)], cwd=project)
    if result:
        return result
    print('Published Mongo-only model run:', publish_models(project, workspace), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
