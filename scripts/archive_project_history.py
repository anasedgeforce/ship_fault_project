"""Hash-verified move of retired project history to the sibling archive.

Run with --dry-run first. --apply copies each file, verifies SHA256, then
unlinks only that verified source. Re-running resumes an interrupted cleanup.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT.parent / 'ship_fault_project_archive'
ACTIVE_RUN = '20261104T041252Z_164da514'
KEEP_SRC = {
    '__init__.py', 'cache_utils.py', 'crossday_advanced.py', 'data_loader.py',
    'deduplicate.py', 'features.py', 'labels.py', 'model_recipes.py', 'models.py',
    'preprocessing.py', 'round4_experiments.py', 'round5_order_benchmark.py',
    'round5_order_features.py', 'settings.py', 'train.py', 'train_with_duplicates.py',
}
KEEP_SCRIPTS = {
    'archive_project_history.py', 'audit_live_connectivity.py', 'init_mongo.py',
    'init_training_mongo.py', 'replay_csv_to_mongo.py', 'train_from_mongo.py',
    'train_novelty_reference.py', 'validate_continuous_stream.py',
}
KEEP_TESTS = {
    'test_external_training.py', 'test_live_mongo_integration.py',
    'test_live_mongo_pipeline.py', 'test_live_monitor.py', 'test_manual_inputs.py',
    'test_mongo_training.py', 'test_presentation.py', 'test_serving.py',
    'test_training_workflow.py', 'test_validation_inventory.py', 'test_web_ui.py',
    'known_labels_live_node.js', 'prediction_summary_node.js',
    'upload_aggregation_node.js', 'upload_selection_node.js', '__init__.py',
}
KEEP_DOCS = {
    'digital_twin_fault_mapping.csv', 'label_catalog.json', 'LABEL_CATALOG.md',
    'LIVE_MONGO_GUIDE.md', 'MONGO_RETRAINING.md', 'WEB_GUIDE.md',
}
KEEP_REPORTS = {
    'holdout_files.txt', 'latest_training.md', 'novelty_reference.json',
    'cleanup_archive_manifest.jsonl',
}
KEEP_DEMO = {'synthetic_unseen_impulsive.csv', 'synthetic_unseen_smooth.csv'}
WHOLE_DIRS = {
    'archive', 'data_quarantine', 'duplicate_quarantine', 'evaluation_data',
    'examples', 'manager_demo', 'manual_comparison', 'unseen_test_data',
    'ship_fault_inference.egg-info',
}
WHOLE_FILES = {'sim.zip', 'train_all.bat', 'run_all.sh', 'run_audit.py'}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _files(path: Path):
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (p for p in path.rglob('*') if p.is_file())


def inventory() -> list[Path]:
    active_pointer = PROJECT / 'reports/duplicate_inclusive/manifest.json'
    active_id = json.loads(active_pointer.read_text(encoding='utf-8'))['run_id']
    if active_id != ACTIVE_RUN:
        raise ValueError('This one-time cleanup plan is not valid after the active model changes')
    files: set[Path] = set()

    def add(path: Path):
        files.update(_files(path))

    for name in WHOLE_DIRS | WHOLE_FILES:
        add(PROJECT / name)
    for root_name, keep in (
        ('src', KEEP_SRC), ('scripts', KEEP_SCRIPTS), ('tests', KEEP_TESTS),
        ('docs', KEEP_DOCS), ('demo_inputs', KEEP_DEMO),
    ):
        root = PROJECT / root_name
        for child in root.iterdir():
            if child.name not in keep and child.name != '__pycache__':
                add(child)
    add(PROJECT / 'ship_fault/desktop.py')
    for child in (PROJECT / 'artifacts').iterdir():
        if child.name == 'runs':
            for run in child.iterdir():
                if run.name != ACTIVE_RUN:
                    add(run)
        elif child.name != 'novelty_reference.joblib':
            add(child)
    for child in (PROJECT / 'reports').iterdir():
        if child.name in KEEP_REPORTS:
            continue
        if child.name == 'duplicate_inclusive':
            for report in child.iterdir():
                if report.name != 'manifest.json':
                    add(report)
        else:
            add(child)
    for child in (PROJECT / 'data').iterdir():
        if child.is_dir() and child.name not in {'data_day1', 'data_day2', 'data_day3', 'unseen_test'}:
            add(child)
    for log in (PROJECT / 'cache').glob('*.log'):
        add(log)

    index = PROJECT / 'artifacts/runs' / ACTIVE_RUN / 'validation_files.csv'
    with index.open(encoding='utf-8-sig', newline='') as stream:
        saved = {row['file'].replace('\\', '/') for row in csv.DictReader(stream)}
    day_files = [p for day in ('data_day1', 'data_day2', 'data_day3')
                 for p in (PROJECT / 'data' / day).rglob('*.csv')]
    if len(saved) != 482 or not 482 <= len(day_files) <= 670:
        raise ValueError(f'Unexpected data inventory: {len(saved)} saved, {len(day_files)} original')
    actual = {p.relative_to(PROJECT / 'data').as_posix() for p in day_files}
    if not saved <= actual:
        raise ValueError('A Saved Recording is missing; refusing to archive original CSVs')
    files.update(p for p in day_files if p.relative_to(PROJECT / 'data').as_posix() not in saved)
    return sorted(files)


def apply(files: list[Path]) -> None:
    journal = PROJECT / 'reports/cleanup_archive_manifest.jsonl'
    archive_root = ARCHIVE.resolve()
    if archive_root != PROJECT.parent.resolve() / 'ship_fault_project_archive':
        raise ValueError('Archive destination changed unexpectedly')
    locked = []
    for number, source in enumerate(files, 1):
        if source.is_symlink() or not _under(source, PROJECT):
            raise ValueError(f'Unsafe source path: {source}')
        relative = source.relative_to(PROJECT)
        dest = ARCHIVE / relative
        if not _under(dest, ARCHIVE):
            raise ValueError(f'Unsafe archive destination: {dest}')
        source_hash = _hash(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _hash(dest) != source_hash:
                raise ValueError(f'Existing archive copy differs: {dest}')
        else:
            pending = dest.with_name(dest.name + '.partial')
            if pending.exists():
                pending.unlink()
            shutil.copy2(source, pending)
            if _hash(pending) != source_hash:
                raise ValueError(f'Archive copy hash mismatch: {relative}')
            pending.replace(dest)
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'file': relative.as_posix(),
                                     'sha256': source_hash, 'bytes': source.stat().st_size}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            source.unlink()
        except PermissionError:
            locked.append(relative.as_posix())
            print(f'Working copy is locked; retained: {relative}', flush=True)
        if number % 50 == 0 or number == len(files):
            print(f'Archived and verified {number}/{len(files)} files', flush=True)
    if locked:
        print(f'{len(locked)} locked working copies remain: {locked}', flush=True)
    # Only remove empty directories within the checked workspace.
    for name in WHOLE_DIRS | {'src', 'scripts', 'tests', 'docs', 'demo_inputs', 'reports', 'artifacts', 'data'}:
        root = PROJECT / name
        if root.is_dir() and _under(root, PROJECT):
            for path in sorted((p for p in root.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                if not path.is_symlink() and _under(path, PROJECT):
                    try:
                        path.rmdir()
                    except OSError:
                        pass
            try:
                root.rmdir()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        journal = PROJECT / 'reports/cleanup_archive_manifest.jsonl'
        rows = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
        for row in rows:
            path = ARCHIVE / row['file']
            if not _under(path, ARCHIVE) or _hash(path) != row['sha256']:
                raise ValueError(f'Archived file changed: {row["file"]}')
        print(f'Verified {len(rows)} archived files')
        return
    files = inventory()
    bytes_total = sum(path.stat().st_size for path in files)
    print(f'{len(files)} files, {bytes_total / (1024 ** 3):.2f} GiB to archive at {ARCHIVE}')
    if args.apply:
        apply(files)
    elif not args.dry_run:
        parser.error('Choose --dry-run or --apply')


if __name__ == '__main__':
    main()
