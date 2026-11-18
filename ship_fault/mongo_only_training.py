"""Isolated Mongo-only training workspace and validated artifact publication."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import yaml

from src.labels import label_for


WORKSPACE_NAME = 'mongo_runs'


def prepare_workspace(project: Path) -> Path:
    project = Path(project).resolve()
    workspace = (project / WORKSPACE_NAME / uuid.uuid4().hex[:8]).resolve()
    if not workspace.is_relative_to(project):
        raise ValueError('Mongo training workspace must stay inside the project')
    for folder in ('config', 'data', 'cache', 'reports', 'artifacts'):
        (workspace / folder).mkdir(parents=True, exist_ok=True)
    for name in ('config.yaml', 'label_map.yaml'):
        shutil.copy2(project / 'config' / name, workspace / 'config' / name)
    return workspace


def check_source_coverage(project: Path, workspace: Path) -> dict:
    """Require enough new, labeled captures to retain the existing catalog."""
    mapping = yaml.safe_load((project / 'config/label_map.yaml').read_text(encoding='utf-8'))
    required = set(mapping.values())
    files = sorted((workspace / 'data').glob('data_day*/**/*.csv'))
    actual = set()
    days = set()
    for path in files:
        relative = path.relative_to(workspace / 'data')
        days.add(relative.parts[0])
        actual.add(label_for('/'.join(relative.parts[1:-1])))
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f'Mongo-only training needs all {len(required)} known subtypes; '
                         f'{len(missing)} are missing. First missing: {missing[0]}')
    if len(days) < 2:
        raise ValueError('Mongo-only training needs recordings from at least two capture days')
    return {'csv_files': len(files), 'days': sorted(days), 'subtypes': len(actual)}


def publish_models(project: Path, workspace: Path) -> str:
    """Copy verified artifacts before changing the app's active model pointer."""
    from .models import InferenceEngine
    from .settings import RuntimeSettings

    project, workspace = Path(project).resolve(), Path(workspace).resolve()
    source_pointer = workspace / 'reports/duplicate_inclusive/manifest.json'
    manifest = json.loads(source_pointer.read_text(encoding='utf-8'))
    run_id = manifest['run_id']
    source_run = workspace / 'artifacts/runs' / run_id
    target_run = project / 'artifacts/runs' / run_id
    if target_run.exists():
        raise FileExistsError(f'Model run already exists in the main project: {run_id}')
    expected_subtypes = len(set(yaml.safe_load(
        (project / 'config/label_map.yaml').read_text(encoding='utf-8')).values()))
    if manifest['detailed_classes'] != expected_subtypes or manifest['family_classes'] != 6:
        raise ValueError('New models do not cover the existing six families and all known subtypes')
    InferenceEngine(RuntimeSettings(project_root=workspace), load_novelty=False)
    shutil.copytree(source_run, target_run)
    try:
        novelty = workspace / 'artifacts/novelty_reference.joblib'
        novelty_report = workspace / 'reports/novelty_reference.json'
        if novelty.is_file() != novelty_report.is_file():
            raise ValueError('New novelty artifact and report must be published together')
        if novelty.is_file():
            shutil.copy2(novelty, target_run / 'novelty_reference.joblib')
            shutil.copy2(novelty_report, target_run / 'novelty_reference.json')
        InferenceEngine(RuntimeSettings(project_root=project),
                        manifest_path=target_run / 'manifest.json')
        for source, destination in (
            (workspace / 'reports/latest_training.md', project / 'reports/latest_training.md'),
            (workspace / 'reports/mongo_training_import.json', project / 'reports/mongo_training_import.json'),
        ):
            if source.is_file():
                shutil.copy2(source, destination)
    except Exception:
        runs_root = (project / 'artifacts/runs').resolve()
        if target_run.resolve().is_relative_to(runs_root) and target_run != runs_root:
            shutil.rmtree(target_run)
        raise
    active = project / 'reports/duplicate_inclusive/manifest.json'
    active.parent.mkdir(parents=True, exist_ok=True)
    staged = active.with_suffix('.json.tmp')
    staged.write_bytes(source_pointer.read_bytes())
    staged.replace(active)
    return run_id
