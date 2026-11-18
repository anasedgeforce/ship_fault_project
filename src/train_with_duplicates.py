"""Retrain exploratory fault classifiers including exact duplicate recordings.

The source CSVs already exist in data/. This script restores their feature rows
from the identical retained recordings. Evaluation uses unique recordings and
removes all matching SHA-256 groups from each validation fold's training set.
The final artifacts fit all valid files, including the duplicate copies.
"""
from __future__ import annotations

if __name__=='__main__' and not __package__:
    import runpy
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'run_train.py'),run_name='__main__')
    raise SystemExit

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from .features import TIME_FEATURE_NAME
from .cache_utils import fresh_cache
from .labels import attributes, label_for
from .model_recipes import family_model, detail_model as make_detail_model
from .settings import load_config
from .train import cv_take, load_cached, make_split


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def original_locked_holdout(project, unique_meta, hashes=None):
    """Preserve the prior fixed holdout when adding new recordings.

    Recomputing a seeded group shuffle after the file inventory changes can
    silently move unrelated recordings between train and validation.
    """
    path = project / 'reports' / 'holdout_files.txt'
    if not path.is_file():
        _, files = make_split(unique_meta)
        path.write_text('\n'.join(files)+'\n',encoding='utf-8')
    holdout_files = set(path.read_text(encoding='utf-8').splitlines())
    if hashes is None:
        inventory=pd.read_csv(project/'reports'/'file_hashes.csv',keep_default_na=False)
        hashes=dict(zip(inventory.file,inventory.sha256))
    missing = holdout_files - set(hashes)
    audit_path=project/'reports'/'data_audit.csv'
    reserved=set()
    if audit_path.is_file():
        audit=pd.read_csv(audit_path,keep_default_na=False)
        reserved=set(audit.loc[audit.status.eq('excluded_reserved'),'file'])
    missing -= reserved
    if missing:
        raise ValueError(f'{len(missing)} locked holdout files are missing from the data; restore them before retraining')
    held_hashes={hashes[file] for file in holdout_files if file in hashes}
    return unique_meta.file.map(hashes).isin(held_hashes).to_numpy(), sorted(holdout_files)


def prepare_cache(project, data_root=None, force=False):
    """Discover new recordings and refresh features before either model trains."""
    project=Path(project).resolve()
    config=load_config(project)
    root=Path(data_root if data_root is not None else config['data_root'])
    if not root.is_absolute():root=project/root
    root=root.resolve()
    config['data_root']=str(root)
    if (config.get('time_feature')!=TIME_FEATURE_NAME or config.get('run_speed_hz') is None
            or float(config['window_seconds'])!=1.0 or float(config['hop_fraction'])!=0.5
            or float(config.get('required_sample_rate_hz',0))!=4000.0):
        raise ValueError('active models require configured speed, elapsed time, 4 kHz, 1-second windows and 0.5-second hops')
    from ship_fault.settings import CHANNELS
    if config['expected_channels']!=list(CHANNELS):
        raise ValueError('training channel order must match the six runtime velocity channels')
    if not root.is_dir():raise FileNotFoundError(f'data directory not found: {root}')
    # Manual analysis fixtures live outside recording-day folders; never train on them.
    allowed=(set(root.glob('data_day*/**/*.csv'))
             | set((root/'test').rglob('*.csv'))
             | set((root/'test_data_orignal').rglob('*.csv'))
             | set((root/'unseen_test').rglob('*.csv')))
    unexpected=set(root.rglob('*.csv'))-allowed
    if unexpected:
        example=sorted(unexpected)[0].relative_to(root)
        raise ValueError(f'CSV outside a recording-day or test folder: {example}; use data_dayN/<condition>/*.csv or a designated test folder')
    stale=force or not fresh_cache(project,config)
    if not stale:
        inventory=pd.read_csv(project/'reports'/'file_hashes.csv',keep_default_na=False)
        stale=any(not (root/row.file).is_file() or _digest(root/row.file)!=row.sha256
                  for row in inventory.itertuples(index=False))
    if stale:
        print('DATA CHECK: new/changed data or settings detected; rebuilding features for both models',flush=True)
        from .data_loader import generate_cache
        generate_cache(project,config)
    else:print('DATA CHECK: raw hashes and feature cache are current',flush=True)
    return config,root


def restore_feature_rows(project, data_root):
    """Return valid unique data plus every currently audited exact copy."""
    config = load_config(project)
    config['data_root'] = str(data_root)
    if not fresh_cache(project, config):
        raise ValueError('feature cache is stale or incomplete; run python run_audit.py first')
    base, meta, names = load_cached(project)
    if names[-1] != TIME_FEATURE_NAME:
        raise ValueError('active training cache must include the Time-column window feature')
    with np.load(project / 'cache' / 'round5_order_features.npz') as data:
        order = data['X']
        order_names = data['names'].tolist()
    if len(base) != len(meta) or len(order) != len(meta):
        raise ValueError('cached feature arrays and metadata do not align')
    manifest = pd.read_csv(project / 'reports' / 'duplicate_recordings.csv', keep_default_na=False)
    hashes = pd.read_csv(project / 'reports' / 'file_hashes.csv', keep_default_na=False)
    stored = dict(zip(hashes.file, hashes.sha256))
    files = sorted(stored)
    live = {}
    for relative in files:
        path = data_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        live[relative] = _digest(path)
        if live[relative] != stored.get(relative):
            raise ValueError(f'raw file changed since feature audit: {relative}; rebuild cache first')
    pieces, extras, order_extras = [meta], [], []
    for row in manifest.itertuples(index=False):
        if row.file in set(meta.file):
            raise ValueError(f'duplicate already included in cache: {row.file}')
        if live[row.file] != live[row.original_file] or live[row.file] != row.sha256:
            raise ValueError(f'duplicate content mismatch: {row.file}')
        original_ids = np.flatnonzero(meta.file.to_numpy() == row.original_file)
        if not len(original_ids):
            raise ValueError(f'original missing from cache: {row.original_file}')
        relative = Path(row.file)
        day = relative.parts[0]
        folder = relative.parent.as_posix().split('/', 1)[1]
        label = label_for(folder)
        if label != meta.label.iloc[original_ids[0]]:
            raise ValueError(f'duplicate label conflict: {row.file}')
        extra = meta.iloc[original_ids].copy()
        extra['file'] = row.file
        extra['day'] = day
        extra['folder'] = folder
        for key, value in attributes(folder).items():
            extra[key] = value
        pieces.append(extra)
        extras.append(base[original_ids])
        order_extras.append(order[original_ids])
    all_meta = pd.concat(pieces, ignore_index=True)
    all_base = np.concatenate([base, *extras])
    all_order = np.concatenate([order, *order_extras])
    if len(all_meta) != len(all_base) or len(all_meta) != len(all_order):
        raise AssertionError('restored feature arrays misaligned')
    if all_meta.file.nunique()!=meta.file.nunique()+len(manifest):
        raise AssertionError('restored file inventory differs from the audit')
    return base, meta, all_base, all_order, all_meta, names, order_names, live


def _family_model(names, n_original, with_order, recipe='baseline'):
    if n_original != len(names):
        raise ValueError('feature names do not match the original feature matrix')
    return family_model(names, with_order, recipe)


def _family_probabilities(original, order, y, train, valid, names, classes, recipe='baseline'):
    probabilities = []
    models = []
    for with_order in (False, True):
        features = np.column_stack((original, order)) if with_order else original
        model = _family_model(names, original.shape[1], with_order, recipe)
        model.fit(features[train], y[train])
        scores = np.zeros((len(valid), classes), dtype=np.float64)
        scores[:, model.classes_.astype(int)] = model.predict_proba(features[valid])
        models.append(model)
        probabilities.append(scores)
    return (probabilities[0] + probabilities[1]) / 2, models


def _validation_inventory(unique_meta, holdout, hashes, folds, run_id, family_task):
    """Describe the unique recordings actually evaluated by completed folds."""
    evaluated_days = set(folds.loc[folds.task == family_task, 'day'])
    validation = unique_meta.loc[(~holdout) & unique_meta.day.isin(evaluated_days)].drop_duplicates('file').copy()
    validation['sha256'] = validation.file.map(hashes)
    validation = validation[['day', 'file', 'label', 'fault_type', 'sha256']].sort_values(['day', 'file'])
    validation.insert(0, 'run_id', run_id)
    expected_counts = dict(zip(folds.loc[folds.task == family_task, 'day'],
                               folds.loc[folds.task == family_task, 'n_valid_files']))
    if validation.sha256.isna().any() or validation.groupby('day').size().to_dict() != expected_counts:
        raise ValueError('validation recording inventory differs from the evaluated folds')
    return validation


@contextmanager
def _training_lock(project):
    """Release the operating-system lock automatically on process exit."""
    path=Path(project)/'cache'/'training.lock'
    path.parent.mkdir(exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            import os
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('another training run is already active') from exc
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle,fcntl.LOCK_UN)


def run(project='.', data_root=None, force_audit=False):
    project=Path(project).resolve()
    with _training_lock(project):
        config,data_root=prepare_cache(project,data_root,force_audit)
        return _train_models(project,data_root,config)


def _train_models(project, data_root, config):
    base, unique_meta, X, E, meta, names, order_names, hashes = restore_feature_rows(project, data_root)
    audit=pd.read_csv(project/'reports'/'data_audit.csv',keep_default_na=False)
    invalid=audit.loc[~audit.status.isin(['ok','excluded_duplicate','excluded_reserved']),['file','reason']]
    reserved_count=int(audit.status.eq('excluded_reserved').sum())
    print(f'DATA CHECK: {len(hashes)} raw CSVs; {len(invalid)} invalid recordings excluded; '
          f'{reserved_count} reserved sensor copies excluded; details in reports/data_audit.csv',flush=True)
    holdout, holdout_files = original_locked_holdout(project, unique_meta, hashes)
    data_settings=json.loads((project/'cache'/'data_settings.json').read_text(encoding='utf-8'))
    if (data_settings.get('speed_source')!='configured_reference'
            or float(data_settings['estimated_speed_hz'])!=float(config['run_speed_hz'])
            or data_settings.get('time_feature')!=TIME_FEATURE_NAME):
        raise ValueError('cache feature settings differ from the active training configuration')
    sample_rates=meta.fs_hz.unique()
    if not np.allclose(sample_rates,4000.0,rtol=0,atol=1e-3):
        raise ValueError('active models require 4 kHz recordings')
    if unique_meta.day.nunique()<2:
        raise ValueError('at least two recording days are needed for day-held-out evaluation')
    held_hashes = {hashes[file] for file in holdout_files}
    unique_hashes = unique_meta.file.map(hashes).to_numpy()
    all_hashes = meta.file.map(hashes).to_numpy()
    families = LabelEncoder().fit(meta.fault_type)
    detailed = LabelEncoder().fit(meta.label)
    detail_parents = np.array([np.flatnonzero(families.classes_ == label.split('__', 1)[0])[0]
                               for label in detailed.classes_])
    family_task=f'{len(families.classes_)}-family'
    detail_task=f'{len(detailed.classes_)}-detail'
    per_file=int(config.get('active_windows_per_file',15))
    trees=int(config.get('active_rf_trees',180))
    family_recipe=config.get('active_family_recipe','baseline')
    detail_recipe=config.get('active_detail_recipe','baseline')
    if per_file<1 or trees<1:raise ValueError('window and tree budgets must be positive')
    family_y = families.transform(meta.fault_type)
    detailed_y = detailed.transform(meta.label)
    unique_family_y = families.transform(unique_meta.fault_type)
    unique_detailed_y = detailed.transform(unique_meta.label)
    out = project / 'reports' / 'duplicate_inclusive'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    recording_rows, recall_rows, family_rows = [], [], []
    skipped=[]
    print(f'TRAINING: {meta.file.nunique()} valid paths, {len(set(meta.file.map(hashes)))} distinct recordings, '
          f'{len(families.classes_)} families, {len(detailed.classes_)} detailed conditions',flush=True)
    for outer_day in sorted(unique_meta.day.unique()):
        unique_valid_mask = (~holdout) & unique_meta.day.eq(outer_day).to_numpy()
        validation_hashes = set(unique_hashes[unique_valid_mask])
        train_mask = (meta.day.ne(outer_day).to_numpy()
                      & ~np.isin(all_hashes, list(held_hashes | validation_hashes)))
        train_families = cv_take(meta, train_mask, per_file)
        valid_families = np.flatnonzero(unique_valid_mask)
        if not len(train_families) or not len(valid_families):
            reason='no independent training or validation recordings after holdout/hash exclusion'
            skipped.append(dict(day=outer_day,reason=reason))
            print(f'SKIP {outer_day}: {reason}',flush=True)
            continue
        train_files = set(meta.file.iloc[train_families])
        valid_files = set(unique_meta.file.iloc[valid_families])
        if {hashes[x] for x in train_files} & {hashes[x] for x in valid_files}:
            raise AssertionError('same sensor recording leaked across family fold')
        # Family validation rows come from the first (unique) part of the cache.
        p, _ = _family_probabilities(
            X, E, family_y, train_families, valid_families, names, len(families.classes_), family_recipe)
        pred = p.argmax(axis=1)
        validation_files = unique_meta.file.iloc[valid_families].to_numpy()
        recording_groups = [np.flatnonzero(validation_files == file)
                            for file in pd.unique(validation_files)]
        recording_family_pred = [p[group].mean(axis=0).argmax() for group in recording_groups]
        recording_family_true = [unique_family_y[valid_families[group[0]]]
                                 for group in recording_groups]
        rows.append(dict(
            task=family_task, day=outer_day, accuracy=accuracy_score(unique_family_y[valid_families], pred),
            recording_accuracy=accuracy_score(recording_family_true, recording_family_pred),
            macro_f1=f1_score(unique_family_y[valid_families], pred,
                              labels=np.arange(len(families.classes_)), average='macro', zero_division=0),
            n_valid_windows=len(valid_families), n_valid_files=len(valid_files),
            n_train_files=len(train_files),
            n_train_duplicate_files=len(train_files & (set(meta.file) - set(unique_meta.file))),
            same_hash_train_valid=0))
        valid_details = np.flatnonzero(unique_valid_mask)
        train_details = cv_take(meta, train_mask, per_file)
        if {all_hashes[x] for x in train_details} & {unique_hashes[x] for x in valid_details}:
            raise AssertionError('same sensor recording leaked across detail fold')
        detail_model = make_detail_model(names, detail_recipe, trees)
        detail_model.fit(X[train_details], detailed_y[train_details])
        detail_scores = np.zeros((len(valid_details), len(detailed.classes_)), dtype=np.float64)
        detail_scores[:, detail_model.classes_.astype(int)] = detail_model.predict_proba(base[valid_details])
        independent_pred = detail_scores.argmax(axis=1)
        pred = np.where(detail_parents[None, :] == p.argmax(axis=1)[:, None],
                        detail_scores, -1).argmax(axis=1)
        recording_detail_pred = [np.where(detail_parents == family_pred,
                                          detail_scores[group].mean(axis=0), -1).argmax()
                                 for family_pred, group in zip(recording_family_pred, recording_groups)]
        recording_detail_true = [unique_detailed_y[valid_details[group[0]]]
                                 for group in recording_groups]
        rows.append(dict(
            task=detail_task, day=outer_day, accuracy=accuracy_score(unique_detailed_y[valid_details], pred),
            recording_accuracy=accuracy_score(recording_detail_true, recording_detail_pred),
            independent_detail_accuracy=accuracy_score(unique_detailed_y[valid_details], independent_pred),
            macro_f1=f1_score(unique_detailed_y[valid_details], pred,
                              labels=np.arange(len(detailed.classes_)), average='macro', zero_division=0),
            n_valid_windows=len(valid_details), n_valid_files=unique_meta.file.iloc[valid_details].nunique(),
            n_train_files=meta.file.iloc[train_details].nunique(),
            n_train_duplicate_files=len(set(meta.file.iloc[train_details]) - set(unique_meta.file)),
            same_hash_train_valid=0))
        for group, true_family, chosen_family, true_detail, chosen_detail in zip(
                recording_groups, recording_family_true, recording_family_pred,
                recording_detail_true, recording_detail_pred):
            recording_rows.append(dict(day=outer_day, file=validation_files[group[0]],
                actual_family=families.classes_[true_family], predicted_family=families.classes_[chosen_family],
                actual_detail=detailed.classes_[true_detail], predicted_detail=detailed.classes_[chosen_detail]))
        for class_id in np.unique(unique_detailed_y[valid_details]):
            mask = unique_detailed_y[valid_details] == class_id
            wrong, counts = np.unique(pred[mask & (pred != class_id)], return_counts=True)
            recall_rows.append(dict(day=outer_day, detailed_condition=detailed.classes_[class_id],
                seen_in_training_days=bool(class_id in detailed_y[train_details]),
                recordings=len(set(validation_files[mask])), windows=int(mask.sum()),
                correct_windows=int(np.sum(pred[mask] == class_id)),
                recall=float(np.mean(pred[mask] == class_id)),
                most_common_wrong_prediction=detailed.classes_[wrong[counts.argmax()]] if len(wrong) else '',
                most_common_wrong_windows=int(counts.max()) if len(counts) else 0))
        for class_id, family_name in enumerate(families.classes_):
            wm = unique_family_y[valid_families] == class_id
            fm = np.asarray(recording_family_true) == class_id
            family_rows.append(dict(day=outer_day, family=family_name, recordings=int(fm.sum()),
                family_recordings_correct=int(np.sum(np.asarray(recording_family_pred)[fm] == class_id)),
                detail_recordings_correct=int(np.sum(np.asarray(recording_detail_pred)[fm] == np.asarray(recording_detail_true)[fm])),
                windows=int(wm.sum()), family_windows_correct=int(np.sum(p.argmax(axis=1)[wm] == class_id)),
                detail_windows_correct=int(np.sum(pred[wm] == unique_detailed_y[valid_details][wm]))))
        print('DUPLICATE-INCLUSIVE', outer_day, 'family', rows[-2]['accuracy'],
              'detail', rows[-1]['accuracy'], flush=True)
    if not rows:raise ValueError('no independent day folds are available; models were not replaced')
    folds = pd.DataFrame(rows)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    run_dir=project/'artifacts'/'runs'/run_id
    run_dir.mkdir(parents=True)
    folds.to_csv(run_dir/'unseen_day_folds.csv',index=False)
    recording_results = pd.DataFrame(recording_rows).assign(run_id=run_id)
    detailed_recall = pd.DataFrame(recall_rows).assign(run_id=run_id)
    family_breakdown = pd.DataFrame(family_rows).assign(run_id=run_id)
    recording_results.to_csv(run_dir/'recording_predictions.csv',index=False)
    detailed_recall.to_csv(run_dir/'detailed_recall_by_day.csv',index=False)
    family_breakdown.to_csv(run_dir/'family_accuracy_by_day.csv',index=False)
    # Keep the exact distinct validation recordings with the model run so the
    # prediction page can list only the recordings actually used in its folds.
    validation = _validation_inventory(unique_meta, holdout, hashes, folds, run_id, family_task)
    validation.to_csv(run_dir/'validation_files.csv', index=False)

    # Match the window budget and estimator settings used in the validation folds.
    all_family_ids = cv_take(meta, np.ones(len(meta), dtype=bool), per_file)
    _, family_models = _family_probabilities(
        X, E, family_y, all_family_ids, all_family_ids[:1], names, len(families.classes_), family_recipe)
    family_path = run_dir / 'family.joblib'
    joblib.dump(dict(
        family_models=family_models, family_names=families.classes_.tolist(),
        original_features=names, order_features=order_names, weights=[.5, .5],
        n_train_files=meta.file.nunique(), n_unique_sensor_recordings=len(set(all_hashes)),
        holdout_excluded=False,
        status='EXPERIMENTAL: trained with exact duplicate recordings; new-day validation is exploratory',
    ), family_path, compress=3)
    detail_model = make_detail_model(names, detail_recipe, trees)
    detail_model.fit(X[all_family_ids], detailed_y[all_family_ids])
    detail_path = run_dir / 'detailed_conditions.joblib'
    joblib.dump(dict(model=detail_model, label_names=detailed.classes_.tolist(),
                     feature_names=names, trained_files=meta.file.nunique()), detail_path, compress=3)
    summary = folds.groupby('task')[['accuracy', 'macro_f1', 'recording_accuracy']].mean()
    summary.to_csv(run_dir/'unseen_day_summary.csv')
    feature_config = run_dir/'config_snapshot.json'
    snapshot=dict(config,estimated_speed_hz=data_settings['estimated_speed_hz'],
                  speed_source=data_settings['speed_source'],class_names=detailed.classes_.tolist())
    feature_config.write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    duplicate_count=meta.file.nunique()-unique_meta.file.nunique()
    manifest = dict(
        run_id=run_id,trained_at_utc=datetime.now(timezone.utc).isoformat(),
        raw_files=len(hashes), valid_files=meta.file.nunique(), duplicate_copies_included=duplicate_count,
        excluded_invalid_files=len(invalid),excluded_reserved_sensor_copies=reserved_count,
        recording_days=sorted(meta.day.unique()),skipped_days=skipped,
        windows=len(meta), classes_family=len(families.classes_), detailed_classes=len(detailed.classes_),
        family_classes=len(families.classes_),training_windows_per_file=per_file,detailed_trees=trees,
        family_recipe=family_recipe,detail_recipe=detail_recipe,
        evaluation='leave one recording day out; original fixed holdout excluded; validation unique files only; raw-hash overlap removed from training',
        final_training='all valid files including duplicate copies',
        primary_task=family_task, primary_output='predictions[].family',
        family_blend_weights=[.5, .5],
        detailed_inference='child chosen within predicted family; raw child score retained',
        family_artifact=str(family_path.relative_to(project)),
        detail_artifact=str(detail_path.relative_to(project)),
        feature_config_artifact=str(feature_config.relative_to(project)),
        family_sha256=_digest(family_path), detail_sha256=_digest(detail_path),
        config_snapshot_sha256=_digest(feature_config),
        raw_hash_inventory_sha256=_digest(project / 'reports' / 'file_hashes.csv'),
        fixed_holdout_sha256=_digest(project / 'reports' / 'holdout_files.txt'),
        speed_source=data_settings['speed_source'],
        time_feature=TIME_FEATURE_NAME,
        sample_rate_hz=4000.0,
        window_seconds=float(snapshot['window_seconds']),
        hop_seconds=float(snapshot['window_seconds'] * snapshot['hop_fraction']),
        expected_channels=snapshot['expected_channels'],
        mean_accuracy={task:float(row.accuracy) for task,row in summary.iterrows()},
        mean_recording_accuracy={task:float(row.recording_accuracy) for task,row in summary.iterrows()},
    )
    (run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from ship_fault.models import InferenceEngine
    from ship_fault.settings import RuntimeSettings
    candidate=InferenceEngine(RuntimeSettings(project_root=project),manifest_path=run_dir/'manifest.json')
    sample=pd.read_csv(data_root/meta.file.iloc[0],nrows=4000)
    candidate.predict_recording(sample['Time'],sample[config['expected_channels']])
    if not fresh_cache(project,config) or any(_digest(data_root/file)!=digest for file,digest in hashes.items()):
        raise RuntimeError('raw data changed during training; rerun training to include the latest recordings')
    # Publish one pointer only after both artifacts and configuration are complete.
    temporary=out/'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    temporary.replace(out/'manifest.json')
    folds.to_csv(out/'unseen_day_folds.csv',index=False)
    summary.to_csv(out/'unseen_day_summary.csv')
    family_breakdown.to_csv(project/'reports'/'family_accuracy_by_day.csv',index=False)
    detailed_recall.loc[detailed_recall.day.eq('data_day2')].sort_values('recall').to_csv(
        out/'day2_detail_recall.csv',index=False)
    display_folds=folds[['task','day','accuracy','recording_accuracy','macro_f1']].copy()
    display_summary=summary.copy()
    for column in ['accuracy','recording_accuracy','macro_f1']:
        display_folds[column]=display_folds[column].map(lambda value:f'{value:.2%}')
        display_summary[column]=display_summary[column].map(lambda value:f'{value:.2%}')
    report=['# Latest training results','',f'Run: `{run_id}`',
            f'Raw CSVs: {len(hashes)}; valid paths: {meta.file.nunique()}; distinct recordings: {unique_meta.file.nunique()}; '
            f'exact copy paths: {duplicate_count}; invalid exclusions: {len(invalid)}.',
            '',display_folds.to_markdown(index=False),
            '', 'Equal-day means:',display_summary.to_markdown(),'',
            f'Both model families use up to {per_file} evenly spaced windows per file in training; '
            f'the detailed Random Forest uses {trees} trees in both evaluation and the final fit.',
            f'Model recipes: family `{family_recipe}`, detailed `{detail_recipe}`.',
            'Validation uses all complete windows of unique held-out recordings. The original locked holdout '
            'and matching raw hashes are excluded from every fold training set.',
            'Detailed predictions are chosen from the predicted family. The independent detailed-model '
            'candidate is retained to flag disagreement for review.',
            '',f'Artifacts: `artifacts/runs/{run_id}/`.',
            'These are retrospective day-held-out results; final models are then fitted on all valid recordings.']
    if skipped:report+=['','Skipped days:',pd.DataFrame(skipped).to_markdown(index=False)]
    if len(invalid):report+=['','Excluded invalid recordings:',invalid.to_markdown(index=False)]
    (project/'reports'/'latest_training.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('SUMMARY\n' + summary.to_string(), flush=True)
    print('MANIFEST', out / 'manifest.json', flush=True)
    return folds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='.')
    parser.add_argument('--data-root')
    parser.add_argument('--force-audit', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workspace = Path(args.project).resolve()
    mongo_runs = (root / 'mongo_runs').resolve()
    if workspace == mongo_runs or not workspace.is_relative_to(mongo_runs):
        parser.error('Direct training on the old CSV dataset is disabled; use train_from_mongo.bat with new labeled MongoDB data.')
    run(workspace, args.data_root, args.force_audit)
