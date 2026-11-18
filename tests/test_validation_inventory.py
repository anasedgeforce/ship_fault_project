"""Evaluation dropdown inventories follow completed folds after retraining."""
import numpy as np
import pandas as pd
import pytest

from src.train_with_duplicates import _validation_inventory


def test_inventory_excludes_locked_files_and_skipped_days():
    meta = pd.DataFrame([
        {'day': 'day1', 'file': 'day1/a.csv', 'label': 'normal__loader', 'fault_type': 'normal'},
        {'day': 'day1', 'file': 'day1/a.csv', 'label': 'normal__loader', 'fault_type': 'normal'},
        {'day': 'day2', 'file': 'day2/b.csv', 'label': 'inner_race__low', 'fault_type': 'inner_race'},
        {'day': 'day3', 'file': 'day3/c.csv', 'label': 'normal__loader', 'fault_type': 'normal'},
    ])
    holdout = np.array([False, False, True, False])
    folds = pd.DataFrame([{'task': '8-family', 'day': 'day1', 'n_valid_files': 1}])
    hashes = {'day1/a.csv': 'a' * 64, 'day2/b.csv': 'b' * 64, 'day3/c.csv': 'c' * 64}
    rows = _validation_inventory(meta, holdout, hashes, folds, 'new-run', '8-family')
    assert rows['file'].tolist() == ['day1/a.csv']
    assert rows['run_id'].tolist() == ['new-run']
    assert rows['sha256'].tolist() == ['a' * 64]
    folds.loc[0, 'n_valid_files'] = 2
    with pytest.raises(ValueError, match='differs from the evaluated folds'):
        _validation_inventory(meta, holdout, hashes, folds, 'new-run', '8-family')
