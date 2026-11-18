"""New recording days/classes must retrain both models without editing counts."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ship_fault.models import InferenceEngine
from ship_fault.settings import CHANNELS, RuntimeSettings
from src.train_with_duplicates import original_locked_holdout, run
from src.data_loader import read_signal


def _recording(root, day, folder, index):
    destination=root/'data'/day/folder/f'{index}.csv'
    destination.parent.mkdir(parents=True,exist_ok=True)
    seed=int.from_bytes(hashlib.sha256(destination.relative_to(root).as_posix().encode()).digest()[:4],'little')
    rng=np.random.default_rng(seed)
    times=np.arange(4000)/4000.0
    values=np.column_stack([np.sin(2*np.pi*(20+channel)*times)+rng.normal(0,.02,4000)
                            for channel in range(6)])
    pd.DataFrame(np.column_stack((times,values)),columns=['Time',*CHANNELS]).to_csv(destination,index=False)
    return destination


def test_new_day_and_condition_refresh_both_models(tmp_path, monkeypatch):
    from src import data_loader
    monkeypatch.setattr(data_loader,'_create_figures',lambda *args:None)
    (tmp_path/'config').mkdir()
    (tmp_path/'reports').mkdir()
    config=yaml.safe_load((Path(__file__).resolve().parents[1]/'config'/'config.yaml').read_text())
    config.update(data_root='data',active_rf_trees=3,active_windows_per_file=1)
    (tmp_path/'config'/'config.yaml').write_text(yaml.safe_dump(config))
    for day in ['data_day1','data_day2']:
        for condition in ['Normal','Cocked Bearing with loader']:
            for index in [1,2]:_recording(tmp_path,day,condition,index)
    run(tmp_path)
    pointer=tmp_path/'reports'/'duplicate_inclusive'/'manifest.json'
    first=json.loads(pointer.read_text())
    assert first['raw_files']==8 and first['detailed_classes']==2
    for condition in ['Normal With Loader','Cocked Bearing with loader']:
        for index in [1,2]:_recording(tmp_path,'data_day3',condition,index)
    run(tmp_path)
    second=json.loads(pointer.read_text())
    assert second['run_id']!=first['run_id']
    assert second['raw_files']==12 and second['detailed_classes']==3
    assert len(second['recording_days'])==3
    assert (tmp_path/first['family_artifact']).is_file()
    engine=InferenceEngine(RuntimeSettings(project_root=tmp_path))
    assert engine.model_info['family_classes']==2
    assert engine.model_info['detailed_classes']==3
    recording=pd.read_csv(tmp_path/'data'/'data_day3'/'Normal With Loader'/'1.csv')
    assert engine.predict_recording(recording.Time,recording.iloc[:,1:])['window_count']==1

    # A failed new run must leave the previous active version usable.
    def fail_save(*args,**kwargs):raise OSError('simulated disk failure')
    import src.train_with_duplicates as trainer
    monkeypatch.setattr(trainer.joblib,'dump',fail_save)
    with pytest.raises(OSError,match='simulated disk failure'):run(tmp_path)
    assert json.loads(pointer.read_text())['run_id']==second['run_id']
    assert InferenceEngine(RuntimeSettings(project_root=tmp_path)).model_info['detailed_classes']==3


def test_holdout_follows_content_when_new_day_sorts_first(tmp_path):
    (tmp_path/'reports').mkdir()
    (tmp_path/'reports'/'holdout_files.txt').write_text('data_day2/Normal/1.csv\n')
    metadata=pd.DataFrame({'file':['data_day10/Normal/copy.csv','data_day1/Normal/other.csv']})
    hashes={'data_day2/Normal/1.csv':'same','data_day10/Normal/copy.csv':'same',
            'data_day1/Normal/other.csv':'different'}
    mask,files=original_locked_holdout(tmp_path,metadata,hashes)
    assert mask.tolist()==[True,False]
    assert files==['data_day2/Normal/1.csv']


def test_training_rejects_reordered_channels_and_missing_time(tmp_path):
    path=_recording(tmp_path,'data_day1','Normal',1)
    frame=pd.read_csv(path)
    frame[['Time',*reversed(CHANNELS)]].to_csv(path,index=False)
    with pytest.raises(ValueError,match='documented order'):read_signal(path,list(CHANNELS))
    frame.loc[1234,'Time']=np.nan
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='finite'):read_signal(path,list(CHANNELS))
