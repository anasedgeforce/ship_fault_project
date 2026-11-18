"""Saved-file and pasted input formats must reach the same prediction path."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ship_fault.catalog import manual_recording_choices
from ship_fault.manual import parse_manual_payload
from ship_fault.models import InferenceEngine
from ship_fault.settings import CHANNELS, RuntimeSettings


def test_pasted_csv_spreadsheet_and_json_predict_identically():
    frame=pd.read_csv('data/data_day1/Cocked Bearing with loader/1.csv',nrows=6000)
    settings=RuntimeSettings()
    values=frame.iloc[:,1:].to_numpy()
    times=frame['Time'].to_numpy()
    sources=[
        frame.to_csv(index=False),
        frame.to_csv(index=False,sep='\t'),
        json.dumps({'timestamps':times.tolist(),'channels':values.tolist()}),
        json.dumps({'samples':[{'timestamp':t,'channels':row} for t,row in zip(times.tolist(),values.tolist())]}),
    ]
    engine=InferenceEngine(settings)
    results=[]
    for source in sources:
        parsed_times,parsed_values=parse_manual_payload(source,settings)
        np.testing.assert_allclose(parsed_times,times)
        np.testing.assert_allclose(parsed_values,values,rtol=2e-6,atol=2e-6)
        results.append(engine.predict_recording(parsed_times,parsed_values))
    assert all(result['window_count']==2 for result in results)
    for result in results[1:]:
        for left,right in zip(results[0]['predictions'],result['predictions']):
            assert left['family']['label']==right['family']['label']
            assert left['family']['score']==pytest.approx(right['family']['score'])
            assert left['detailed_condition']['label']==right['detailed_condition']['label']


def test_paste_rejects_incomplete_recording():
    with pytest.raises(ValueError,match='4000|at least|expected'):
        parse_manual_payload('Time,'+','.join(CHANNELS)+'\n0,'+'0,'*5+'0\n',RuntimeSettings())


def test_saved_recording_dropdown_lists_days_conditions_and_examples(tmp_path):
    (tmp_path/'config').mkdir()
    (tmp_path/'config'/'config.yaml').write_text('data_root: data\n')
    sample=tmp_path/'examples'/'demo.csv'
    sample.parent.mkdir()
    sample.write_text('example')
    nested=tmp_path/'data'/'data_day4'/'Outer Race Fault with loader'/'Outer Race Fault High Severity'/'1.csv'
    nested.parent.mkdir(parents=True)
    nested.write_text('recording')
    catalog=manual_recording_choices(tmp_path)
    assert catalog['Examples']['Sample recordings']==[sample]
    assert catalog['data_day4']['Outer Race Fault with loader/Outer Race Fault High Severity']==[nested]
