"""The display catalog must preserve every deployed machine label and its parent."""
import pytest
from fastapi.testclient import TestClient
from ship_fault.api import create_app
from ship_fault.settings import RuntimeSettings
from ship_fault.presentation import label_catalog


def test_catalog_matches_active_models_and_preserves_original_meanings():
    with TestClient(create_app(RuntimeSettings())) as api:
        response = api.get('/v1/labels')
        assert response.status_code == 200
        catalog = response.json()
        engine = api.app.state.engine
        assert {f['id'] for f in catalog['families']} == set(engine.family['family_names'])
        assert {s['id'] for s in catalog['subtypes']} == set(engine.detail['label_names'])
        assert len(catalog['families']) == 6 and len(catalog['subtypes']) == 33
        assert len(set(s['display_name'] for s in catalog['subtypes'])) == 33
        assert catalog['labels']['foundation_looseness__loose_bottom_bolts'].startswith('Foundation Looseness')
        assert 'Not Specified' in catalog['labels']['outer_race__high_severity__no_ub_specified']
        assert catalog['labels']['normal__1500_rpm'] != catalog['labels']['normal__1500_rpm__v1']
        assert '180°' in catalog['labels']['dynamic_unbalance__phase_180__without_loader']
        assert all(s['id'].startswith(s['family_id'] + '__') for s in catalog['subtypes'])
        assert api.get('/v1/models').json()['label_catalog'] == catalog
        days = api.get('/v1/recordings').json()['days']
        assert all(day['display_name'].startswith('Day ') for day in days)
        assert all('__' not in c['display_name'] for d in days for c in d['conditions'])


def test_unknown_model_label_requires_catalog_update():
    with pytest.raises(ValueError, match='subtype'):
        label_catalog(['normal'], ['normal__unknown'])
