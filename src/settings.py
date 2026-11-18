"""User-configurable project paths and reproducibility settings."""
import os
from pathlib import Path
import yaml

def load_config(project='.'):
    conf=yaml.safe_load((Path(project)/'config'/'config.yaml').read_text())
    if os.environ.get('DATA_ROOT'):conf['data_root']=os.environ['DATA_ROOT']
    if os.environ.get('DEEP_EPOCHS'):conf['deep_epochs']=int(os.environ['DEEP_EPOCHS'])
    return conf
