"""Refresh data, train both active models, evaluate, and publish one model version."""
import argparse
from pathlib import Path
from src.train_with_duplicates import run
from scripts.train_novelty_reference import fit_novelty_reference


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--force-audit',action='store_true')
    args=parser.parse_args()
    project = args.project.resolve()
    mongo_runs = (Path(__file__).resolve().parent / 'mongo_runs').resolve()
    if not project.is_relative_to(mongo_runs) or project == mongo_runs:
        parser.error('Direct training from the original CSV dataset is disabled. Run train_from_mongo.bat when new labeled MongoDB data is ready.')
    run(project,args.data_root,args.force_audit)
    fit_novelty_reference(project)
