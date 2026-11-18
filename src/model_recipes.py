"""Shared estimator recipes for evaluation, final training, and fold reproduction."""
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from .crossday_advanced import FeatureView
from .round5_order_benchmark import OrderFeatureView
from .models import make_model


FAMILY_SPECS = {
    'baseline': (2, 'sqrt', 'balanced_subsample'),
    'leaf1': (1, 'sqrt', 'balanced_subsample'),
    'more_features': (2, .5, 'balanced_subsample'),
    'unweighted': (2, 'sqrt', None),
}
DETAIL_SPECS = ('baseline', 'rf_raw', 'rf_relative', 'rf_relative_amp',
                'et_relative', 'et_relative_amp')


def family_model(names, with_order, spec='baseline'):
    leaf, maximum, weight = FAMILY_SPECS[spec]
    view = (OrderFeatureView(tuple(names), 'relative_amp', len(names)) if with_order
            else FeatureView(tuple(names), 'relative'))
    return Pipeline([('view', view), ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('variance', VarianceThreshold(1e-12)),
                     ('clf', ExtraTreesClassifier(n_estimators=220, min_samples_leaf=leaf,
                        max_features=maximum, class_weight=weight, random_state=42, n_jobs=1))])


def detail_model(names, spec='baseline', trees=180):
    if spec not in DETAIL_SPECS:
        raise ValueError(f'unknown detail model recipe: {spec}')
    if spec == 'baseline':
        return make_model('random_forest', 42, trees)
    kind, view = spec.split('_', 1)
    estimator = RandomForestClassifier if kind == 'rf' else ExtraTreesClassifier
    return Pipeline([('view', FeatureView(tuple(names), view)),
                     ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('variance', VarianceThreshold(1e-12)),
                     ('clf', estimator(n_estimators=trees, min_samples_leaf=2,
                        max_features='sqrt', class_weight='balanced_subsample', random_state=42, n_jobs=1))])
