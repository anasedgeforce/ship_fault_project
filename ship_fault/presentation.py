"""Stable machine IDs and human-readable terminology for API/UI consumers."""
FAMILIES = {
    'normal': 'Normal',
    'inner_race': 'Inner Race Fault',
    'outer_race': 'Outer Race Fault',
    'cocked_bearing': 'Cocked Bearing',
    'foundation_looseness': 'Foundation Looseness',
    'dynamic_unbalance': 'Dynamic Unbalance',
}

CHILDREN = {
    'normal': {**{f'{rpm}_rpm': f'{rpm:,} RPM' for rpm in range(1200, 1800, 100)},
               '1500_rpm__v1': '1,500 RPM (Variant 1)',
               'loading_unspecified': 'Loading Unspecified', 'with_loader': 'With Loader'},
    **{family: {f'{severity}_severity__{mass}': f'{severity.title()} Severity — {description}'
                for severity in ('high', 'low')
                for mass, description in (
                    ('ub_double_mass', 'Double-Mass Unbalance'),
                    ('ub_single_mass', 'Single-Mass Unbalance'),
                    ('no_ub_specified', 'Unbalance Not Specified'))}
       for family in ('inner_race', 'outer_race')},
    'cocked_bearing': {'with_loader': 'With Loader', 'without_loader': 'Without Loader',
                       'without_loader__ub_double_mass': 'Without Loader — Double-Mass Unbalance'},
    'foundation_looseness': {'level_1': 'Level 1', 'level_2': 'Level 2',
                            'level_1__ub_single_mass': 'Level 1 — Single-Mass Unbalance',
                            'level_2__ub_single_mass': 'Level 2 — Single-Mass Unbalance',
                            'loose_bottom_bolts': 'Loose Bottom Bolts'},
    'dynamic_unbalance': {'phase_180__without_loader': '180° — Without Loader',
                         'phase_90__without_loader': '90° — Without Loader',
                         'double_mass__without_loader': 'Double Mass — Without Loader',
                         'single_mass__without_loader': 'Single Mass — Without Loader'},
}
LABELS = {**FAMILIES, **{f'{family}__{child}': f'{FAMILIES[family]} — {name}'
                        for family, children in CHILDREN.items() for child, name in children.items()}}
DECISIONS = {'classified': 'Prediction Available', 'needs_review': 'Review Required',
             'invalid_input': 'Invalid Input', 'failed': 'Analysis Failed'}
REVIEW_REASONS = {
    'continuous_duration_unvalidated': 'Continuous operation has not been validated for this recording duration.',
    'physical_axis_mapping_unverified': 'Physical sensor-axis mapping has not been verified.',
    'equipment_domain_unvalidated': 'Model suitability for this equipment has not been validated.',
    'family_detail_disagree': 'The family and independent subtype predictions disagree.',
    'low_model_score': 'One or more model scores are below the review threshold.',
    'novelty_reference_unavailable': 'The reference for unfamiliar signals is unavailable.',
    'outside_training_feature_reference': 'The signal differs from the training reference.',
    'flat_channel': 'A channel is flat or nearly flat.',
    'possible_clipping': 'A channel may be clipped at its measurement limit.',
    'identical_channels': 'Two channels contain identical readings; check acquisition and wiring.',
    'mixed_fault_predictions_within_recording': 'The recording contains inconsistent fault predictions.',
}


def label_catalog(families=None, subtypes=None):
    family_ids = list(FAMILIES) if families is None else list(families)
    subtype_ids = ([f'{f}__{c}' for f, children in CHILDREN.items() for c in children]
                   if subtypes is None else list(subtypes))
    if any(f not in FAMILIES for f in family_ids):
        raise ValueError('Model family is missing from the presentation label catalog')
    if any(s not in LABELS or '__' not in s or s.split('__', 1)[0] not in family_ids for s in subtype_ids):
        raise ValueError('Model subtype is missing from the presentation label catalog')
    return {'version': 1, 'families': [{'id': f, 'display_name': FAMILIES[f]} for f in family_ids],
            'subtypes': [{'id': s, 'family_id': s.split('__', 1)[0], 'display_name': LABELS[s],
                          'subtype_name': CHILDREN[s.split('__', 1)[0]][s.split('__', 1)[1]]}
                         for s in subtype_ids],
            'labels': {key: LABELS[key] for key in [*family_ids, *subtype_ids]},
            'decisions': DECISIONS.copy(), 'review_reasons': REVIEW_REASONS.copy(),
            'score_description': 'Model score, not measured accuracy or a calibrated probability of a fault.'}
