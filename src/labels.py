"""Map the simulator's original condition folders to its six parent families."""
from __future__ import annotations

import re


def clean(value: str) -> str:
    value = value.lower().replace('cocker bearing', 'cocked bearing').replace('un balance', 'unbalance')
    return re.sub(r'\s+', ' ', value).strip()


def attributes(folder: str) -> dict:
    path = clean(folder)
    if 'without loader' in path:
        loader = 'no'
    elif 'with loader' in path:
        loader = 'yes'
    else:
        loader = 'unspecified'
    if 'inner race' in path:
        family = 'inner_race'
    elif 'outer race' in path:
        family = 'outer_race'
    elif 'cocked bearing' in path:
        family = 'cocked_bearing'
    elif 'foundation looseness' in path or re.search(r'(^|/)level [12]\b', path):
        family = 'foundation_looseness'
    elif 'dynamic unbalance' in path or path.startswith('ub '):
        family = 'dynamic_unbalance'
    elif path == 'normal' or path.startswith('normal/') or path.startswith('normal '):
        family = 'normal'
    else:
        family = 'unmapped'
    severity = ('high' if 'high severity' in path else 'low' if 'low severity' in path
                else 'level_1' if 'level 1' in path else 'level_2' if 'level 2' in path
                else 'unspecified')
    mass = 'double' if 'double mass' in path else 'single' if 'single mass' in path else 'none'
    phase = '90' if '90 deg' in path else '180' if '180 deg' in path else 'none'
    rpm = re.search(r'\b(1[2-7]00)\s*rpm\b', path)
    return dict(fault_type=family, severity=severity, loader=loader, mass_config=mass,
                unbalance_phase=phase, compound=mass != 'none' and family not in ('dynamic_unbalance', 'normal'),
                rpm=int(rpm.group(1)) if rpm else None)


def label_for(folder: str) -> str:
    """Return one distinct child for every condition in the supplied taxonomy."""
    path = clean(folder)
    a = attributes(folder)
    family = a['fault_type']
    if family == 'unmapped':
        raise ValueError(f'Unrecognized condition: {folder}')
    if family == 'normal':
        if a['rpm'] is not None:
            version = '__v1' if re.search(r'\b1500\s*rpm\s*v1\b', path) else ''
            return f"normal__{a['rpm']}_rpm{version}"
        return 'normal__with_loader' if a['loader'] == 'yes' else 'normal__loading_unspecified'
    if family in ('inner_race', 'outer_race'):
        if a['severity'] not in ('high', 'low'):
            raise ValueError(f'Missing bearing severity: {folder}')
        suffix = f"__ub_{a['mass_config']}_mass" if a['mass_config'] != 'none' else '__no_ub_specified'
        return f"{family}__{a['severity']}_severity{suffix}"
    if family == 'cocked_bearing':
        if a['loader'] == 'yes':
            return 'cocked_bearing__with_loader'
        if a['loader'] == 'no':
            return ('cocked_bearing__without_loader__ub_double_mass' if a['mass_config'] == 'double'
                    else 'cocked_bearing__without_loader')
        raise ValueError(f'Missing cocked-bearing loader status: {folder}')
    if family == 'foundation_looseness':
        if '/normal' in path:
            return 'foundation_looseness__loose_bottom_bolts'
        if a['severity'] not in ('level_1', 'level_2'):
            raise ValueError(f'Missing foundation level: {folder}')
        return (f"foundation_looseness__{a['severity']}__ub_single_mass"
                if a['mass_config'] == 'single' else f"foundation_looseness__{a['severity']}")
    if a['unbalance_phase'] != 'none':
        return f"dynamic_unbalance__phase_{a['unbalance_phase']}__without_loader"
    if a['mass_config'] in ('single', 'double'):
        return f"dynamic_unbalance__{a['mass_config']}_mass__without_loader"
    raise ValueError(f'Missing unbalance subtype: {folder}')
