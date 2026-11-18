// Exact model subtype IDs -> supplied twin preset IDs. Display names are not identifiers.
(function (root) {
  const presets = Object.freeze({
    normal__1200_rpm: 'normal',
    normal__1300_rpm: 'normal',
    normal__1400_rpm: 'normal',
    normal__1500_rpm: 'normal',
    normal__1500_rpm__v1: 'normal',
    normal__1600_rpm: 'normal',
    normal__1700_rpm: 'normal',
    normal__loading_unspecified: 'normal',
    normal__with_loader: 'normal',
    inner_race__high_severity__ub_double_mass: 'f15',
    inner_race__high_severity__ub_single_mass: 'f16',
    inner_race__high_severity__no_ub_specified: 'f14',
    inner_race__low_severity__ub_double_mass: 'f18',
    inner_race__low_severity__ub_single_mass: 'f19',
    inner_race__low_severity__no_ub_specified: 'f17',
    outer_race__high_severity__ub_double_mass: 'f26',
    outer_race__high_severity__ub_single_mass: 'f25',
    outer_race__high_severity__no_ub_specified: 'f24',
    outer_race__low_severity__ub_double_mass: 'f28',
    outer_race__low_severity__ub_single_mass: 'f29',
    outer_race__low_severity__no_ub_specified: 'f27',
    cocked_bearing__with_loader: 'f01',
    cocked_bearing__without_loader: 'f02',
    cocked_bearing__without_loader__ub_double_mass: 'f04',
    foundation_looseness__level_1: 'f20',
    foundation_looseness__level_1__ub_single_mass: 'f21',
    foundation_looseness__level_2: 'f22',
    foundation_looseness__level_2__ub_single_mass: 'f23',
    foundation_looseness__loose_bottom_bolts: 'f12',
    dynamic_unbalance__phase_180__without_loader: 'f06',
    dynamic_unbalance__phase_90__without_loader: 'f07',
    dynamic_unbalance__double_mass__without_loader: 'f30',
    dynamic_unbalance__single_mass__without_loader: 'f31',
  });
  const componentNames = Object.freeze({bearings: 'Bearings', bearingHousings: 'Bearing housings',
    rotorBase: 'Rotor base', rotorDiscs: 'Rotor discs'});

  function compactLabel(subtype) {
    if (!Object.hasOwn(presets, subtype)) return '';
    const parts = subtype.split('__');
    const family = parts[0];
    if (family === 'normal') return 'Normal';
    const unbalance = parts.some((part) => part.startsWith('ub_')) ? ' + UB' : '';
    if (family === 'inner_race' || family === 'outer_race') {
      return `${family === 'inner_race' ? 'Inner' : 'Outer'} Race · ${parts[1] === 'high_severity' ? 'High' : 'Low'}${unbalance}`;
    }
    if (family === 'cocked_bearing') return `Cocked Bearing${unbalance}`;
    if (family === 'foundation_looseness') {
      const level = parts[1] === 'loose_bottom_bolts' ? 'Bolts' : parts[1].replace('level_', 'Level ');
      return `Foundation · ${level}${unbalance}`;
    }
    if (family === 'dynamic_unbalance') {
      const kind = {phase_90: '90°', phase_180: '180°', single_mass: 'Single Mass', double_mass: 'Double Mass'};
      return `Unbalance · ${kind[parts[1]]}`;
    }
    return '';
  }

  function candidate(family, detail) {
    const subtype = detail?.label;
    if (!subtype || !Object.hasOwn(presets, subtype)) return null;
    const expected = subtype.split('__')[0];
    if (family?.label && family.label !== expected) return null;
    const regions = [];
    if (expected === 'inner_race' || expected === 'outer_race') regions.push('bearings');
    if (expected === 'cocked_bearing') regions.push('bearingHousings');
    if (expected === 'foundation_looseness') regions.push('rotorBase');
    if (expected === 'dynamic_unbalance' || subtype.includes('__ub_')) regions.push('rotorDiscs');
    return {family: expected, subtype, presetId: presets[subtype], regions};
  }

  function predictionCandidate(family, detail, independentDetail) {
    const primary = candidate(family, detail);
    if (primary && primary.family !== 'normal') return primary;
    // The independent subtype model can disagree with a Normal family result.
    // Preserve that fault as a review candidate, as the live Mongo path does.
    if (family?.label === 'normal') {
      for (const alternative of [detail, independentDetail]) {
        const label = alternative?.label;
        if (!label || label.split('__')[0] === 'normal') continue;
        const found = candidate({label: label.split('__')[0]}, alternative);
        if (found) return {...found, source: 'independent_subtype'};
      }
    }
    return primary;
  }

  function manualEntries(entries) {
    const rows = [];
    for (const entry of entries) {
      const mixed = Boolean(entry.mixed_windows);
      const main = predictionCandidate(entry.family, entry.detailed_condition,
        entry.independent_detailed_condition);
      const candidates = main && main.family !== 'normal' ? [main] : [];
      const unknown = !main;
      rows.push({name: entry.name || 'Recording', candidates,
        review: mixed || unknown || main?.source === 'independent_subtype' || entry.decision !== 'classified',
        mixed, normal: !candidates.length && entry.family?.label === 'normal' && !mixed,
        invalid: unknown || !entry.family || !entry.detailed_condition});
    }
    return rows;
  }

  function regionsFor(rows) {
    return [...new Set(rows.flatMap((row) => row.candidates.flatMap((c) => c.regions)))];
  }

  class LiveState {
    constructor() { this.reset(); }
    reset() { this.candidate = null; this.normalStreak = 0; return this.snapshot(); }
    update(window) {
      const found = predictionCandidate(window?.family, window?.detailed_condition,
        window?.independent_detailed_condition);
      if (!found || window?.decision === 'invalid_input' || window?.decision === 'failed') {
        this.normalStreak = 0;
      } else if (found.family !== 'normal') {
        this.candidate = {...found, review: found.source === 'independent_subtype' || window.decision !== 'classified'};
        this.normalStreak = 0;
      } else if (window.decision === 'classified') {
        this.normalStreak += 1;
        if (this.normalStreak >= 3) this.candidate = null;
      } else {
        this.normalStreak = 0;
      }
      return this.snapshot();
    }
    snapshot() { return {candidate: this.candidate, normalStreak: this.normalStreak,
      regions: this.candidate?.regions || []}; }
  }

  const api = {presets, componentNames, compactLabel, candidate, predictionCandidate,
    manualEntries, regionsFor, LiveState};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.SimulatorTwinMap = api;
})(typeof window === 'undefined' ? globalThis : window);
