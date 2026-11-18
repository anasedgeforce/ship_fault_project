const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const twin = require('../ship_fault/web/twin-map.js');

const catalog = JSON.parse(fs.readFileSync(path.join(__dirname, '../docs/label_catalog.json'), 'utf8'));
const source = fs.readFileSync(path.join(__dirname, '../ship_fault/web/vendor/digital_twin_original.html'), 'utf8');
const presets = JSON.parse(source.match(/const FP_FAM=(\[.*?\]);/s)[1]);
const presetIDs = new Set(presets.map((item) => item.id));
assert.equal(catalog.subtypes.length, 33);
assert.deepEqual(Object.keys(twin.presets).sort(), catalog.subtypes.map((item) => item.id).sort());
for (const subtype of catalog.subtypes) {
  assert.ok(presetIDs.has(twin.presets[subtype.id]), subtype.id);
  const result = twin.candidate({label: subtype.family_id}, {label: subtype.id});
  assert.ok(result, subtype.id);
  assert.ok(twin.compactLabel(subtype.id).length > 0 && twin.compactLabel(subtype.id).length <= 40,
    `compact label missing for ${subtype.id}`);
  const regions = [];
  if (['inner_race', 'outer_race'].includes(subtype.family_id)) regions.push('bearings');
  if (subtype.family_id === 'cocked_bearing') regions.push('bearingHousings');
  if (subtype.family_id === 'foundation_looseness') regions.push('rotorBase');
  if (subtype.family_id === 'dynamic_unbalance' || subtype.id.includes('__ub_')) regions.push('rotorDiscs');
  assert.deepEqual(result.regions, regions, subtype.id);
}
assert.equal(twin.candidate({label: 'normal'}, {label: 'outer_race__high_severity__no_ub_specified'}), null);
assert.equal(twin.candidate({label: 'outer_race'}, {label: 'unknown'}), null);
assert.equal(twin.compactLabel('unknown'), '');
const independent = twin.predictionCandidate({label: 'normal'}, {label: 'normal__with_loader'},
  {label: 'inner_race__low_severity__no_ub_specified'});
assert.equal(independent.subtype, 'inner_race__low_severity__no_ub_specified');
assert.equal(independent.source, 'independent_subtype');
assert.deepEqual(independent.regions, ['bearings']);

const normal = {name: 'normal.csv', family: {label: 'normal'},
  detailed_condition: {label: 'normal__with_loader'}, decision: 'classified'};
const bearing = {name: 'bearing.csv', family: {label: 'inner_race'},
  detailed_condition: {label: 'inner_race__high_severity__ub_single_mass'}, decision: 'needs_review'};
const foundation = {name: 'foundation.csv', family: {label: 'foundation_looseness'},
  detailed_condition: {label: 'foundation_looseness__level_2'}, decision: 'classified'};
const rows = twin.manualEntries([normal, bearing, foundation]);
assert.equal(rows.length, 3);
assert.equal(rows[0].candidates.length, 0);
assert.equal(rows[1].review, true);
assert.deepEqual(twin.regionsFor(rows).sort(), ['bearings', 'rotorBase', 'rotorDiscs'].sort());
const mixed = twin.manualEntries([{name: 'mixed.csv', family: {label: 'normal'},
  detailed_condition: {label: 'normal__with_loader'}, decision: 'needs_review', mixed_windows: true,
  windows: [{family: 'normal', detailed_condition: 'normal__with_loader'},
    {family: 'outer_race', detailed_condition: 'outer_race__low_severity__no_ub_specified'}]}]);
assert.equal(mixed[0].review, true);
assert.deepEqual(twin.regionsFor(mixed), []);
const mixedFault = twin.manualEntries([{name: 'mixed-fault.csv', family: {label: 'outer_race'},
  detailed_condition: {label: 'outer_race__low_severity__no_ub_specified'},
  decision: 'needs_review', mixed_windows: true}]);
assert.deepEqual(twin.regionsFor(mixedFault), ['bearings']);
const independentManual = twin.manualEntries([{name: 'disagreement.csv', family: {label: 'normal'},
  detailed_condition: {label: 'normal__with_loader'},
  independent_detailed_condition: {label: 'inner_race__low_severity__no_ub_specified'},
  decision: 'needs_review'}]);
assert.deepEqual(twin.regionsFor(independentManual), ['bearings']);
assert.equal(independentManual[0].review, true);
assert.equal(independentManual[0].normal, false);

const live = new twin.LiveState();
live.update({family: {label: 'cocked_bearing'},
  detailed_condition: {label: 'cocked_bearing__without_loader__ub_double_mass'}, decision: 'needs_review'});
assert.deepEqual(live.snapshot().regions, ['bearingHousings', 'rotorDiscs']);
const good = {family: {label: 'normal'}, detailed_condition: {label: 'normal__with_loader'}, decision: 'classified'};
live.update(good);
live.update({decision: 'invalid_input'});
assert.equal(live.snapshot().normalStreak, 0);
live.update(good);
live.update(good);
assert.equal(live.snapshot().regions.length, 2);
live.update(good);
assert.deepEqual(live.snapshot().regions, []);
live.update({family: {label: 'outer_race'},
  detailed_condition: {label: 'outer_race__high_severity__no_ub_specified'}, decision: 'classified'});
assert.deepEqual(live.snapshot().regions, ['bearings']);
live.reset();
assert.deepEqual(live.snapshot().regions, []);
live.update({family: {label: 'outer_race'},
  detailed_condition: {label: 'outer_race__high_severity__no_ub_specified'}, decision: 'classified'});
live.update(good);
live.update({family: {label: 'normal'}, detailed_condition: {label: 'normal__with_loader'},
  independent_detailed_condition: {label: 'inner_race__low_severity__no_ub_specified'},
  decision: 'needs_review'});
assert.deepEqual(live.snapshot().regions, ['bearings']);
assert.equal(live.snapshot().candidate.source, 'independent_subtype');
assert.equal(live.snapshot().candidate.review, true);
assert.equal(live.snapshot().normalStreak, 0);
live.update(good);
live.update(good);
live.update(good);
assert.deepEqual(live.snapshot().regions, []);
console.log('Digital twin mappings, manual aggregation, and live clearing passed.');
