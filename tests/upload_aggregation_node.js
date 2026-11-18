// Run with: node tests/upload_aggregation_node.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '..', 'ship_fault', 'web', 'app.js'), 'utf8');
const context = vm.createContext({document: {addEventListener() {}}});
vm.runInContext(source, context);
const aggregate = vm.runInContext('aggregateUploadResults', context);

function recording(family, detail, familyScore, detailScore, count, decision = 'classified') {
  return {
    recording_decision: decision,
    recording_family: {label: family, score: familyScore},
    recording_detailed_condition: {label: detail, score: detailScore},
    valid_window_count: count, window_count: count,
    predictions: Array.from({length: count}, (_, index) => ({
      start_seconds: index * 0.5, family: {label: family}, detailed_condition: {label: detail}, decision,
    })),
  };
}
const mixed = recording('inner_race', 'inner_race__low', 0.8, 0.8, 2);
mixed.predictions[1].family.label = 'outer_race';
mixed.predictions[1].detailed_condition.label = 'outer_race__high';
const entries = [
  {name: 'upload/pump_a/low_1.csv', result: recording('inner_race', 'inner_race__low', 0.8, 0.75, 1)},
  {name: 'upload/pump_a/low_2.csv', result: recording('inner_race', 'inner_race__low', 0.9, 0.95, 3)},
  {name: 'upload/pump_b/low_1.csv', result: recording('inner_race', 'inner_race__low', 0.6, 0.7, 2)},
  {name: 'upload/pump_b/low_2.csv', result: recording('inner_race', 'inner_race__low', 0.8, 0.9, 2)},
  {name: 'upload/pump_a/high.csv', result: recording('inner_race', 'inner_race__high', 0.9, 0.9, 2)},
  {name: 'upload/pump_a/normal.csv', result: recording('normal', 'normal__loader', 0.9, 0.9, 2)},
  {name: 'upload/pump_a/mixed.csv', result: mixed},
  {name: 'upload/pump_a/review.csv', result: recording('inner_race', 'inner_race__low', 0.7, 0.7, 1, 'needs_review')},
  {name: 'upload/no_equipment.csv', result: recording('inner_race', 'inner_race__low', 0.9, 0.9, 1)},
  {name: 'upload/pump_a/broken.csv', error: 'Invalid CSV'},
];
const summary = aggregate(entries);
const disagreement = recording('normal', 'normal__with_loader', 0.8, 0.6, 1, 'needs_review');
disagreement.independent_recording_detailed_condition = {
  label: 'inner_race__low_severity__no_ub_specified', score: 0.55};
const disagreementSummary = aggregate([{name: 'upload/disagreement.csv', result: disagreement}]);
assert.equal(disagreementSummary.fault_candidate_files, 1);
assert.match(disagreementSummary.files[0].finding, /Independent subtype candidate/);
assert.equal(disagreementSummary.files[0].independent_detailed_condition.label,
  'inner_race__low_severity__no_ub_specified');
const addAccuracy = vm.runInContext('addUploadAccuracy', context);
addAccuracy(summary, {
  'pump_a/low_1.csv': {family: 'inner_race', detail: 'inner_race__low'},
  'pump_a/low_2.csv': {family: 'inner_race', detail: 'inner_race__high'},
  'pump_b/low_1.csv': {family: 'outer_race', detail: 'outer_race__low'},
  'pump_a/broken.csv': {family: 'normal', detail: 'normal__loader'},
});
assert.equal(summary.upload_accuracy.labeled_recordings, 3);
assert.equal(summary.upload_accuracy.family_correct, 2);
assert.equal(summary.upload_accuracy.detail_correct, 1);
assert.equal(summary.file_count, 10);
assert.equal(summary.successful_file_count, 9);
assert.equal(summary.failed_file_count, 1);
assert.equal(summary.matching_groups.length, 2);
const pumpA = summary.matching_groups.find((group) => group.equipment_folder === 'pump_a');
const pumpB = summary.matching_groups.find((group) => group.equipment_folder === 'pump_b');
assert.equal(pumpA.family, 'inner_race');
assert.equal(pumpA.detailed_condition, 'inner_race__low');
assert.equal(pumpA.files.length, 2);
assert.ok(Math.abs(pumpA.family_score_mean - 0.85) < 1e-12);
assert.ok(Math.abs(pumpA.detailed_score_mean - 0.85) < 1e-12);
assert.equal(pumpB.files.length, 2);
assert.equal(summary.files[6].finding, 'Mixed signals; review required');
assert.equal(summary.files[8].equipment_folder, null);
assert.equal(summary.files[9].error, 'Invalid CSV');
assert.equal('average_family' in summary, false);
assert.equal('average_detailed_condition' in summary, false);
assert.equal('average_rms_mm_s' in summary, false);
const elements = new Map();
function element() {
  return {textContent: '', children: [], classList: {add() {}, remove() {}, toggle() {}},
    append(child) { this.children.push(child); }, replaceChildren() { this.children = []; },
    get lastChild() { return this.children[this.children.length - 1]; }};
}
context.document.getElementById = (id) => {
  if (!elements.has(id)) elements.set(id, element());
  return elements.get(id);
};
context.document.createElement = element;
vm.runInContext('renderUploadSummary', context)(summary);
assert.equal(elements.get('upload-result-rows').children.length, 10);
assert.equal(elements.get('upload-group-rows').children.length, 2);
assert.equal(elements.get('upload-group-rows').children[0].children[0].textContent, 'Inner Race');
assert.equal(elements.has('upload-family'), false);
assert.match(elements.get('upload-accuracy-note').textContent, /Known labels matched 3 successful recordings/);
const ambiguous = aggregate([
  {name: 'upload/pump_a/same.csv', result: recording('inner_race', 'inner_race__low', 0.8, 0.8, 1)},
  {name: 'upload/pump_b/same.csv', result: recording('inner_race', 'inner_race__low', 0.8, 0.8, 1)},
]);
addAccuracy(ambiguous, {'same.csv': {family: 'inner_race', detail: 'inner_race__low'}});
assert.equal(ambiguous.upload_accuracy.labeled_recordings, 0);
assert.equal(ambiguous.upload_accuracy.unmatched_label_paths.length, 1);
console.log('Upload grouping passed');
