// Run with: node tests/prediction_summary_node.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const elements = new Map();
function element() {
  return {textContent: '', className: '', children: [], classList: {add() {}, remove() {}, toggle() {}},
    closest() { return {classList: {remove() {}, add() {}}}; },
    append(child) { this.children.push(child); }, replaceChildren(...children) { this.children = children; }};
}
const document = {addEventListener() {}, createElement: element,
  getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }};
const context = vm.createContext({document, requestAnimationFrame() {}});
vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'ship_fault', 'web', 'app.js'), 'utf8'), context);
const render = vm.runInContext('renderResult', context);
const scoreTone = vm.runInContext('scoreTone', context);
assert.equal(scoreTone(0.71), 'score-high');
assert.equal(scoreTone(0.70), 'score-medium');
assert.equal(scoreTone(0.50), 'score-medium');
assert.equal(scoreTone(0.49), 'score-low');
const family = {label: 'inner_race', score: 0.9};
const detail = {label: 'inner_race__low__loader', score: 0.8};
render({recording_family: family, recording_detailed_condition: detail,
  recording_decision: 'classified', window_count: 1, recording_duration_seconds: 1,
  predictions: [{start_seconds: 0, family, detailed_condition: detail}]});
assert.equal(elements.get('family-label').textContent, 'Inner Race');
assert.match(elements.get('detail-label').textContent, /Inner Race.*Low.*Loader/);
assert.equal(elements.get('detail-score').children[1].textContent, '0.800');

const disagree = {label: 'outer_race__high__loader', score: 0.7};
render({recording_family: family, recording_detailed_condition: detail,
  independent_recording_detailed_condition: disagree,
  recording_decision: 'needs_review', window_count: 1, recording_duration_seconds: 1,
  predictions: [{start_seconds: 0, family, detailed_condition: detail} ]});
assert.match(elements.get('detail-score').children[2].textContent, /independent subtype model disagreed/);
assert.equal(elements.has('health-score'), false);
console.log('Prediction summary passed');
