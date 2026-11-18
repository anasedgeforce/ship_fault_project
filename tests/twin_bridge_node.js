const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const map = require('../ship_fault/web/twin-map.js');

const calls = [];
const elements = {};
function node() {
  const classes = new Set();
  return {textContent: '', children: [], classList: {
    add(value) { classes.add(value); calls.push(['add', value]); },
    remove(value) { classes.delete(value); calls.push(['remove', value]); },
    toggle(value, force) { if (force) classes.add(value); else classes.delete(value); return force; },
    contains(value) { return classes.has(value); }},
  replaceChildren(...children) { this.children = children; },
  append(...children) { this.children.push(...children); }, after() {}};
}
for (const key of ['digital-twin-frame', 'digital-twin-panel', 'twin-fault-list', 'twin-status',
  'prediction-panel', 'live-panel']) elements[key] = node();
const frame = elements['digital-twin-frame'];
frame.src = '';
frame.contentWindow = {postMessage(message, target) { calls.push(['post', message, target]); }};
let onMessage;
const window = {SimulatorTwinMap: map, crypto: {randomUUID: () => 'test-nonce'},
  addEventListener(event, handler) { if (event === 'message') onMessage = handler; }};
const document = {getElementById: (id) => elements[id], createElement: () => node()};
vm.runInNewContext(fs.readFileSync(require.resolve('../ship_fault/web/twin-view.js'), 'utf8'),
  {window, document, encodeURIComponent});
const view = window.SimulatorTwinView;
view.init();
const display = (value) => value === 'inner_race' ? 'Inner Race Fault' : value;
view.manual([{name: 'a.csv', family: {label: 'inner_race'},
  detailed_condition: {label: 'inner_race__high_severity__ub_single_mass'}, decision: 'needs_review'},
{name: 'b.csv', family: {label: 'foundation_looseness'},
  detailed_condition: {label: 'foundation_looseness__level_2'}, decision: 'classified'}], display);
assert.match(frame.src, /nonce=test-nonce/);
assert.equal(elements['twin-fault-list'].children.length, 2);
assert.match(elements['twin-fault-list'].children[0].textContent, /Review Required/);
onMessage({source: frame.contentWindow, data: {type: 'simulator-twin/ready', version: 1, nonce: 'wrong'}});
assert.equal(calls.filter((call) => call[0] === 'post').length, 0);
onMessage({source: frame.contentWindow, data: {type: 'simulator-twin/ready', version: 1, nonce: 'test-nonce'}});
let sent = calls.filter((call) => call[0] === 'post').at(-1)[1];
assert.equal(sent.type, 'simulator-twin/update');
assert.deepEqual(Array.from(sent.regions), ['bearings', 'rotorDiscs', 'rotorBase']);
assert.equal(sent.faults.length, 2);
assert.match(sent.faults[0].subtype, /inner_race__high/);
assert.equal(sent.faults[0].shortSubtype, 'Inner Race · High + UB');
assert.deepEqual(Array.from(sent.faults[0].regions), ['bearings', 'rotorDiscs']);
assert.equal(sent.faults[0].review, true);
view.manual([{name: 'c.csv', family: {label: 'inner_race'},
  detailed_condition: {label: 'inner_race__high_severity__ub_single_mass'}, decision: 'classified'},
{name: 'd.csv', family: {label: 'inner_race'},
  detailed_condition: {label: 'inner_race__high_severity__ub_single_mass'}, decision: 'classified'}], display);
sent = calls.filter((call) => call[0] === 'post').at(-1)[1];
assert.equal(sent.faults.length, 1);
assert.equal(sent.faults[0].count, 2);
assert.equal(sent.faults[0].shortSubtype, 'Inner Race · High + UB');
view.manual([{name: 'normal.csv', family: {label: 'normal'},
  detailed_condition: {label: 'normal__with_loader'}, decision: 'classified'}]);
sent = calls.filter((call) => call[0] === 'post').at(-1)[1];
assert.deepEqual(Array.from(sent.faults), []);
view.manual([{name: 'disagreement.csv', family: {label: 'normal'},
  detailed_condition: {label: 'normal__with_loader'},
  independent_detailed_condition: {label: 'inner_race__low_severity__no_ub_specified'},
  decision: 'needs_review'}]);
sent = calls.filter((call) => call[0] === 'post').at(-1)[1];
assert.deepEqual(Array.from(sent.regions), ['bearings']);
assert.equal(sent.faults[0].review, true);
assert.match(elements['twin-fault-list'].children[0].textContent, /Independent Subtype Candidate/);
view.live({candidate: map.candidate({label: 'outer_race'},
  {label: 'outer_race__high_severity__no_ub_specified'}), regions: ['bearings']});
sent = calls.filter((call) => call[0] === 'post').at(-1)[1];
assert.equal(sent.mode, 'live');
assert.deepEqual(Array.from(sent.regions), ['bearings']);
assert.equal(sent.faults.length, 1);
assert.match(sent.faults[0].subtype, /outer_race__high/);
view.live({candidate: null, regions: []});
sent = calls.filter((call) => call[0] === 'post').at(-1)[1];
assert.equal(sent.presetId, 'normal');
assert.deepEqual(Array.from(sent.regions), []);
assert.deepEqual(Array.from(sent.faults), []);
view.hide();
assert.equal(frame.src.match(/nonce=/g).length, 1);
console.log('Digital twin parent bridge passed.');
