const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(require.resolve('../ship_fault/web/digital_twin.html'), 'utf8');
assert.match(html, /front:\{label:"Front",position:\[0,\.62,1\.6\]\}/);
assert.match(html, /autoRotateSpeed:\.65,minDistance:\.35/);
const start = html.indexOf('function TwinFaultLabels(');
const end = html.indexOf('function E4(n){', start);
assert.ok(start > 0 && end > start, 'generated twin must contain the 3D callout component');
const points = [
  {id: 'pde', pos: [-0.2, 0.3, 0], anchor: [-0.06, 0.27, 0]},
  {id: 'pnde', pos: [0.3, 0.3, 0], anchor: [0.26, 0.27, 0]},
  {id: 'rot', pos: [0.1, 0.5, 0], anchor: [0.1, 0.27, 0]},
  {id: 'fnd', pos: [0.3, 0.1, 0], anchor: [0.26, 0.12, 0]},
];
const L = {jsx: (type, props, key) => ({type, props, key}),
  jsxs: (type, props, key) => ({type, props, key})};
const window = {__twinFaults: [], __twinMode: 'manual'};
const callouts = vm.runInNewContext(`${html.slice(start, end)}; TwinFaultLabels`,
  {L, vB: 'Html', FP_LABELS: points, window, Float32Array});
assert.equal(callouts({revision: 1}), null, 'normal has no callout');

window.__twinFaults = [{subtype: 'Outer Race Fault - Low Severity', shortSubtype: 'Outer Race · Low',
  regions: ['bearings'], review: true, count: 1}];
let tree = callouts({revision: 2});
assert.equal(tree.props.children.length, 1);
let bearing = tree.props.children[0];
assert.deepEqual(Array.from(bearing.props.position), points[0].pos);
assert.equal(bearing.props.children.filter((child) => child.type === 'line').length, 2,
  'bearing callout must point to both bearing positions when side is unknown');
let card = bearing.props.children.at(-1).props.children;
assert.equal(card.props.children[0].props.children, 'BEARINGS');
assert.equal(card.props.children[1].props.children, 'Outer Race · Low');
assert.equal(card.props.children.filter(Boolean).at(-1).props.children, 'Review');
assert.match(card.props['aria-label'], /Outer Race Fault - Low Severity/);
assert.equal(bearing.props.children.at(-1).props.distanceFactor, 1);

window.__twinFaults = [
  {subtype: 'Outer Race Fault - Low Severity', shortSubtype: 'Outer Race · Low',
    regions: ['bearings'], review: false, count: 1},
  {subtype: 'Inner Race Fault - High Severity', shortSubtype: 'Inner Race · High',
    regions: ['bearings'], review: true, count: 1},
];
tree = callouts({revision: 3});
card = tree.props.children[0].props.children.at(-1).props.children;
assert.equal(card.props.children[1].props.children, 'Outer Race · Low');
assert.equal(card.props.children[2].props.children, '+1 more faults');
assert.equal(card.props.children.filter(Boolean).at(-1).props.children, 'Review');
assert.equal(card.props.children.filter(Boolean).length, 4,
  'multiple faults still use one compact card per region');

window.__twinFaults = [{subtype: 'Cocked Bearing - With Loader', shortSubtype: 'Cocked Bearing',
  regions: ['bearingHousings'], review: true, count: 1}];
tree = callouts({revision: 3});
const housing = tree.props.children[0];
assert.ok(Math.abs(housing.props.position[0] - (points[1].pos[0] - 0.16)) < 1e-9);
assert.equal(housing.props.children.filter((child) => child.type === 'line').length, 2);

window.__twinFaults = [
  {subtype: 'Foundation Looseness - Level 2', shortSubtype: 'Foundation · Level 2',
    regions: ['rotorBase'], review: false, count: 1},
  {subtype: 'Dynamic Unbalance - 90 degrees', shortSubtype: 'Unbalance · 90°',
    regions: ['rotorDiscs'], review: false, count: 1},
];
window.__twinMode = 'live';
tree = callouts({revision: 4});
assert.equal(tree.props.children.length, 2);
assert.deepEqual(Array.from(tree.props.children[0].props.position), points[2].pos);
assert.ok(Math.abs(tree.props.children[1].props.position[0] - (points[3].pos[0] - 0.10)) < 1e-9);
console.log('Digital twin 3D callout anchors passed.');
