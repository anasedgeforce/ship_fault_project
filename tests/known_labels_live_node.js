// Run with the local server on port 8000: node tests/known_labels_live_node.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function main() {
  const root = path.join(__dirname, '..');
  const labels = JSON.parse(fs.readFileSync(path.join(root, 'data/unseen_test/known_labels.json')));
  const selected = Object.keys(labels);
  const labelInput = {files: [{text: async () => JSON.stringify(labels)}]};
  const context = vm.createContext({document: {
    addEventListener() {}, getElementById(id) { return id === 'labels-input' ? labelInput : null; },
  }});
  vm.runInContext(fs.readFileSync(path.join(root, 'ship_fault/web/app.js'), 'utf8'), context);
  const parsed = await vm.runInContext('readUploadLabels()', context);
  assert.equal(Object.keys(parsed).length, 8);
  const aggregate = vm.runInContext('aggregateUploadResults', context);
  const addAccuracy = vm.runInContext('addUploadAccuracy', context);
  const entries = [];
  for (const relative of selected) {
    const csv = fs.readFileSync(path.join(root, 'data/unseen_test', relative));
    const response = await fetch('http://127.0.0.1:8000/v1/predict/input', {
      method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: csv,
    });
    assert.equal(response.status, 200, `Prediction failed for ${relative}`);
    entries.push({name: `unseen_test/${relative}`, result: await response.json()});
  }
  const summary = aggregate(entries);
  addAccuracy(summary, parsed);
  assert.equal(summary.upload_accuracy.labeled_recordings, selected.length);
  assert.equal(summary.upload_accuracy.provided_labels, 8);
  assert.equal(summary.upload_accuracy.unmatched_label_paths.length, 0);
  for (const file of summary.files) {
    const truth = labels[file.name.slice('unseen_test/'.length)];
    assert.equal(file.family_correct, file.family.label === truth.family);
    assert.equal(file.detail_correct, file.detailed_condition.label === truth.detail);
  }
  console.log(JSON.stringify({accuracy: summary.upload_accuracy,
    predictions: summary.files.map((file) => ({file: file.name,
      predicted_family: file.family.label, predicted_subtype: file.detailed_condition.label,
      family_correct: file.family_correct, subtype_correct: file.detail_correct}))}, null, 2));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
