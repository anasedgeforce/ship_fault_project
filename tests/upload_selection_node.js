// Run with: node tests/upload_selection_node.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function fileInput() {
  let value = '';
  return {files: [], get value() { return value; }, set value(next) {
    value = next;
    if (!next) this.files = [];
  }};
}
const elements = {
  'file-input': fileInput(), 'folder-input': fileInput(), 'upload-count': {},
  'upload-summary': {classList: {add() {}}}, 'forecast-panel': {classList: {add() {}}},
  result: {classList: {add() {}}}, download: {classList: {add() {}}},
  'recording-duration-card': {classList: {add() {}}},
};
const context = vm.createContext({document: {
  addEventListener() {}, getElementById(id) { return elements[id]; },
}});
vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'ship_fault', 'web', 'app.js'), 'utf8'), context);
const select = vm.runInContext('selectUpload', context);
const chosen = vm.runInContext('chosenUploadFiles', context);
const kind = vm.runInContext('uploadKind', context);

elements['file-input'].files = [{name: 'one.csv'}];
select('single');
assert.equal(kind(), 'single');
assert.equal(chosen()[0].name, 'one.csv');
elements['folder-input'].files = [{name: 'first.csv'}, {name: 'notes.txt'}, {name: 'second.CSV'}];
select('folder');
assert.equal(kind(), 'folder');
assert.equal(elements['file-input'].files.length, 0);
assert.deepEqual(Array.from(chosen(), (file) => file.name), ['first.csv', 'second.CSV']);
elements['file-input'].files = [{name: 'new.json'}];
select('single');
assert.equal(kind(), 'single');
assert.equal(elements['folder-input'].files.length, 0);
assert.equal(chosen()[0].name, 'new.json');
console.log('Upload selection passed');
