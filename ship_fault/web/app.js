const $ = (id) => document.getElementById(id);
const twinView = globalThis.SimulatorTwinView || null;
const liveTwinState = globalThis.SimulatorTwinMap ? new globalThis.SimulatorTwinMap.LiveState() : null;
let days = [];
let lastResult = null;
let lastUploadReport = null;
let activeUploadKind = null;
let modelInfo = null;
let liveTimer = null;
let lastLiveWindow = -1;
let liveHistory = [];
let watchedStreamId = null;
let watchedSessionId = null;
let livePolling = false;
let lastLiveAlert = null;

async function api(path, options = {}) {
  const key = $('api-key')?.value.trim();
  const headers = {...(options.headers || {})};
  if (key) headers['X-API-Key'] = key;
  const response = await fetch(path, {cache: 'no-store', ...options, headers});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`);
  return data;
}

function fillSelect(select, values, label, key = 'name') {
  select.replaceChildren();
  if (!values.length) {
    select.add(new Option(label, ''));
    select.disabled = true;
    return;
  }
  select.disabled = false;
  values.map((item, index) => ({item, index}))
    .sort((a, b) => String(a.item.display_name || a.item[key]).localeCompare(String(b.item.display_name || b.item[key]), undefined, {numeric: true, sensitivity: 'base'}))
    .forEach(({item, index}) => select.add(new Option(item.display_name || item[key], String(index))));
}

function selectedDay() { return days[Number($('day-select').value)]; }
function selectedCondition() { return selectedDay()?.conditions[Number($('condition-select').value)]; }
function selectedRecording() { return selectedCondition()?.recordings[Number($('file-select').value)]; }

function updateConditions() {
  fillSelect($('condition-select'), selectedDay()?.conditions || [], 'No conditions');
  updateFiles();
}
function updateFiles() {
  fillSelect($('file-select'), selectedCondition()?.recordings || [], 'No recordings');
}

async function loadCatalog() {
  try {
    const data = await api('/v1/recordings');
    days = data.days;
    fillSelect($('day-select'), days, 'No recordings found');
    updateConditions();
  } catch (error) {
    showStatus(error.message, 'error');
  }
}

function showStatus(message, kind = '') {
  $('status').textContent = message;
  $('status').className = `notice ${kind}`;
}

function setAnalyzing(active, message = 'Analyzing recording...') {
  $('analysis-loader').classList.toggle('hidden', !active);
  $('analysis-loader-message').textContent = message;
  document.querySelector('main').inert = active;
  document.querySelector('main').setAttribute('aria-busy', String(active));
  document.body.classList.toggle('is-analyzing', active);
}

function revealResult(id) {
  const panel = $(id).closest('.results-panel');
  panel.classList.remove('result-reveal');
  requestAnimationFrame(() => {
    panel.classList.add('result-reveal');
    panel.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start'});
  });
}

function updatePasteCount() {
  const value = $('paste-input').value.trim();
  const lines = value ? value.split(/\r?\n/).length : 0;
  $('paste-count').textContent = lines ? `${lines.toLocaleString()} lines in the paste box` : '';
}

async function loadSynthetic(name, button) {
  button.disabled = true;
  showStatus('Loading the full synthetic recording…', 'busy');
  try {
    const response = await fetch(`/v1/demo-inputs/${name}`, {cache: 'no-store'});
    if (!response.ok) throw new Error('Could not load the synthetic recording.');
    $('paste-input').value = await response.text();
    updatePasteCount();
    showStatus('Full synthetic recording loaded. Click Run Analysis.');
  } catch (error) {
    showStatus(error.message, 'error');
  } finally {
    button.disabled = false;
  }
}

function familyDetailDisagree(family, detail) {
  return Boolean(family?.label && detail?.label && detail.label.split('__')[0] !== family.label);
}

function displayLabel(label) {
  if (!label) return 'Unavailable';
  return modelInfo?.label_catalog?.labels?.[label] || label.replaceAll('__', ' — ').replaceAll('_', ' ').replace(/\b[a-z]/g, c => c.toUpperCase());
}

function displayDecision(code) {
  return modelInfo?.label_catalog?.decisions?.[code] || ({classified: 'Prediction Available', needs_review: 'Review Required', invalid_input: 'Invalid Input', failed: 'Analysis Failed'}[code]) || 'Unavailable';
}

function displayReviewReason(code) {
  return modelInfo?.label_catalog?.review_reasons?.[code] || code.replaceAll('_', ' ');
}

function scoreTone(score) {
  if (!Number.isFinite(score)) return '';
  if (score > 0.70) return 'score-high';
  if (score >= 0.50) return 'score-medium';
  return 'score-low';
}

function boxScore(element, score) {
  const tone = scoreTone(score);
  if (!tone) return;
  const value = document.createElement('span');
  value.className = `score-value ${tone}`;
  value.textContent = element.textContent;
  element.replaceChildren(value);
}

function renderScoreLine(id, label, score, note = '') {
  const element = $(id);
  if (!Number.isFinite(score)) {
    element.textContent = note;
    return;
  }
  const caption = document.createElement('span');
  caption.className = 'score-caption';
  caption.textContent = label;
  const value = document.createElement('span');
  value.className = `score-value ${scoreTone(score)}`;
  value.textContent = score.toFixed(3);
  const parts = [caption, value];
  if (note) {
    const noteElement = document.createElement('span');
    noteElement.className = 'score-note';
    noteElement.textContent = note.trimStart();
    parts.push(noteElement);
  }
  element.replaceChildren(...parts);
}

function renderResult(result, live = false, recordingName = 'Selected recording') {
  lastResult = result;
  const rows = live ? result.predictions : [];
  $('prediction-window-summary').classList.toggle('hidden', !live);
  $('prediction-window-table').classList.toggle('hidden', !live);
  const scoreLabel = live ? 'Candidate Model Score' : 'Whole-Recording Model Score';
  const family = live ? result.latest.family : result.recording_family;
  const detail = live ? result.latest.detailed_condition : result.recording_detailed_condition;
  const decision = live ? result.latest.decision : result.recording_decision;
  $('result-rows').replaceChildren();
  for (const prediction of rows) {
    const tr = document.createElement('tr');
    const start = live ? (prediction.elapsed_start_seconds ?? prediction.start_seconds) : prediction.start_seconds;
    const values = [Number(start).toFixed(3), prediction.family ? displayLabel(prediction.family.label) : 'Invalid Input',
      prediction.family ? Number(prediction.family.score).toFixed(3) : '—',
      displayLabel(prediction.detailed_condition?.label),
      prediction.detailed_condition ? Number(prediction.detailed_condition.score).toFixed(3) : '—'];
    for (const [index, value] of values.entries()) {
      const td = document.createElement('td');
      td.textContent = value;
      if (index === 2) boxScore(td, prediction.family?.score);
      if (index === 4) boxScore(td, prediction.detailed_condition?.score);
      tr.append(td);
    }
    $('result-rows').append(tr);
  }
  $('family-label').textContent = family ? displayLabel(family.label) : 'No valid window';
  const familyReviewNote = decision === 'needs_review' && family?.score <= 0.75
    ? ' · review required' : '';
  renderScoreLine('family-score', scoreLabel, family?.score,
    family ? familyReviewNote : 'No model prediction');
  $('detail-label').textContent = displayLabel(detail?.label);
  const independentDetail = (live ? result.latest.independent_detailed_condition
    : result.independent_recording_detailed_condition) || detail;
  const independentFault = family?.label === 'normal' && independentDetail?.label
    && independentDetail.label.split('__')[0] !== 'normal';
  const detailNeedsInlineReview = decision === 'needs_review' && detail?.score <= 0.75;
  const detailNote = independentFault
    ? `; independent subtype candidate: ${displayLabel(independentDetail.label)}; review required`
    : familyDetailDisagree(family, independentDetail)
    ? ` · independent subtype model disagreed${detailNeedsInlineReview ? '; review required' : ''}`
    : detailNeedsInlineReview ? ' · review required' : '';
  renderScoreLine('detail-score', scoreLabel, detail?.score,
    detail ? detailNote : 'No valid subtype prediction');
  $('window-count').textContent = String(live ? result.latest.window_index + 1 : result.window_count);
  const duration = live ? result.recording_time?.duration_seconds : result.recording_duration_seconds;
  const durationText = Number.isFinite(duration) ? `${duration.toFixed(2)} s` : 'Unavailable';
  $('recording-duration').textContent = durationText;
  $('live-recording-duration').textContent = durationText;
  $('recording-duration-card').classList.toggle('hidden', live);
  $('live-recording-duration-card').classList.toggle('hidden', !live);
  $('result-note').textContent = live ? 'Recent live prediction windows.' : 'One overall result for this recording. Scores combine its valid analysis windows.';
  $('result').classList.remove('hidden');
  $('download').classList.remove('hidden');
  showStatus(live ? (result.latest.decision === 'classified' ? 'Live prediction updated.' : 'Live data review required.') : 'Prediction complete. Equipment suitability and physical sensor-axis mapping have not been verified for this recording.');
  if (!live && twinView) {
    const windows = (result.predictions || []).map((item) => ({family: item.family?.label || null,
      detailed_condition: item.detailed_condition?.label || null}));
    const pairs = new Set(windows.filter((item) => item.family && item.detailed_condition)
      .map((item) => JSON.stringify([item.family, item.detailed_condition])));
    twinView.manual([{name: recordingName, family, detailed_condition: detail,
      independent_detailed_condition: result.independent_recording_detailed_condition,
      decision, mixed_windows: pairs.size > 1, windows}], displayLabel);
  }
  if (!live) revealResult('result');
}

function uploadKind() {
  return activeUploadKind;
}

function selectUpload(kind) {
  const input = $(kind === 'folder' ? 'folder-input' : 'file-input');
  if (input.files.length) {
    activeUploadKind = kind;
    $(kind === 'folder' ? 'file-input' : 'folder-input').value = '';
  } else if (activeUploadKind === kind) {
    activeUploadKind = null;
  }
  clearUploadSummary();
  twinView?.hide();
  $('result').classList.add('hidden');
  $('download').classList.add('hidden');
  $('recording-duration-card').classList.add('hidden');
  updateUploadCount();
}

function clearUploadSummary() {
  lastUploadReport = null;
  $('upload-summary').classList.add('hidden');
}

function chosenUploadFiles() {
  if (uploadKind() === 'folder')
    return Array.from($('folder-input').files).filter((file) => /\.csv$/i.test(file.name));
  return uploadKind() === 'single' ? Array.from($('file-input').files) : [];
}

async function readUploadLabels() {
  const file = $('labels-input').files[0];
  if (!file) return null;
  let labels;
  try { labels = JSON.parse(await file.text()); }
  catch (_) { throw new Error('Known-label file must contain valid JSON.'); }
  if (labels?.purpose === 'metadata_only_not_known_labels_for_accuracy')
    throw new Error('This JSON contains synthetic recording metadata, not verified fault labels. Clear the label file to analyze these recordings.');
  if (!labels || Array.isArray(labels) || typeof labels !== 'object')
    throw new Error('Known-label JSON must map recording paths to family and detail labels.');
  for (const [path, value] of Object.entries(labels)) {
    if (!path || !value || typeof value.family !== 'string' || typeof value.detail !== 'string'
      || !value.family.trim() || !value.detail.trim()
      || value.detail.split('__', 1)[0] !== value.family)
      throw new Error(`Invalid family/detail labels for ${path || 'a recording'}.`);
  }
  return labels;
}

function addUploadAccuracy(summary, labels) {
  let familyCorrect = 0, detailCorrect = 0, matched = 0;
  const matchedPaths = new Set();
  for (const file of summary.files) {
    const parts = file.name.split('/');
    const relative = parts.length > 1 ? parts.slice(1).join('/') : file.name;
    const labelPath = labels && Object.hasOwn(labels, relative) ? relative
      : parts.length === 1 && labels && Object.hasOwn(labels, file.name) ? file.name : null;
    const truth = labelPath === null ? null : labels[labelPath];
    if (!truth || !file.family || !file.detailed_condition) continue;
    matchedPaths.add(labelPath);
    matched += 1;
    file.actual_family = truth.family;
    file.actual_detail = truth.detail;
    file.family_correct = file.family.label === truth.family;
    file.detail_correct = file.detailed_condition.label === truth.detail;
    familyCorrect += Number(file.family_correct);
    detailCorrect += Number(file.detail_correct);
  }
  summary.upload_accuracy = {labeled_recordings: matched, family_correct: familyCorrect,
    detail_correct: detailCorrect, family: matched ? familyCorrect / matched : null,
    detail: matched ? detailCorrect / matched : null,
    provided_labels: labels ? Object.keys(labels).length : 0,
    unmatched_label_paths: labels ? Object.keys(labels).filter((path) => !matchedPaths.has(path)) : []};
}

function updateUploadCount() {
  const files = chosenUploadFiles();
  $('upload-count').textContent = files.length
    ? `${uploadKind() === 'folder' ? 'Folder' : 'File'} selected: ${files.length.toLocaleString()} ${files.length === 1 ? 'recording' : 'recordings'}. Each recording is predicted separately.`
    : (uploadKind() === 'folder' ? 'The selected folder contains no CSV recordings.' : 'Choose a file or a folder. Your most recent selection is used.');
}

function aggregateUploadResults(entries) {
  const groups = new Map();
  let validWindows = 0, totalWindows = 0, successfulFiles = 0;
  let faultCandidateFiles = 0, reviewFiles = 0;
  const files = entries.map((entry) => {
    const result = entry.result;
    const pathParts = entry.name.split('/').filter(Boolean);
    const equipment = pathParts.length >= 3 ? pathParts[1] : null;
    if (!result) return {name: entry.name, equipment_folder: equipment,
      family: null, detailed_condition: null, decision: 'failed', finding: 'Unavailable',
      valid_window_count: 0, error: entry.error || null, windows: []};
    successfulFiles += 1;
    const weight = result.valid_window_count || 0;
    validWindows += weight;
    totalWindows += result.window_count || 0;
    const family = result.recording_family || null;
    const detail = result.recording_detailed_condition || null;
    const windows = result.predictions.map((item) => ({start_seconds: item.start_seconds,
      family: item.family?.label || null, detailed_condition: item.detailed_condition?.label || null,
      independent_detailed_condition: item.independent_detailed_condition?.label || null,
      decision: item.decision, review_reasons: item.review_reasons || []}));
    const pairs = new Set(windows.filter((item) => item.family && item.detailed_condition)
      .map((item) => JSON.stringify([item.family, item.detailed_condition])));
    const mixed = pairs.size > 1;
    const independentDetail = result.independent_recording_detailed_condition || null;
    const independentFault = family?.label === 'normal' && independentDetail?.label
      && independentDetail.label.split('__')[0] !== 'normal';
    const primaryFault = windows.some((item) => (item.family && item.family !== 'normal') ||
      (item.detailed_condition && item.detailed_condition.split('__')[0] !== 'normal'));
    const hasFault = primaryFault || independentFault;
    if (hasFault) faultCandidateFiles += 1;
    if (result.recording_decision !== 'classified' || mixed) reviewFiles += 1;
    const groupable = Boolean(equipment && weight > 0 && family && detail && family.label !== 'normal'
      && result.recording_decision === 'classified' && windows.length === weight && !mixed
      && !familyDetailDisagree(family, detail)
      && windows.every((item) => item.decision === 'classified' && item.family === family.label
        && item.detailed_condition === detail.label)
      && Number.isFinite(family.score) && Number.isFinite(detail.score));
    if (groupable) {
      const key = JSON.stringify([equipment, family.label, detail.label]);
      if (!groups.has(key)) groups.set(key, {equipment_folder: equipment,
        family: family.label, detailed_condition: detail.label, files: [],
        family_score_total: 0, detail_score_total: 0});
      const group = groups.get(key);
      group.files.push(entry.name);
      group.family_score_total += family.score;
      group.detail_score_total += detail.score;
    }
    return {name: entry.name, equipment_folder: equipment, family, detailed_condition: detail,
      independent_detailed_condition: independentDetail,
      decision: result.recording_decision || 'failed', mixed_windows: mixed,
      finding: !weight ? 'Unavailable' : mixed ? 'Mixed signals; review required'
        : independentFault && !primaryFault ? 'Independent subtype candidate; review required'
        : hasFault ? 'Fault candidate' : 'Normal candidate',
      valid_window_count: weight, error: entry.error || null, windows};
  });
  const matchingGroups = Array.from(groups.values()).filter((group) => group.files.length > 1)
    .map((group) => ({equipment_folder: group.equipment_folder, family: group.family,
      detailed_condition: group.detailed_condition, files: group.files,
      family_score_mean: group.family_score_total / group.files.length,
      detailed_score_mean: group.detail_score_total / group.files.length}));
  return {
    file_count: entries.length, successful_file_count: successfulFiles,
    failed_file_count: entries.length - successfulFiles,
    valid_window_count: validWindows, window_count: totalWindows,
    fault_candidate_files: faultCandidateFiles, review_file_count: reviewFiles,
    matching_groups: matchingGroups, files,
  };
}

function renderUploadSummary(summary) {
  lastUploadReport = summary;
  twinView?.manual(summary.files, displayLabel);
  $('upload-summary').classList.remove('hidden');
  $('upload-summary-note').textContent = `${summary.successful_file_count} of ${summary.file_count} recordings analyzed successfully. Equipment suitability and physical sensor-axis mapping are unverified.`;
  $('upload-fault-status').textContent = !summary.valid_window_count ? 'No Valid Result'
    : summary.fault_candidate_files ? 'Fault Candidate Found'
      : (summary.review_file_count || summary.failed_file_count) ? 'Review Required' : 'Normal Candidate';
  $('upload-fault-count').textContent = `${summary.fault_candidate_files} recordings contain a non-normal candidate; ${summary.review_file_count} need review.`;
  $('upload-file-count').textContent = String(summary.successful_file_count);
  $('upload-window-count').textContent = `One overall result per CSV; ${summary.failed_file_count} files failed`;
  const reference = modelInfo?.primary_recording_validation_accuracy ?? modelInfo?.primary_validation_accuracy;
  $('upload-validation-accuracy').textContent = Number.isFinite(reference)
    ? `${(reference * 100).toFixed(2)}%` : 'Unavailable';
  const detailReference = modelInfo?.detailed_recording_validation_accuracy ?? modelInfo?.detailed_validation_accuracy;
  $('upload-detail-accuracy').textContent = Number.isFinite(detailReference)
    ? `${(detailReference * 100).toFixed(2)}%` : 'Unavailable';
  const measured = summary.upload_accuracy;
  const labelNote = measured?.provided_labels
    ? measured.labeled_recordings
      ? `Known labels matched ${measured.labeled_recordings} successful recordings. Family accuracy: ${(measured.family * 100).toFixed(1)}%; subtype accuracy: ${(measured.detail * 100).toFixed(1)}%. ${measured.unmatched_label_paths.length} label paths did not match a successfully analyzed CSV. `
      : 'No known-label paths matched a successfully analyzed CSV. Check paths and filenames in the JSON file. '
    : '';
  $('upload-accuracy-note').textContent = `${labelNote}Model scores are uncalibrated. Reference accuracy comes from previous day-held-out evaluation.`;
  $('upload-result-rows').replaceChildren();
  for (const file of summary.files) {
    const tr = document.createElement('tr');
    const values = [file.name, displayLabel(file.family?.label),
      file.family ? file.family.score.toFixed(3) : '—',
      displayLabel(file.detailed_condition?.label), file.detailed_condition ? file.detailed_condition.score.toFixed(3) : '—',
      file.error || `${displayDecision(file.decision)}${file.mixed_windows ? ' · mixed windows; review required' : ''}`];
    for (const [index, value] of values.entries()) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
      if (index === 2) boxScore(td, file.family?.score);
      if (index === 4) boxScore(td, file.detailed_condition?.score);
    }
    if (file.family?.label === 'normal' && file.independent_detailed_condition?.label
        && file.independent_detailed_condition.label.split('__')[0] !== 'normal') {
      const note = document.createElement('div');
      note.textContent = `Independent subtype candidate: ${displayLabel(file.independent_detailed_condition.label)}; review required`;
      tr.lastChild.append(note);
    }
    if (file.mixed_windows || familyDetailDisagree(file.family, file.independent_detailed_condition)) {
      const details = document.createElement('details');
      const title = document.createElement('summary'); title.textContent = 'View recording review reasons';
      details.append(title);
      const list = document.createElement('ul');
      const reasons = new Set(file.windows.flatMap(window => window.review_reasons));
      if (file.mixed_windows) reasons.add('mixed_fault_predictions_within_recording');
      for (const reason of reasons) {
        const item = document.createElement('li');
        item.textContent = displayReviewReason(reason);
        list.append(item);
      }
      details.append(list); tr.lastChild.append(details);
    }
    $('upload-result-rows').append(tr);
  }
  $('upload-group-rows').replaceChildren();
  $('upload-groups-table').classList.toggle('hidden', summary.matching_groups.length === 0);
  $('upload-group-note').textContent = `${summary.matching_groups.length ? 'Group scores are the average of individual file scores.' : 'No eligible groups to average.'} Only two or more fault recordings with the same predicted family and subtype, matching classified windows, and the same equipment subfolder are grouped. Folder names are supplied by the user and do not verify equipment identity. Choose a parent folder arranged as equipment-name/recordings; files directly in the selected folder stay separate.`;
  for (const group of summary.matching_groups) {
    const tr = document.createElement('tr');
    for (const [index, value] of [displayLabel(group.family), displayLabel(group.detailed_condition),
      `${group.files.length}: ${group.files.join(', ')}`, group.family_score_mean.toFixed(3), group.detailed_score_mean.toFixed(3)].entries()) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
      if (index === 3) boxScore(td, group.family_score_mean);
      if (index === 4) boxScore(td, group.detailed_score_mean);
    }
    $('upload-group-rows').append(tr);
  }
}

async function predictUploads() {
  const files = chosenUploadFiles();
  if (!files.length) return showStatus(uploadKind() === 'folder'
    ? 'Choose a folder containing CSV files.' : 'Choose a file or a folder first.', 'error');
  const entries = [];
  setAnalyzing(true);
  $('predict').disabled = true;
  $('file-input').disabled = true;
  $('folder-input').disabled = true;
  $('result').classList.add('hidden');
  $('download').classList.add('hidden');
  $('recording-duration-card').classList.add('hidden');
  clearUploadSummary();
  twinView?.hide();
  try {
    const labels = await readUploadLabels();
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      const name = file.webkitRelativePath || file.name;
      $('analysis-loader-message').textContent = `Analyzing recording ${index + 1} of ${files.length}`;
      showStatus(`Analyzing recording ${index + 1} of ${files.length}: ${name}`, 'busy');
      try {
        if (file.size > 32000000) throw new Error('File exceeds the 32 MB input limit');
        const result = await api('/v1/predict/input', {method: 'POST',
          headers: {'Content-Type': 'application/octet-stream'}, body: await file.arrayBuffer()});
        entries.push({name, result});
      } catch (error) {
        entries.push({name, error: error.message});
      }
    }
    const summary = aggregateUploadResults(entries);
    addUploadAccuracy(summary, labels);
    renderUploadSummary(summary);
    if (files.length > 1) revealResult('upload-summary');
    if (files.length === 1 && entries[0].result) renderResult(entries[0].result, false, entries[0].name);
    else showStatus(`${summary.successful_file_count} of ${summary.file_count} recordings analyzed. See the dataset summary.`,
      summary.failed_file_count ? 'error' : '');
  } catch (error) {
    showStatus(error.message, 'error');
  } finally {
    $('predict').disabled = false;
    $('file-input').disabled = false;
    $('folder-input').disabled = false;
    setAnalyzing(false);
  }
}

async function predict() {
  const source = document.querySelector('input[name="source"]:checked').value;
  if (source === 'file') return predictUploads();
  $('upload-summary').classList.add('hidden');
  let body, contentType;
  if (source === 'saved') {
    const selected = selectedRecording();
    if (!selected) return showStatus('Choose a saved recording first.', 'error');
    body = JSON.stringify({id: selected.id});
    contentType = 'application/json';
  } else {
    body = $('paste-input').value;
    if (!body.trim()) return showStatus('Paste a complete recording first.', 'error');
    if (!/^[\[{]/.test(body.trimStart()) && body.trim().split(/\r?\n/).length < 4001)
      return showStatus('A single row is not enough. Paste the CSV header plus at least 4,000 data rows, or click Load Smooth Sample.', 'error');
    contentType = 'text/plain; charset=utf-8';
  }
  $('predict').disabled = true;
  setAnalyzing(true);
  $('result').classList.add('hidden');
  $('download').classList.add('hidden');
  $('recording-duration-card').classList.add('hidden');
  twinView?.hide();
  showStatus('Analyzing recording…', 'busy');
  try {
    renderResult(await api(source === 'saved' ? '/v1/predict/saved' : '/v1/predict/input',
      {method: 'POST', headers: {'Content-Type': contentType}, body}));
  } catch (error) {
    showStatus(error.message, 'error');
  } finally {
    $('predict').disabled = false;
    setAnalyzing(false);
  }
}

function showLiveTwin() {
  if ($('manual-tab')?.classList.contains('active')) return;
  twinView?.live(liveTwinState?.snapshot() || {candidate: null, regions: []}, displayLabel);
}

function selectMode(mode) {
  const manual = mode === 'manual';
  $('manual-workspace').classList.toggle('hidden', !manual);
  $('context-mode').textContent = manual ? 'Recording Analysis' : 'Live Sensor Stream';
  $('manual-panel').classList.toggle('hidden', !manual);
  $('live-panel').classList.toggle('hidden', manual);
  $('result').classList.add('hidden');
  $('download').classList.add('hidden');
  $('recording-duration-card').classList.add('hidden');
  $('result-note').textContent = manual
    ? 'Results appear here after a recording is processed.'
    : 'Recent live prediction windows appear here after a stream is watched.';
  showStatus(manual ? 'Select an input above and run a prediction.' : 'Watch a stream to see live predictions.');
  if (!manual) {
    $('upload-summary').classList.add('hidden');
    $('result').classList.add('hidden');
    $('download').classList.add('hidden');
  }
  else if (document.querySelector('input[name="source"]:checked').value === 'file' && lastUploadReport) {
    $('upload-summary').classList.remove('hidden');
  }
  $('manual-tab').classList.toggle('active', manual);
  $('live-tab').classList.toggle('active', !manual);
  $('manual-tab').setAttribute('aria-selected', String(manual));
  $('live-tab').setAttribute('aria-selected', String(!manual));
  if (manual && liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  if (manual) twinView?.hide();
  else {
    watchedStreamId = null;
    watchedSessionId = null;
    lastLiveWindow = -1;
    liveHistory = [];
    lastLiveAlert = null;
    liveTwinState?.reset();
    showLiveTwin();
  }
}

function streamId() {
  const id = $('stream-id').value.trim();
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(id)) throw new Error('Stream ID needs 1–64 letters, numbers, underscores, or hyphens.');
  $('stream-endpoint').textContent = `/v1/streams/${id}/samples`;
  return id;
}

function drawSeries(id, values, color) {
  const canvas = $(id);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0a0e13';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!values.length) return;
  let low = Math.min(...values), high = Math.max(...values);
  if (high === low) { high += 1; low -= 1; }
  ctx.strokeStyle = '#33414d';
  ctx.beginPath(); ctx.moveTo(0, canvas.height / 2); ctx.lineTo(canvas.width, canvas.height / 2); ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  values.forEach((value, index) => {
    const x = index * (canvas.width - 12) / Math.max(1, values.length - 1) + 6;
    const y = canvas.height - 8 - (value - low) / (high - low) * (canvas.height - 16);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderLiveMonitor(data, diagnostics) {
  $('live-monitor').classList.remove('hidden');
  const latest = data.latest;
  const expected = $('equipment-id').textContent.trim();
  const mismatch = expected && data.equipment_id && expected !== data.equipment_id;
  $('live-decision').textContent = mismatch ? 'Equipment ID mismatch — check stream selection'
    : latest ? `${displayDecision(latest.decision)}${latest.review_reasons.length ? ': ' + latest.review_reasons.map(displayReviewReason).join(' ') : ''}`
      : 'Waiting for first full window';
  const independentFault = latest?.family?.label === 'normal'
    && latest?.independent_detailed_condition?.label
    && latest.independent_detailed_condition.label.split('__')[0] !== 'normal';
  $('live-candidate').textContent = independentFault
    ? `${displayLabel(latest.independent_detailed_condition.label)} (independent subtype candidate; family model: Normal; review required)`
    : latest?.family && latest?.detailed_condition
      ? `${displayLabel(latest.family.label)} / ${displayLabel(latest.detailed_condition.label)}${latest.decision === 'needs_review' ? ' (review required)' : ''}`
    : 'No valid prediction yet';
  $('live-quality').textContent = latest
    ? (latest.quality.issues.length ? latest.quality.issues.map((issue) => issue.message).join('; ') : 'Six channels passed waveform checks')
    : 'Waiting for 4,000 samples';
  const ref = latest?.mongo_references?.at(-1);
  $('live-source').textContent = ref ? `${ref.database}.${ref.collection} / ${ref.document_id}` : 'No saved batch reference yet';
  if (data.alerts?.length) lastLiveAlert = data.alerts.at(-1);
  $('live-alert').textContent = lastLiveAlert
    ? `${lastLiveAlert.fault_type} / ${lastLiveAlert.fault_subtype} after ${lastLiveAlert.consecutive_windows} windows`
    : 'No persistent fault alert';
  const time = data.recording_time;
  const sourceTime = Number.isFinite(time?.source_timestamp_ms)
    && Number.isFinite(new Date(time.source_timestamp_ms).getTime())
      ? `Source timestamp: ${new Date(time.source_timestamp_ms).toISOString()}`
      : 'Absolute source time not provided';
  $('live-time').textContent = time ? `${time.duration_seconds.toFixed(2)} s of sensor time. ${sourceTime}.`
    : 'Recording time unavailable';
  drawSeries('waveform-chart', diagnostics.waveform.map((point) => point.channels[0]), '#126fa7');
  drawSeries('spectrum-chart', diagnostics.spectrum.map((point) => point.magnitude), '#9461a8');
  drawSeries('trend-chart', diagnostics.trends.map((point) => point.rms_mm_s[0]), '#be772b');
  return mismatch;
}

async function pollStream() {
  if (livePolling) return;
  livePolling = true;
  try {
    const id = streamId();
    if (id !== watchedStreamId) {
      watchedStreamId = id;
      watchedSessionId = null;
      lastLiveWindow = -1;
      liveHistory = [];
      lastLiveAlert = null;
      $('result').classList.add('hidden');
      $('download').classList.add('hidden');
      $('live-monitor').classList.add('hidden');
      lastResult = null;
      liveTwinState?.reset();
      showLiveTwin();
    }
    let data = await api(`/v1/streams/${id}?after=${lastLiveWindow}`);
    if (data.session_id !== watchedSessionId) {
      watchedSessionId = data.session_id;
      lastLiveWindow = -1;
      liveHistory = [];
      lastLiveAlert = null;
      liveTwinState?.reset();
      showLiveTwin();
      data = await api(`/v1/streams/${id}?after=-1`);
    }
    if (data.history_truncated) {
      liveHistory = [];
      liveTwinState?.reset();
    }
    if ($('manual-tab').classList.contains('active')) return;
    if (data.predictions.length) {
      for (const prediction of data.predictions) liveTwinState?.update(prediction);
      showLiveTwin();
      liveHistory.push(...data.predictions);
      liveHistory = liveHistory.slice(-100);
      lastLiveWindow = data.predictions[data.predictions.length - 1].window_index;
      renderResult({...data, predictions: liveHistory}, true);
    }
    const diagnostics = await api(`/v1/streams/${id}/diagnostics`);
    const mismatch = renderLiveMonitor(data, diagnostics);
    const note = data.history_truncated ? ' Some older windows were dropped from the display.' : '';
    const timeWarning = data.latest?.beyond_training_window_budget
      ? ' This stream is beyond the short capture window budget used in training; accuracy here is unvalidated.' : '';
    const savePending = data.persistence === 'result_save_failed';
    $('retry-live-storage').classList.toggle('hidden', !savePending);
    const saveWarning = savePending ? ' Raw readings are stored, but prediction results still need to be saved to MongoDB. Retry saving results.' : '';
    $('live-status').textContent = `${data.total_samples.toLocaleString()} samples received. ${data.ready ? 'Window results available.' : 'Waiting for 4,000 samples.'}${note}${timeWarning}${saveWarning}${mismatch ? ' Equipment ID does not match.' : ''}`;
    $('live-status').classList.toggle('warning', Boolean(timeWarning || note || mismatch || savePending));
  } catch (error) {
    $('live-status').textContent = error.message === 'stream not found' ? 'Waiting for the DAQ adapter to send its first batch.' : error.message;
  } finally {
    livePolling = false;
  }
}

async function resetStream() {
  liveTwinState?.reset();
  showLiveTwin();
  try {
    const id = streamId();
    await api(`/v1/streams/${id}`, {method: 'DELETE'});
    lastLiveWindow = -1;
    watchedSessionId = null;
    liveHistory = [];
    lastLiveAlert = null;
    lastResult = null;
    $('live-status').textContent = 'Stream reset. Send new samples to begin again.';
    $('result').classList.add('hidden');
    $('download').classList.add('hidden');
    $('live-monitor').classList.add('hidden');
    $('retry-live-storage').classList.add('hidden');
  } catch (error) {
    $('live-status').textContent = error.message;
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  twinView?.init();
  $('manual-tab').onclick = () => selectMode('manual');
  $('live-tab').onclick = () => selectMode('live');
  document.querySelectorAll('input[name="source"]').forEach((input) => input.onchange = () => {
    for (const choice of ['saved', 'file', 'paste']) $('' + choice + '-source').classList.toggle('hidden', input.value !== choice);
    $('upload-summary').classList.toggle('hidden', input.value !== 'file' || !lastUploadReport);
    $('recording-duration-card').classList.add('hidden');
  });
  $('file-input').onchange = () => selectUpload('single');
  $('folder-input').onchange = () => selectUpload('folder');
  $('labels-input').onchange = () => $('clear-labels').classList.toggle('hidden', !$('labels-input').files.length);
  $('clear-labels').onclick = () => {
    $('labels-input').value = '';
    $('clear-labels').classList.add('hidden');
    showStatus('Known-label file cleared. Run analysis without labels.', '');
  };
  $('clear-labels').classList.toggle('hidden', !$('labels-input').files.length);
  updateUploadCount();
  $('day-select').onchange = updateConditions;
  $('condition-select').onchange = updateFiles;
  $('api-key').onchange = loadCatalog;
  $('predict').onclick = predict;
  $('paste-input').oninput = updatePasteCount;
  document.querySelectorAll('.sample-load').forEach((button) => {
    button.onclick = () => loadSynthetic(button.dataset.sample, button);
  });
  $('watch-stream').onclick = () => { if (liveTimer) clearInterval(liveTimer); pollStream(); liveTimer = setInterval(pollStream, 1000); };
  $('reset-stream').onclick = resetStream;
  $('retry-live-storage').onclick = async () => {
    try {
      await api(`/v1/streams/${streamId()}/retry-storage`, {method: 'POST'});
      await pollStream();
    } catch (error) {
      $('live-status').textContent = error.message;
    }
  };
  $('stream-id').oninput = () => {
    if ($('stream-id').value.trim() !== watchedStreamId) {
      liveTwinState?.reset();
      showLiveTwin();
    }
    try { streamId(); } catch (_) {}
  };
  $('download').onclick = () => {
    if (!lastResult) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], {type: 'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = 'simulator_fault_prediction.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  $('download-upload').onclick = () => {
    if (!lastUploadReport) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(lastUploadReport, null, 2)], {type: 'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = 'simulator_fault_upload_summary.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  try {
    const ready = await api('/health/ready');
    const info = ready.models;
    modelInfo = info;
    $('equipment-id').textContent = ready.equipment_id || 'Not configured';
    $('context-model-state').textContent = 'Models Loaded';
    $('context-model-state').classList.add('state-ready');
    $('sidebar-model-state').textContent = 'Models Loaded';
    $('model-badge').textContent = `${info.family_classes} Fault Families · ${info.detailed_classes} Fault Subtypes`;
  } catch (error) {
    $('model-badge').textContent = 'Saved models unavailable';
    $('context-model-state').textContent = 'Unavailable';
    $('sidebar-model-state').textContent = 'Models Unavailable';
    showStatus(error.message, 'error');
  }
  await loadCatalog();
});
