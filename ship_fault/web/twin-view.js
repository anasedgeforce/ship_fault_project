// The iframe is sandboxed with an opaque origin. Send only prediction labels and component IDs.
(function (root) {
  const map = root.SimulatorTwinMap;
  let frame, panel, list, status, nonce, ready = false;
  let current = {mode: 'manual', presetId: 'normal', regions: [], faults: []};

  function send() {
    if (!ready || !frame?.contentWindow) return;
    frame.contentWindow.postMessage({type: 'simulator-twin/update', version: 1, nonce, ...current}, '*');
  }
  function mount() {
    if (frame.src && !frame.src.endsWith('about:blank')) return;
    nonce = root.crypto.randomUUID();
    frame.src = `/assets/digital-twin.html?embedded=1&v=6&nonce=${encodeURIComponent(nonce)}`;
  }
  function show() {
    panel.classList.remove('hidden');
    mount();
    send();
  }
  function renderLines(lines) {
    list.replaceChildren();
    for (const line of lines) {
      const item = document.createElement('li');
      item.textContent = line;
      list.append(item);
    }
  }
  function faultLabels(rows, label = (value) => value) {
    const groups = new Map();
    for (const row of rows) {
      for (const fault of row.candidates) {
        if (!groups.has(fault.subtype)) groups.set(fault.subtype,
          {fault, count: 0, review: false});
        const group = groups.get(fault.subtype);
        group.count += 1;
        group.review ||= row.review;
      }
    }
    return [...groups.values()].map(({fault, count, review}) => ({
      subtype: label(fault.subtype), shortSubtype: map.compactLabel(fault.subtype),
      regions: fault.regions, count, review,
    }));
  }
  function manual(entries, label = (value) => value) {
    document.getElementById('prediction-panel').after(panel);
    const rows = map.manualEntries(entries);
    const regions = map.regionsFor(rows);
    const primary = rows.flatMap((row) => row.candidates)[0];
    current = {mode: 'manual', presetId: primary?.presetId || 'normal', regions,
      faults: faultLabels(rows, label)};
    const bearingNote = regions.some((key) => key === 'bearings' || key === 'bearingHousings')
      ? ' Exact bearing side is unknown.' : '';
    status.textContent = regions.length
      ? `Highlighted: ${regions.map((key) => map.componentNames[key]).join(', ')}.${bearingNote}`
      : 'No fault region highlighted.';
    renderLines(rows.map((row) => {
      const faults = row.candidates.length ? row.candidates.map((c) => label(c.subtype)).join(' + ')
        : row.invalid ? 'No valid prediction' : 'Normal';
      const independent = row.candidates.some((fault) => fault.source === 'independent_subtype');
      return `${row.name}: ${faults}${independent ? ' (Independent Subtype Candidate)' : ''}${row.review ? ' (Review Required)' : ''}${row.mixed ? ' (Mixed windows)' : ''}`;
    }));
    show();
  }
  function live(snapshot, label = (value) => value) {
    document.getElementById('live-panel').after(panel);
    const fault = snapshot.candidate;
    current = {mode: 'live', presetId: fault?.presetId || 'normal', regions: snapshot.regions,
      faults: fault ? faultLabels([{candidates: [fault], review: fault.review}], label) : []};
    const bearingNote = fault?.regions.some((key) => key === 'bearings' || key === 'bearingHousings')
      ? ' Exact bearing side is unknown.' : '';
    status.textContent = fault
      ? `Highlighted: ${fault.regions.map((key) => map.componentNames[key]).join(', ')}.${bearingNote}`
      : 'Watching vibration stream. No active fault highlight.';
    renderLines(fault ? [`Latest candidate: ${label(fault.subtype)}${fault.source === 'independent_subtype' ? ' (Independent Subtype Candidate)' : ''}${fault.review ? ' (Review Required)' : ''}`]
      : ['No active fault candidate.']);
    show();
  }
  function hide() {
    current = {mode: 'manual', presetId: 'normal', regions: [], faults: []};
    send();
    panel.classList.add('hidden');
  }
  function init() {
    frame = document.getElementById('digital-twin-frame');
    panel = document.getElementById('digital-twin-panel');
    list = document.getElementById('twin-fault-list');
    status = document.getElementById('twin-status');
    root.addEventListener('message', (event) => {
      if (event.source !== frame.contentWindow || event.data?.type !== 'simulator-twin/ready'
          || event.data.version !== 1 || event.data.nonce !== nonce) return;
      ready = true;
      send();
    });
  }
  root.SimulatorTwinView = {init, manual, live, hide};
})(window);
