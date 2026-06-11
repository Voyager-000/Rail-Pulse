// RailPulse Dashboard — app.js
'use strict';

let DATA = null;
let radarChart = null;
let congestionChart = null;
let selectedCoach = null;
let refreshTimer = null;
let trainMap = null;
let mapMarkers = {};
let specificTrainMap = null;
let specificTrainLayer = null;

const STATION_COORDS = {
  'NDLS': [28.6139, 77.2090], // Delhi
  'BCT':  [18.9690, 72.8205], // Mumbai
  'HWH':  [22.5855, 88.3412], // Howrah
  'SBC':  [12.9781, 77.5695], // Bangalore
  'BPL':  [23.2599, 77.4126], // Bhopal
  'SDAH': [22.5678, 88.3712], // Sealdah
  'BDTS': [19.0553, 72.8354], // Bandra
  'CSTM': [18.9398, 72.8354], // CSMT
  'FZR':  [30.9304, 74.6186], // Firozpur
  'RJPB': [25.5960, 85.1517], // Patna
  'TVC':  [8.4875,  76.9486], // Trivandrum
  'LTT':  [19.0683, 72.8906], // Kurla
  'DBRG': [27.4728, 94.9120], // Dibrugarh
  'ERS':  [9.9691,  76.2778], // Ernakulam
  'VAR':  [25.3176, 82.9739], // Varanasi
  'RTE':  [17.3850, 78.4867], // Hyderabad
};
// FALLBACK_DATA is loaded from predictions.js

async function pollLive() {
  try {
    const res = await fetch('/api/live', { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      DATA = await res.json();
      setStatus('LIVE', true);
      renderAll();
      return;
    }
  } catch (_) {}

  // Fallback: use embedded data if offline
  if (typeof FALLBACK_DATA !== 'undefined' && FALLBACK_DATA) {
      DATA = FALLBACK_DATA;
      setStatus('OFFLINE', false);
      renderAll();
      return;
  }
  setStatus('NO DATA', false);
}

function setStatus(text, live) {
  const pill = document.getElementById('status-pill');
  const dot = pill.querySelector('.status-dot');
  document.getElementById('status-text').textContent = text;
  dot.style.background = live ? '#4ade80' : '#f97316';
  pill.style.borderColor = live ? 'rgba(34,197,94,0.3)' : 'rgba(249,115,22,0.3)';
  pill.style.color = live ? '#4ade80' : '#fb923c';
}

function showError() {
  document.getElementById('heatmap-container').innerHTML =
    `<div class="loading-state"><p>⚠️ Run <code>python model/predict.py</code> to generate predictions.json</p></div>`;
}

// ── Clock ─────────────────────────────────────────────────────────────────────
function startClock() {
  const tick = () => {
    const now = new Date();
    document.getElementById('live-clock').textContent =
      now.toLocaleTimeString('en-IN', { hour12: false });
    document.getElementById('header-date').textContent =
      now.toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
  };
  tick();
  setInterval(tick, 1000);
  document.getElementById('filter-date').value = new Date().toISOString().split('T')[0];
}

// ── Tab Navigation ────────────────────────────────────────────────────────────
let activeTab = 'overview';

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.view-section').forEach(v => v.style.display = 'none');
  document.getElementById('tab-' + tab).classList.add('active');
  document.getElementById('view-' + tab).style.display = 'block';

  if (tab === 'train') {
    // Re-render the heatmap for the currently selected train
    renderSpecificTrainView();
  }
}

// ── Summary Cards ─────────────────────────────────────────────────────────────
function renderSummary() {
  const s = DATA.summary;
  const allCoaches = DATA.trains.flatMap(t => t.coaches);

  // Compute real counts live from coach data — never trust a capped static field
  const criticalCount = allCoaches.filter(c => c.risk_level === 'critical').length;
  const ticketlessCount = allCoaches.filter(c => c.ticketless_risk > 0.5).length;
  const staffCount = allCoaches.reduce((acc, c) => {
    if (c.risk_level === 'critical')      return acc + (c.overcrowding_risk > 0.85 ? 4 : c.overcrowding_risk > 0.70 ? 3 : 2);
    else if (c.risk_level === 'high')     return acc + (c.ticketless_risk > 0.6 ? 2 : 1);
    return acc;
  }, 0);
  // Add station management staff (6 per critical station, 4 per high)
  const stationStaff = (DATA.stations || []).reduce((acc, st) => {
    if (st.current_level === 'critical') return acc + 6;
    if (st.current_level === 'high')     return acc + 4;
    return acc;
  }, 0);

  animateCount('stat-critical', criticalCount);
  animateCount('stat-high', s.high_risk_trains);
  animateCount('stat-stations', s.stations_monitored);
  animateCount('stat-staff', staffCount + stationStaff);
  animateCount('stat-ticketless', ticketlessCount);

  document.getElementById('stat-critical-sub').textContent =
    `${Math.round(s.avg_risk_score * 100)}% avg risk score`;
  document.getElementById('stat-high-sub').textContent =
    `${s.medium_risk_trains} medium-risk`;
  document.getElementById('stat-stations-sub').textContent = 'Real-time monitoring';
  document.getElementById('stat-staff-sub').textContent =
    `${staffCount} on-train + ${stationStaff} station`;
  document.getElementById('stat-ticketless-sub').textContent = 'Coaches flagged';
  document.getElementById('footer-auc').textContent =
    `OC: ${s.model_accuracy?.overcrowding_auc || '--'} | TL: ${s.model_accuracy?.ticketless_auc || '--'}`;
  document.getElementById('last-sync').textContent =
    new Date(DATA.generated_at).toLocaleTimeString('en-IN', { hour12: false, hour: '2-digit', minute: '2-digit' });
}

function animateCount(id, target) {
  const el = document.getElementById(id);
  let cur = 0;
  const step = Math.ceil(target / 30);
  const t = setInterval(() => {
    cur = Math.min(cur + step, target);
    el.textContent = cur;
    if (cur >= target) clearInterval(t);
  }, 30);
}

// ── Filters / Selects ─────────────────────────────────────────────────────────
let filtersPopulated = false;

function populateFilters() {
  const stSel = document.getElementById('filter-station');

  if (filtersPopulated) return;
  filtersPopulated = true;

  DATA.stations.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.station_code;
    opt.textContent = `${s.station_code} — ${s.station_name}`;
    stSel.appendChild(opt);
  });
}

let trainSelectPopulated = false;

function populateTrainSelect() {
  const heatSel = document.getElementById('heatmap-train-select');
  if (trainSelectPopulated) return;
  trainSelectPopulated = true;

  DATA.trains.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.train_id;
    opt.textContent = `${t.train_id} — ${t.train_name}`;
    heatSel.appendChild(opt);
  });

  // Auto-select first train
  if (DATA.trains.length) {
    heatSel.value = DATA.trains[0].train_id;
  }
}

// ── Train Map ─────────────────────────────────────────────────────────────────
function initMap() {
  if (trainMap) return;
  trainMap = L.map('train-map', { zoomControl: false }).setView([22.5937, 78.9629], 4);
  L.control.zoom({ position: 'bottomright' }).addTo(trainMap);
  
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; CARTO'
  }).addTo(trainMap);
}

function renderMap() {
  if (!trainMap) initMap();
  
  if (!DATA || !DATA.trains) return;

  DATA.trains.forEach(t => {
    let coords = [t.lat, t.lon];
    
    // Fallback if backend simulator hasn't initialized coords
    if (!coords[0] || isNaN(coords[0])) {
       const parts = t.route.split('→');
       coords = STATION_COORDS[parts[0]] || STATION_COORDS[parts[1]] || [22.0, 78.0];
    }

    const pct = Math.round(t.aggregate_risk * 100);
    const color = pct >= 70 ? '#ef4444' : pct >= 45 ? '#f97316' : pct >= 25 ? '#eab308' : '#22c55e';
    
    if (mapMarkers[t.train_id]) {
      // Smoothly animate existing marker
      mapMarkers[t.train_id].setLatLng(coords);
      mapMarkers[t.train_id].setStyle({ fillColor: color });
      mapMarkers[t.train_id]._popup.setContent(`
        <div style="color: #333; font-family: Inter, sans-serif; font-size: 12px;">
          <strong style="font-size: 14px;">${t.train_name}</strong><br/>
          Route: ${t.route}<br/>
          Agg. Risk: <strong>${pct}%</strong>
        </div>
      `);
    } else {
      // Create new marker
      const circle = L.circleMarker(coords, {
        radius: 5,
        fillColor: color,
        color: '#fff',
        weight: 1,
        opacity: 0.9,
        fillOpacity: 0.9
      }).addTo(trainMap);
      
      circle.bindPopup(`
        <div style="color: #333; font-family: Inter, sans-serif; font-size: 12px;">
          <strong style="font-size: 14px;">${t.train_name}</strong><br/>
          Route: ${t.route}<br/>
          Agg. Risk: <strong>${pct}%</strong>
        </div>
      `);
      
      circle.on('click', () => {
        document.getElementById('heatmap-train-select').value = t.train_id;
        switchTab('train');
      });
      
      mapMarkers[t.train_id] = circle;
    }
  });
}

// ── Coach Heatmap ─────────────────────────────────────────────────────────────
function renderHeatmap(overrideTrainId) {
  const trainId = overrideTrainId || document.getElementById('heatmap-train-select').value;
  const train = DATA.trains.find(t => t.train_id === trainId);
  if (!train) return;

  // Train info bar
  document.getElementById('train-route').textContent = `🛤 ${train.route}`;
  document.getElementById('train-departs').textContent = `🕐 Departs ${train.departs}`;
  const aggBadge = document.getElementById('agg-risk-badge');
  const aggPct = Math.round(train.aggregate_risk * 100);
  const lvl = aggPct >= 70 ? 'critical' : aggPct >= 45 ? 'high' : aggPct >= 25 ? 'medium' : 'low';
  aggBadge.textContent = `Aggregate Risk: ${aggPct}%`;
  aggBadge.className = `aggregate-risk-badge lb-badge ${lvl}`;

  // Group coaches by type
  const groups = { SL: [], '3A': [], '2A': [], '1A': [] };
  train.coaches.forEach(c => { if (groups[c.coach_type]) groups[c.coach_type].push(c); });

  const container = document.getElementById('heatmap-container');
  container.innerHTML = '';

  Object.entries(groups).forEach(([type, coaches]) => {
    if (!coaches.length) return;
    const row = document.createElement('div');
    row.className = 'coach-row';

    const label = document.createElement('div');
    label.className = 'coach-row-label';
    label.textContent = type;
    row.appendChild(label);

    const strip = document.createElement('div');
    strip.className = 'coaches-strip';

    coaches.forEach(c => {
      const tile = document.createElement('div');
      tile.className = `coach-tile risk-${c.risk_level}`;
      tile.id = `tile-${c.coach_id}`;
      tile.innerHTML = `<span class="tile-id">${c.coach_id}</span><span class="tile-pct">${Math.round(c.composite_risk * 100)}%</span>`;
      tile.title = `${c.coach_id} — Overcrowding: ${Math.round(c.overcrowding_risk*100)}% | Ticketless: ${Math.round(c.ticketless_risk*100)}%`;
      tile.onclick = () => openCoachDetail(c, train);
      strip.appendChild(tile);
    });

    row.appendChild(strip);
    container.appendChild(row);
  });
}

// ── Coach Detail Modal ────────────────────────────────────────────────────────
function openCoachDetail(coach, train) {
  selectedCoach = coach;
  document.querySelectorAll('.coach-tile').forEach(t => t.classList.remove('selected'));
  const tile = document.getElementById(`tile-${coach.coach_id}`);
  if (tile) tile.classList.add('selected');

  document.getElementById('modal-title').textContent =
    `${coach.coach_id} — ${train.train_name} (${train.train_id})`;

  const lvl = coach.risk_level;
  const badge = document.getElementById('modal-risk-badge');
  badge.textContent = lvl.toUpperCase();
  badge.className = `modal-risk-badge lb-badge ${lvl}`;

  document.getElementById('m-occupancy').textContent =
    `${Math.round(coach.occupancy_ratio * 100)}%`;
  document.getElementById('m-overcrowd').textContent =
    `${Math.round(coach.overcrowding_risk * 100)}%`;
  document.getElementById('m-ticketless').textContent =
    `${Math.round(coach.ticketless_risk * 100)}%`;
  document.getElementById('m-seats').textContent =
    `${coach.booked_seats}/${coach.capacity}`;

  const ttes = coach.composite_risk > 0.7 ? 3 : coach.composite_risk > 0.45 ? 2 : 1;
  document.getElementById('modal-rec').innerHTML =
    `<strong>🤖 AI Directive:</strong> Deploy <strong>${ttes} TTE(s)</strong> to Coach ${coach.coach_id} of ${train.train_name} departing at <strong>${train.departs}</strong>. ` +
    `Monitor for ticketless passengers — risk at ${Math.round(coach.ticketless_risk*100)}%.`;

  document.getElementById('coach-overlay').classList.add('open');
  updateRiskBars(coach);
}

function closeCoachDetail(event, force) {
  if (!force && event && event.target !== document.getElementById('coach-overlay')) return;
  document.getElementById('coach-overlay').classList.remove('open');
  document.querySelectorAll('.coach-tile').forEach(t => t.classList.remove('selected'));
}

// ── Risk Factor Bars ──────────────────────────────────────────────────────────
function updateRiskBars(coach) {
  const rf = coach.risk_factors;
  const occ = rf.occupancy_pct;
  const hol = Math.min(100, rf.holiday_weight);
  const tl  = Math.min(100, (rf.historical_incidents / coach.capacity) * 100);
  const ev  = rf.event_pressure;
  const cn  = rf.cancellation_surge;

  setBar('rf-occupancy', 'rfv-occupancy', occ, `${occ.toFixed(0)}%`);
  setBar('rf-holiday',   'rfv-holiday',   hol, `${hol.toFixed(0)}%`);
  setBar('rf-ticketless','rfv-ticketless', tl,  rf.historical_incidents);
  setBar('rf-event',     'rfv-event',     ev,  `${ev.toFixed(0)}%`);
  setBar('rf-cancel',    'rfv-cancel',    cn,  `${cn.toFixed(1)}%`);

  document.getElementById('radar-subtitle').textContent =
    `Coach ${coach.coach_id} · Risk ${Math.round(coach.composite_risk * 100)}%`;

  updateRadar(coach);
}

function setBar(barId, valId, pct, label) {
  document.getElementById(barId).style.width = Math.min(100, pct) + '%';
  document.getElementById(valId).textContent = label;
}

// ── Radar Chart ───────────────────────────────────────────────────────────────
function updateRadar(coach) {
  const rf = coach.risk_factors;
  const data = [
    Math.min(100, rf.occupancy_pct),
    Math.min(100, rf.holiday_weight),
    Math.min(100, (rf.historical_incidents / coach.capacity) * 100),
    Math.min(100, rf.event_pressure),
    Math.min(100, rf.cancellation_surge),
  ];

  if (radarChart) {
    radarChart.data.datasets[0].data = data;
    radarChart.update('active');
    return;
  }

  const ctx = document.getElementById('radar-chart').getContext('2d');
  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Occupancy', 'Holiday', 'Ticketless Hist.', 'Event Pressure', 'Cancellation'],
      datasets: [{
        data,
        backgroundColor: 'rgba(99,102,241,0.18)',
        borderColor: '#818cf8',
        pointBackgroundColor: '#818cf8',
        pointRadius: 4,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { r: { min: 0, max: 100, ticks: { display: false }, grid: { color: 'rgba(255,255,255,0.07)' }, pointLabels: { color: '#94a3b8', font: { size: 10 } }, angleLines: { color: 'rgba(255,255,255,0.07)' } } },
      plugins: { legend: { display: false } },
    }
  });
}

// ── Train-Specific Stats (Inspector Tab) ──────────────────────────────────────
function renderSpecificTrainView() {
  if (!DATA) return;
  populateTrainSelect();

  const heatSel = document.getElementById('heatmap-train-select');
  const trainId = heatSel.value || (DATA.trains[0] && DATA.trains[0].train_id);
  if (!trainId) return;
  if (!heatSel.value && trainId) heatSel.value = trainId;

  const train = DATA.trains.find(t => t.train_id === trainId);
  if (!train) return;

  // Per-train summary cards
  const criticalCoaches = train.coaches.filter(c => c.risk_level === 'critical').length;
  const ticketlessCoaches = train.coaches.filter(c => c.ticketless_risk > 0.5).length;
  const staffNeeded = train.coaches.reduce((acc, c) => {
    return acc + (c.composite_risk > 0.7 ? 3 : c.composite_risk > 0.45 ? 2 : 1);
  }, 0);

  document.getElementById('train-stat-critical').textContent = criticalCoaches;
  document.getElementById('train-stat-staff').textContent = staffNeeded;
  document.getElementById('train-stat-ticketless').textContent = ticketlessCoaches;
  document.getElementById('train-stat-agg').textContent =
    `${Math.round(train.aggregate_risk * 100)}% aggregate risk`;

  // Per-train recommendations
  const recList = document.getElementById('rec-list');
  recList.innerHTML = '';
  const trainRecs = DATA.recommendations
    .filter(r => r.train_id === trainId || r.train_name === train.train_name)
    .slice(0, 15);

  if (trainRecs.length === 0) {
    recList.innerHTML = `<div class="loading-state"><p style="color: var(--text-muted);">No specific directives for this train. It may be running normally.</p></div>`;
  } else {
    trainRecs.forEach((r, i) => {
      const item = document.createElement('div');
      item.className = `rec-item ${r.priority}`;
      item.style.animationDelay = `${i * 50}ms`;
      item.innerHTML = `
        <div class="rec-action">
          ${r.priority === 'critical' ? '🔴' : r.priority === 'high' ? '🟠' : '🟡'}
          ${r.action}
        </div>
        <div class="rec-meta">
          <span class="rec-reason">${r.reason}</span>
          <span class="rec-confidence ${r.priority}">${r.confidence}% confidence</span>
        </div>
      `;
      recList.appendChild(item);
    });
  }

  // Draw Specific Train Route Map
  if (specificTrainMap && specificTrainLayer) {
    specificTrainLayer.clearLayers();
    const parts = train.route.split('→');
    const origCoords = STATION_COORDS[parts[0]] || [28.6139, 77.2090];
    const destCoords = STATION_COORDS[parts[1]] || [18.9690, 72.8205];
    
    // Draw route line
    const routeLine = L.polyline([origCoords, destCoords], {
      color: '#818cf8',
      weight: 3,
      opacity: 0.6,
      dashArray: '5, 10'
    }).addTo(specificTrainLayer);

    // Draw origin and destination markers
    L.circleMarker(origCoords, { radius: 6, fillColor: '#22c55e', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(specificTrainLayer)
     .bindTooltip('Origin: ' + parts[0], { permanent: false });
    L.circleMarker(destCoords, { radius: 6, fillColor: '#ef4444', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(specificTrainLayer)
     .bindTooltip('Destination: ' + parts[1], { permanent: false });

    // Draw current train position (interpolate based on a pseudo-progress)
    const progress = 0.3 + (Math.random() * 0.4); // between 30% and 70%
    const trainLat = origCoords[0] + (destCoords[0] - origCoords[0]) * progress;
    const trainLng = origCoords[1] + (destCoords[1] - origCoords[1]) * progress;
    
    const color = train.aggregate_risk >= 0.7 ? '#ef4444' : train.aggregate_risk >= 0.45 ? '#f97316' : '#eab308';
    L.circleMarker([trainLat, trainLng], {
      radius: 8,
      fillColor: color,
      color: '#fff',
      weight: 2,
      opacity: 1,
      fillOpacity: 1
    }).addTo(specificTrainLayer).bindTooltip('Live Location', { permanent: true, direction: 'top' }).openTooltip();

    // Fit map bounds to show the entire route with padding
    setTimeout(() => {
      specificTrainMap.fitBounds(routeLine.getBounds(), { padding: [30, 30] });
    }, 100);
  }

  // Render the coach heatmap for this train
  renderHeatmap(trainId);
}

// ── Train Leaderboard ─────────────────────────────────────────────────────────
function renderLeaderboard() {
  const sortBy = document.getElementById('sort-trains').value;
  const riskFilter = document.getElementById('filter-risk').value;
  let trains = [...DATA.trains];

  if (sortBy === 'departure') trains.sort((a, b) => a.departs.localeCompare(b.departs));
  else trains.sort((a, b) => b.aggregate_risk - a.aggregate_risk);

  const list = document.getElementById('leaderboard-list');
  list.innerHTML = '';

  trains.forEach((train, i) => {
    const pct = Math.round(train.aggregate_risk * 100);
    const lvl = pct >= 70 ? 'critical' : pct >= 45 ? 'high' : pct >= 25 ? 'medium' : 'low';
    if (riskFilter !== 'all' && lvl !== riskFilter) return;

    const barColor = lvl === 'critical' ? '#ef4444' : lvl === 'high' ? '#f97316' : lvl === 'medium' ? '#eab308' : '#22c55e';
    const delayMin = train.delay_minutes || 0;
    const delayBadge = delayMin === 0
      ? `<span style="font-size:0.68rem;color:#22c55e;font-weight:600;">🟢 On Time</span>`
      : `<span style="font-size:0.68rem;color:${delayMin > 60 ? '#ef4444' : '#f97316'};font-weight:600;">🔴 ${train.delay_status}</span>`;

    const icons = [
      train.coaches.some(c => c.overcrowding_risk > 0.6) ? '🚨 Overcrowding' : '',
      train.coaches.some(c => c.ticketless_risk > 0.5)   ? '⚠️ Ticketless' : '',
      train.coaches.some(c => c.coach_type === 'GEN')    ? '🚃 GEN coaches' : '',
    ].filter(Boolean).join(' · ');

    const item = document.createElement('div');
    item.className = 'lb-item';
    item.innerHTML = `
      <div class="lb-header">
        <div>
          <div class="lb-name">${train.train_name}</div>
          <div class="lb-id">${train.train_id} · ${train.route} · Dep ${train.departs}</div>
        </div>
        <div class="lb-meta" style="display:flex;flex-direction:column;align-items:flex-end;gap:4px;">
          <span class="lb-badge ${lvl}">${pct}% risk</span>
          ${delayBadge}
        </div>
      </div>
      ${icons ? `<div style="font-size:0.7rem;color:var(--text-muted);margin-bottom:6px">${icons}</div>` : ''}
      <div class="lb-bar-bg"><div class="lb-bar" style="width:0%;background:${barColor}" data-target="${pct}"></div></div>
    `;
    // Click → switch to Train Inspector tab with that train selected
    item.onclick = () => {
      document.getElementById('heatmap-train-select').value = train.train_id;
      switchTab('train');
    };
    list.appendChild(item);

    // Animate bar
    setTimeout(() => {
      const bar = item.querySelector('.lb-bar');
      if (bar) bar.style.width = pct + '%';
    }, 50 + i * 30);
  });
}

// ── Station-Filtered Leaderboard ──────────────────────────────────────────────
function renderLeaderboardForStation(stationCode) {
  // Filter trains whose route contains the selected station code
  const stationStation = DATA.stations.find(s => s.station_code === stationCode);
  const list = document.getElementById('leaderboard-list');
  list.innerHTML = '';

  // Trains that pass through this station (match by origin/destination code in route)
  let filteredTrains = DATA.trains.filter(t => t.route.includes(stationCode));

  // If no exact route match, just show all trains sorted by risk (fallback)
  if (filteredTrains.length === 0) filteredTrains = [...DATA.trains];

  filteredTrains.sort((a, b) => b.aggregate_risk - a.aggregate_risk);

  // Header showing what we're filtering by
  const header = document.createElement('div');
  header.style.cssText = 'padding: 8px 12px; font-size: 0.75rem; color: #818cf8; font-weight: 600; border-bottom: 1px solid rgba(99,102,241,0.2); margin-bottom: 6px;';
  header.textContent = `📍 Trains at ${stationCode}${stationStation ? ' — ' + stationStation.station_name : ''} (${filteredTrains.length} found)`;
  list.appendChild(header);

  filteredTrains.slice(0, 10).forEach((train, i) => {
    const pct = Math.round(train.aggregate_risk * 100);
    const lvl = pct >= 70 ? 'critical' : pct >= 45 ? 'high' : pct >= 25 ? 'medium' : 'low';
    const barColor = lvl === 'critical' ? '#ef4444' : lvl === 'high' ? '#f97316' : lvl === 'medium' ? '#eab308' : '#22c55e';

    const item = document.createElement('div');
    item.className = 'lb-item';
    item.innerHTML = `
      <div class="lb-header">
        <div>
          <div class="lb-name">${train.train_name}</div>
          <div class="lb-id">${train.train_id} · ${train.route} · Dep ${train.departs}</div>
        </div>
        <div class="lb-meta">
          <span class="lb-badge ${lvl}">${pct}% risk</span>
        </div>
      </div>
      <div class="lb-bar-bg"><div class="lb-bar" style="width:0%;background:${barColor}"></div></div>
    `;
    item.onclick = () => {
      document.getElementById('heatmap-train-select').value = train.train_id;
      switchTab('train');
    };
    list.appendChild(item);
    setTimeout(() => {
      const bar = item.querySelector('.lb-bar');
      if (bar) bar.style.width = pct + '%';
    }, 50 + i * 30);
  });
}

// ── Station Congestion Chart ──────────────────────────────────────────────────
function renderCongestionChart() {
  const selectedCode = document.getElementById('filter-station').value;
  const colors = ['#818cf8', '#c084fc', '#f97316', '#22c55e', '#eab308'];

  // If a specific station is selected, show only that one with full detail
  // Otherwise show top 5 as overview
  let stations;
  if (selectedCode && selectedCode !== 'all') {
    const found = DATA.stations.find(s => s.station_code === selectedCode);
    stations = found ? [found] : DATA.stations.slice(0, 5);
  } else {
    stations = DATA.stations.slice(0, 5);
  }

  const labels = stations[0]?.timeline.map(t => t.hour) || [];

  const datasets = stations.map((s, i) => ({
    label: `${s.station_code} — ${s.station_name}`,
    data: s.timeline.map(t => Math.round(t.congestion * 100)),
    borderColor: colors[i % colors.length],
    backgroundColor: colors[i % colors.length] + '22',
    borderWidth: selectedCode && selectedCode !== 'all' ? 3 : 2,
    pointRadius: selectedCode && selectedCode !== 'all' ? 5 : 3,
    pointHoverRadius: 8,
    tension: 0.4,
    fill: selectedCode && selectedCode !== 'all',
  }));

  const ctx = document.getElementById('congestion-chart').getContext('2d');
  if (congestionChart) congestionChart.destroy();

  // If single station selected, also highlight its peak info in the panel title
  const titleText = (selectedCode && selectedCode !== 'all')
    ? `Station: ${stations[0]?.station_code} — ${stations[0]?.station_name}`
    : 'Top 5 Stations (All)';

  congestionChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        title: {
          display: !!(selectedCode && selectedCode !== 'all'),
          text: titleText,
          color: '#818cf8',
          font: { size: 12, weight: '600' }
        },
        legend: { labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 12, padding: 14 } },
        tooltip: {
          backgroundColor: 'rgba(13,17,23,0.95)',
          borderColor: 'rgba(99,102,241,0.3)',
          borderWidth: 1,
          callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}% congestion` }
        }
      },
      scales: {
        x: { ticks: { color: '#475569', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: {
          min: 0, max: 100,
          ticks: { color: '#475569', font: { size: 10 }, callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.06)' }
        }
      },
      animation: { duration: 800, easing: 'easeInOutQuart' }
    }
  });

  // Also filter the leaderboard to only trains passing through selected station
  if (selectedCode && selectedCode !== 'all') {
    renderLeaderboardForStation(selectedCode);
  } else {
    renderLeaderboard();
  }
}

// ── Refresh ───────────────────────────────────────────────────────────────────
async function refreshDashboard() {
  const btn = document.getElementById('refresh-btn');
  btn.classList.add('spinning');
  btn.disabled = true;
  await pollLive();
  btn.classList.remove('spinning');
  btn.disabled = false;
}

function renderAll() {
  renderSummary();
  populateFilters();
  renderMap();
  renderLeaderboard();
  renderCongestionChart();
  // If inspector tab is active, refresh it too
  if (activeTab === 'train') renderSpecificTrainView();
}

// ── Keyboard shortcut ─────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeCoachDetail(null, true);
  if (e.key === 'r' && e.ctrlKey) { e.preventDefault(); refreshDashboard(); }
});

// Filter events
document.getElementById('filter-risk').addEventListener('change', renderLeaderboard);
document.getElementById('filter-station').addEventListener('change', () => renderCongestionChart());

// ── Login ─────────────────────────────────────────────────────────────────────
async function handleLogin() {
  const btn = document.querySelector('.login-btn');
  const id = document.getElementById('login-id').value;
  const pass = document.getElementById('login-pass').value;

  btn.textContent = 'Authenticating...';
  btn.style.opacity = '0.7';
  
  try {
    const res = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: id, password: pass })
    });
    
    if (res.ok) {
      const data = await res.json();
      document.getElementById('login-overlay').classList.add('hidden');
      document.getElementById('user-name-display').textContent = data.user;
      
      // Start live simulation polling
      await pollLive();
      setInterval(pollLive, 10000); // 10 second polling for smoother hackathon demo
    } else {
      btn.textContent = 'Invalid Password';
      btn.style.background = '#ef4444';
      setTimeout(() => {
        btn.textContent = 'Secure Login';
        btn.style.opacity = '1';
        btn.style.background = '';
      }, 2000);
    }
  } catch (e) {
    btn.textContent = 'API Offline - Using Fallback';
    setTimeout(() => {
      document.getElementById('login-overlay').classList.add('hidden');
      pollLive(); // Will fall back to offline data
    }, 1500);
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  startClock();
})();
