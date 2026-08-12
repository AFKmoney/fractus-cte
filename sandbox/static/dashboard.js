/* Fractus Sandbox — dashboard logic. Vanilla JS, no framework. */

const $ = (id) => document.getElementById(id);

// ── State ─────────────────────────────────────────────────────────
let ws = null;
let lossHistory = [];        // for training curve
let routingHistory = [];     // recent routing snapshots (not used for chart, kept for debug)
let connStatus = 'connecting';

// ── WebSocket ─────────────────────────────────────────────────────
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.onopen = () => { connStatus = 'live'; updateConn(); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'state') renderState(msg);
  };
  ws.onclose = () => { connStatus = 'down'; updateConn(); setTimeout(connect, 2000); };
  ws.onerror = () => { ws.close(); };
}

function updateConn() {
  const dot = $('conn-dot');
  const txt = $('conn-txt');
  dot.className = 'dot ' + (connStatus === 'live' ? 'live' : '');
  txt.textContent = connStatus === 'live' ? 'ENGINE LIVE' : 'RECONNECTING…';
}

// ── Render live state ─────────────────────────────────────────────
function renderState(s) {
  const e = s.engine;
  // Live monitor
  $('thought-norm').textContent = e.thought_norm.toFixed(3);
  $('thought-mean').textContent = e.thought_mean.toFixed(4);
  $('thought-std').textContent = e.thought_std.toFixed(4);
  $('confidence').textContent = (e.confidence * 100).toFixed(1) + '%';
  $('conf-gauge').style.width = (e.confidence * 100) + '%';
  $('tick-count').textContent = e.tick_count;
  $('lb-loss').textContent = e.lb_loss.toFixed(4);
  $('salience-loss').textContent = e.salience_loss.toFixed(4);
  $('n-blocks').textContent = e.n_blocks;
  $('n-experts').textContent = e.n_experts;
  $('mem-count').textContent = e.memory_count;

  // Oscilloscope
  drawScope(s.phases);

  // Routing
  drawRouting(s.routing);

  // Mode
  drawMode(s.mode);
}

// ── Oscilloscope: Kuramoto phases on unit circle per block ────────
function drawScope(phases) {
  const cv = $('scope');
  const ctx = cv.getContext('2d');
  const W = cv.width = cv.clientWidth;
  const H = cv.height = cv.clientHeight;
  ctx.clearRect(0, 0, W, H);
  if (!phases || !phases.length) return;

  const nBlocks = phases.length;
  const cellW = W / nBlocks;
  const r = Math.min(cellW, H) * 0.38;

  phases.forEach((blk, bi) => {
    const cx = bi * cellW + cellW / 2;
    const cy = H / 2;
    // circle
    ctx.strokeStyle = '#243040';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    // oscillators
    const colors = ['#00e5a8', '#4fc3f7', '#ffb74d', '#ff5252', '#b388ff', '#69f0ae', '#ff80ab', '#18ffff'];
    blk.phases.forEach((phi, oi) => {
      const x = cx + r * Math.cos(phi);
      const y = cy + r * Math.sin(phi);
      ctx.fillStyle = colors[oi % colors.length];
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
      // trail line
      ctx.strokeStyle = colors[oi % colors.length] + '55';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();
    });
    // label
    ctx.fillStyle = '#7a8699';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`blk ${bi}`, cx, H - 4);
  });
}

// ── Routing: bar chart of expert hits ─────────────────────────────
function drawRouting(routing) {
  const cv = $('routing');
  const ctx = cv.getContext('2d');
  const W = cv.width = cv.clientWidth;
  const H = cv.height = cv.clientHeight;
  ctx.clearRect(0, 0, W, H);
  if (!routing || !routing.expert_hits) return;

  const hits = routing.expert_hits;
  const n = hits.length;
  const max = Math.max(...hits, 1);
  const barW = (W - 20) / n;
  const baseY = H - 16;

  hits.forEach((h, i) => {
    const bh = (h / max) * (H - 30);
    const x = 10 + i * barW;
    const isDom = (i === routing.dominant);
    ctx.fillStyle = isDom ? '#ff5252' : '#00e5a8';
    ctx.fillRect(x + 1, baseY - bh, barW - 2, bh);
    // label
    ctx.fillStyle = '#7a8699';
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`e${i}`, x + barW / 2, baseY + 10);
  });
  // dominance line
  $('routing-info').textContent =
    `dom: e${routing.dominant} · ${routing.dominance ? (routing.dominance*100).toFixed(0) : 0}% · Σ${routing.total}`;
}

// ── Mode: current cognitive mode + probabilities ──────────────────
function drawMode(mode) {
  $('mode-name').textContent = mode.mode;
  $('mode-conf').textContent = (mode.confidence * 100).toFixed(0) + '%';
  const wrap = $('mode-bars');
  wrap.innerHTML = '';
  const entries = Object.entries(mode.all_modes || {});
  entries.sort((a, b) => b[1] - a[1]);
  for (const [name, p] of entries) {
    const row = document.createElement('div');
    row.className = 'mode-bar';
    row.innerHTML = `
      <span class="name">${name}</span>
      <span class="track"><span class="bar" style="width:${(p*100).toFixed(1)}%"></span></span>
      <span class="pct">${(p*100).toFixed(0)}%</span>`;
    wrap.appendChild(row);
  }
}

// ── Training curve ────────────────────────────────────────────────
function drawCurve() {
  const cv = $('curve');
  const ctx = cv.getContext('2d');
  const W = cv.width = cv.clientWidth;
  const H = cv.height = cv.clientHeight;
  ctx.clearRect(0, 0, W, H);
  if (lossHistory.length < 2) {
    ctx.fillStyle = '#7a8699';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('no training data yet — POST /api/train', W / 2, H / 2);
    return;
  }
  const data = lossHistory.slice(-100);
  const max = Math.max(...data, 0.1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  // grid
  ctx.strokeStyle = '#1a212c';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * (H - 20) + 5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  // line
  ctx.strokeStyle = '#ff5252';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - 15 - ((v - min) / range) * (H - 25);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // ppl label
  const avg = data.reduce((a, b) => a + b, 0) / data.length;
  $('curve-info').textContent = `loss ${data[data.length-1].toFixed(3)} · avg ${avg.toFixed(3)} · ppl ${Math.exp(Math.min(avg, 20)).toFixed(1)}`;
}

// ── API calls ─────────────────────────────────────────────────────
async function api(path, body) {
  const opt = body ? { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  return r.json();
}

function addChat(role, text) {
  const log = $('chat-log');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = `<span class="role">${role}:</span> <span class="text"></span>`;
  div.querySelector('.text').textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function doChat() {
  const inp = $('chat-input');
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  addChat('user', msg);
  addChat('system', 'thinking…');
  try {
    const r = await api('/api/chat', { message: msg });
    $('chat-log').lastChild.remove(); // remove "thinking…"
    addChat('fractus', r.reply);
    renderState({ engine: r.state, phases: null, routing: r.routing, mode: r.mode });
    if (r.routing) drawRouting(r.routing);
    if (r.mode) drawMode(r.mode);
  } catch (err) {
    $('chat-log').lastChild.querySelector('.text').textContent = 'ERROR: ' + err.message;
  }
}

async function doTick() { await api('/api/tick'); }
async function doReset() { await api('/api/reset'); addChat('system', 'thought state reset'); }
async function doGrow() {
  const r = await api('/api/grow');
  addChat('system', r.grew ? `grew → ${r.n_experts} experts` : `no growth (needs imbalance)`);
  if (r.routing) drawRouting(r.routing);
}
async function doInject() {
  const inp = $('inject-input');
  if (!inp.value.trim()) return;
  await api('/api/inject', { text: inp.value });
  inp.value = '';
  addChat('system', 'injected');
}

async function doTrain() {
  const inp = $('train-input');
  if (!inp.value.trim()) return;
  addChat('system', 'training…');
  const r = await api('/api/train', { text: inp.value, steps: 12 });
  if (r.training) {
    lossHistory = r.training.losses || lossHistory;
    drawCurve();
    addChat('system', `trained ${r.steps} steps · final loss ${r.final_loss.toFixed(3)}`);
  } else {
    addChat('system', 'train error: ' + (r.error || 'unknown'));
  }
  inp.value = '';
}

async function delMem(idx) {
  await api(`/api/memory/delete/${idx}`, {});
  loadMemory();
}
async function addMem() {
  const inp = $('mem-add-input');
  if (!inp.value.trim()) return;
  await api('/api/memory/add', { text: inp.value });
  inp.value = '';
  loadMemory();
}

async function loadMemory() {
  const r = await api('/api/memory');
  const list = $('mem-list');
  list.innerHTML = '';
  if (!r.memories || !r.memories.length) {
    list.innerHTML = '<div class="empty">no memories yet — chat or inject to fill</div>';
    return;
  }
  for (const m of r.memories) {
    const div = document.createElement('div');
    div.className = 'mem-item';
    div.innerHTML = `<span class="ctx"></span><span class="imp">${m.importance.toFixed(2)}</span>
      <span class="imp">|v|${m.vector_norm.toFixed(1)}</span>
      <button onclick="delMem(${m.idx})">DEL</button>`;
    div.querySelector('.ctx').textContent = m.context;
    list.appendChild(div);
  }
}

// ── Config controls ───────────────────────────────────────────────
function setToggle(id, on) {
  $(id).classList.toggle('on', on);
}
async function toggleMemory() {
  const on = !$('tog-memory').classList.contains('on');
  setToggle('tog-memory', on);
  await api('/api/config', { memory_active: on });
}
async function salienceChange(v) {
  $('salience-v').textContent = v;
  await api('/api/config', { salience_bias: parseFloat(v) });
}

// ── Init ──────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  connect();
  loadMemory();
  drawCurve();
  // initial params fetch
  api('/api/params').then(p => {
    $('p-dmodel').textContent = p.d_model;
    $('p-layers').textContent = p.n_layers;
    $('p-experts').textContent = p.n_experts;
    $('p-params').textContent = (p.params / 1e6).toFixed(2) + 'M';
    $('p-topk').textContent = p.top_k;
    $('p-rank').textContent = p.expert_rank;
    $('p-osc').textContent = p.n_oscillators;
  });
  // poll memory + training every 3s
  setInterval(loadMemory, 3000);
  setInterval(() => {
    api('/api/training').then(t => {
      if (t.losses && t.losses.length) { lossHistory = t.losses; drawCurve(); }
    });
  }, 3000);
  // Enter keys
  $('chat-input').addEventListener('keydown', e => { if (e.key === 'Enter') doChat(); });
  $('inject-input').addEventListener('keydown', e => { if (e.key === 'Enter') doInject(); });
  $('train-input').addEventListener('keydown', e => { if (e.key === 'Enter') doTrain(); });
  $('mem-add-input').addEventListener('keydown', e => { if (e.key === 'Enter') addMem(); });
});
