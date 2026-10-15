/* TrainWise AI website: thin client over the REST API (no framework, no build step). */
'use strict';

const KEY_PASSENGER = 'demo-passenger-key';
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x) => `${Math.round(100 * (Number(x) || 0))}%`;
const rupee = (n) => `₹${Number(n || 0).toLocaleString('en-IN')}`;
const when = (iso) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString('en-IN', { weekday: 'short', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false });
};
const dateInDays = (n) => {
  const d = new Date(); d.setDate(d.getDate() + Number(n || 0));
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};
const state = { trains: [], cities: [], stations: {}, plan: null, chatTurns: 0, scenarios: [] };
const opsKey = () => ($('#opsKey').value.trim() || 'demo-ops-key');

async function api(path, { method = 'GET', body, key = KEY_PASSENGER } = {}) {
  const res = await fetch(path, { method, headers: { 'Content-Type': 'application/json', 'X-API-Key': key }, body: body ? JSON.stringify(body) : undefined });
  const text = await res.text();
  let data; try { data = JSON.parse(text); } catch { data = text; }
  if (!res.ok) {
    const detail = data && data.detail !== undefined ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : `${res.status} ${res.statusText}`;
    throw new Error(detail);
  }
  return data;
}
const busy = (el, msg = 'Working…') => { el.innerHTML = `<p class="loading">${esc(msg)}</p>`; };
const fail = (el, err) => { el.innerHTML = `<p class="error">${esc(err && err.message ? err.message : err)}</p>`; };
const formData = (form) => Object.fromEntries(new FormData(form).entries());
const metrics = (items) => `<div class="metrics">${items.map(([k, v]) => `<div><small>${esc(k)}</small><b>${esc(v)}</b></div>`).join('')}</div>`;
const table = (headers, rows, cls = '') => `<div class="table-wrap ${cls}"><table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;

/* ---------------------------------------------------------------- navigation */
$('#menuBtn').addEventListener('click', () => {
  const nav = $('#nav'); const open = nav.classList.toggle('open');
  $('#menuBtn').setAttribute('aria-expanded', String(open));
});
$$('#nav a').forEach((a) => a.addEventListener('click', () => $('#nav').classList.remove('open')));

/* ---------------------------------------------------------------- lookups */
const trainByNumber = (n) => state.trains.find((t) => t.number === n);

function bindClassSelects() {
  $$('select[data-classes-for]').forEach((clsSel) => {
    const form = clsSel.closest('form');
    const trainSel = $(`select[name="${clsSel.dataset.classesFor}"]`, form);
    const fill = () => {
      const t = trainByNumber(trainSel.value);
      clsSel.innerHTML = (t ? t.classes : []).map((c) => `<option>${esc(c)}</option>`).join('');
    };
    trainSel.addEventListener('change', fill);
  });
}

async function loadTrains() {
  const data = await api('/v1/trains');
  Object.assign(state, { trains: data.trains, cities: data.cities, stations: data.stations });
  $$('select[data-trains]').forEach((sel) => {
    const list = sel.dataset.trains === 'dynamic'
      ? [...state.trains].sort((a, b) => Number(b.dynamic_pricing) - Number(a.dynamic_pricing) || a.number.localeCompare(b.number))
      : state.trains;
    sel.innerHTML = list.map((t) => `<option value="${esc(t.number)}">${esc(t.number)} ${esc(t.name)}${t.dynamic_pricing ? ' (flexi fare)' : ''}</option>`).join('');
    if (sel.dataset.default) sel.value = sel.dataset.default;
    sel.dispatchEvent(new Event('change'));
  });
  $$('select[data-cities]').forEach((sel) => {
    sel.innerHTML = state.cities.map((c) => `<option>${esc(c)}</option>`).join('');
    if (sel.dataset.default) sel.value = sel.dataset.default;
  });
}

/* ---------------------------------------------------------------- planner */
function ticket(o, plan) {
  const dur = `${Math.floor(o.duration_min / 60)}h ${String(o.duration_min % 60).padStart(2, '0')}m`;
  const deadline = plan && plan.deadline && o.prob_meet_deadline !== undefined
    ? `<div><small>Makes the deadline</small><b>${pct(o.prob_meet_deadline)}</b></div>` : '';
  return `<div class="ticket">
    <div class="ticket-main">
      <div class="ticket-route">
        <div><small>${esc(o.origin)}</small><b>${esc(when(o.departure))}</b></div>
        <div class="ticket-arrow">→</div>
        <div><small>${esc(o.destination)}</small><b>${esc(when(o.arrival))}</b></div>
      </div>
      <h3>${esc(o.train_number)} ${esc(o.train_name)}</h3>
      <p class="muted">${esc(o.class_label)} (${esc(o.class)}), ${dur}, expected delay ${Math.round(o.expected_delay_min)} min</p>
    </div>
    <div class="ticket-stub">
      <div><small>Fare per passenger</small><b>${rupee(o.fare)}</b></div>
      <div><small>Seat status</small><b>${esc(o.availability)}</b></div>
      <div><small>Chance of travelling</small><b>${pct(o.travel_probability)}</b></div>
      ${deadline}
    </div>
  </div>`;
}

function renderPlan(plan) {
  const out = $('#planResult');
  if (plan.status !== 'plan_ready') {
    out.innerHTML = `<div class="notice warn">${esc(plan.question || plan.message || plan.status)}</div>`;
    return;
  }
  const b = plan.recommendation;
  const others = [...plan.alternatives.map((a) => ({ ...a, note: '' })), ...plan.also_consider.map((a) => ({ ...a.option, note: a.trade_off }))];
  out.innerHTML = `
    ${ticket(b, plan)}
    <p class="summary">${esc(plan.summary)}</p>
    <ul class="why">${b.rationale.map((r) => `<li>${esc(r)}</li>`).join('')}</ul>
    ${plan.risks.map((r) => `<div class="notice warn">${esc(r)}</div>`).join('')}
    ${plan.assumptions.map((a) => `<div class="notice">${esc(a)}</div>`).join('')}
    ${others.length ? `<h4>Other options</h4>${table(['Train', 'Class', 'Fare', 'Seat', 'Chance', 'Trade-off'],
      others.map((a) => [`${esc(a.train_number)} ${esc(a.train_name)}`, esc(a.class), rupee(a.fare), esc(a.availability), pct(a.travel_probability), esc(a.note)]))}` : ''}
    <details><summary>Options ruled out (${plan.excluded.length})</summary>
      <ul class="small">${plan.excluded.map((e) => `<li><b>${esc(e.option)}</b>: ${esc(e.reasons.join('; '))}</li>`).join('')}</ul></details>
    <details><summary>How the agent worked</summary>
      <ol class="small">${plan.trace.map((s) => `<li><b>${esc(s.step)}</b> <span class="mono">${esc(s.t_ms)} ms</span></li>`).join('')}</ol></details>
    <div class="actions">
      <button class="btn accent" id="btnBook" type="button">Approve and book</button>
      <button class="btn" id="btnNoApprove" type="button" title="Watchtower reports this attempt in the next governance cycle">Test: book without approval</button>
    </div>
    <div id="bookResult"></div>`;
  $('#btnBook').onclick = () => book(plan, plan.approval.token);
  $('#btnNoApprove').onclick = () => book(plan, null);
}

async function book(plan, token) {
  const el = $('#bookResult'); busy(el, 'Booking…');
  try {
    const r = await api('/v1/planner/book', { method: 'POST', body: { plan_id: plan.plan_id, option_id: plan.recommendation.option_id, approval_token: token } });
    el.innerHTML = r.status === 'booked'
      ? `<div class="notice ok">Booked. PNR <span class="mono">${esc(r.pnr)}</span>, total ${rupee(r.total_fare)}. ${esc(r.note)}</div>`
      : `<div class="notice bad">Booking ${esc(r.status)}: ${esc(r.reason || '')}</div>`;
  } catch (e) { fail(el, e); }
}

$('#planForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const out = $('#planResult'); busy(out, 'The agent is searching trains and checking seats, delays and fares…');
  try {
    const plan = await api('/v1/planner/plan', { method: 'POST', body: { query: $('#planQuery').value, user_id: $('#planUser').value || null } });
    state.plan = plan; renderPlan(plan);
  } catch (e) { fail(out, e); }
});

/* ---------------------------------------------------------------- confirmation */
const bars = (probs) => `<div class="bars">${Object.entries(probs).map(([k, v]) => `<div class="bar"><span>${esc(k)}</span><i style="width:${Math.round(100 * v)}%"></i><b>${pct(v)}</b></div>`).join('')}</div>`;
const explainTable = (rows, unit) => rows && rows.length
  ? `<h4>What drives this estimate</h4>${table(['Factor', 'Your value', 'Typical', 'Effect'], rows.map((e) => [esc(e.feature), esc(e.value), esc(e.typical), unit === 'min' ? `${e.effect > 0 ? '+' : ''}${Number(e.effect).toFixed(1)} min` : `${e.effect > 0 ? '+' : ''}${(100 * e.effect).toFixed(1)} pts`]))}` : '';

$('#confirmForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const f = formData(ev.target); const out = $('#confirmResult'); busy(out);
  try {
    const r = await api('/v1/confirmation/predict', { method: 'POST', body: { train_number: f.train, travel_class: f.cls, journey_date: dateInDays(f.days), status: `${f.kind} ${f.position}`, quota: f.quota } });
    if (r.status !== 'ok') { out.innerHTML = `<div class="notice bad">No prediction: ${esc((r.issues || []).join('; ') || r.status)}</div>`; return; }
    out.innerHTML = metrics([['Confirmed berth', pct(r.confirmation_probability)], ['At least RAC', pct(r.travel_probability)], ['Likely at charting', r.predicted_final_status], ['Model confidence', pct(r.confidence)]])
      + bars(r.final_status_probabilities) + explainTable(r.explanation, 'pts')
      + `<p class="fineprint">Model ${esc(r.model)} ${esc(r.version)}; request ${esc(r.request_id)} is logged for monitoring.</p>`;
  } catch (e) { fail(out, e); }
});

/* ---------------------------------------------------------------- recommendations */
$('#trainsForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const form = ev.target; const f = formData(form); const out = $('#trainsResult'); busy(out);
  const prefs = $$('input[name="prefs"]:checked', form).map((i) => i.value);
  try {
    const r = await api('/v1/recommendations', { method: 'POST', body: { origin: f.origin, destination: f.destination, journey_date: dateInDays(f.days), preferences: prefs, user_id: f.user || null, budget: f.budget ? Number(f.budget) : null } });
    if (!r.recommendations.length) { out.innerHTML = '<div class="notice warn">No trains match. Try another date, route or budget.</div>'; return; }
    out.innerHTML = table(['#', 'Train', 'Class', 'Fare', 'Seat', 'Chance', 'Delay', 'Why'], r.recommendations.map((x) => [x.rank, `${esc(x.train_number)} ${esc(x.train_name)}`, esc(x.class), rupee(x.fare), esc(x.availability), pct(x.travel_probability), `${Math.round(x.expected_delay_min)} min`, esc(x.rationale.join('; '))]))
      + `<p class="fineprint">Ranking policy ${esc(r.ranker_version)}, ${r.candidates} options considered${r.personalised ? ', personalised to the passenger history' : ''}.</p>`;
  } catch (e) { fail(out, e); }
});

/* ---------------------------------------------------------------- delays */
$('#delayForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const f = formData(ev.target); const out = $('#delayResult'); busy(out);
  try {
    const r = await api('/v1/delay/predict', { method: 'POST', body: { train_number: f.train, journey_date: dateInDays(f.days) } });
    const iv = r.delay_interval_min || [0, 0];
    out.innerHTML = metrics([['Expected delay', `${Math.round(r.expected_delay_min)} min`], ['80% range', `${Math.round(iv[0])} to ${Math.round(iv[1])} min`], ['On time (15 min)', pct(r.prob_on_time_15min)], ['Risk', r.risk_level]])
      + (r.alerts || []).map((a) => `<div class="notice warn">${esc(a)}</div>`).join('')
      + table(['Station', 'Around', 'Congestion', 'Index'], (r.congestion || []).map((c) => [esc(c.station_name), esc(c.time.slice(11)), esc(c.level), c.congestion_index]))
      + ((r.alternatives || []).length ? `<h4>Better-running alternatives</h4>${table(['Train', 'Departs', 'Expected delay', 'On time'], r.alternatives.map((a) => [esc(a.train), esc(a.departure), `${Math.round(a.expected_delay_min)} min`, pct(a.prob_on_time_15min)]))}` : '')
      + explainTable(r.explanation, 'min');
  } catch (e) { fail(out, e); }
});

/* ---------------------------------------------------------------- fares */
function lineChart(points) {
  const w = 600, h = 200, p = 36;
  const fares = points.map((q) => q.fare);
  const min = Math.min(...fares), max = Math.max(...fares);
  const span = Math.max(max - min, 1);
  const x = (i) => p + (w - 2 * p) * (points.length === 1 ? 0 : i / (points.length - 1));
  const y = (f) => h - p - (h - 2 * p) * ((f - min) / span);
  const d = points.map((q, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(q.fare).toFixed(1)}`).join(' ');
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Forecast fare by booking date">
    <line class="grid" x1="${p}" y1="${h - p}" x2="${w - p}" y2="${h - p}"/><line class="grid" x1="${p}" y1="${p}" x2="${w - p}" y2="${p}"/>
    <path d="${d}" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linejoin="round"/>
    <text class="axis" x="${p}" y="${h - p + 16}">${esc(points[0].date)}</text>
    <text class="axis" x="${w - p}" y="${h - p + 16}" text-anchor="end">${esc(points[points.length - 1].date)} (departure)</text>
    <text class="axis" x="${p}" y="${p - 8}">${esc(rupee(max))}</text>
    <text class="axis" x="${w - p}" y="${p - 8}" text-anchor="end">lowest ${esc(rupee(min))}</text>
  </svg>`;
}

$('#fareForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const f = formData(ev.target); const out = $('#fareResult'); busy(out);
  try {
    const r = await api('/v1/fare/forecast', { method: 'POST', body: { train_number: f.train, travel_class: f.cls, journey_date: dateInDays(f.days) } });
    if (r.status !== 'ok') { out.innerHTML = `<div class="notice warn">${esc(r.message || r.status)}</div>`; return; }
    const s = r.demand_surge; const comp = r.explanation.fare_components;
    out.innerHTML = metrics([['Fare today', rupee(r.current.fare)], ['Flexi multiplier', `x${r.current.flexi_multiplier}`], ['Demand index', s.demand_index], ['Surge expected', s.surge ? 'Yes' : 'No']])
      + `<div class="notice ok">${esc(r.advice)}</div>`
      + (s.surge && s.drivers.length ? `<div class="notice warn">Demand drivers: ${esc(s.drivers.join(', '))}</div>` : '')
      + lineChart(r.trajectory)
      + (r.savings.length ? `<h4>Ways to save</h4>${table(['Option', 'Fare', 'Saving'], r.savings.map((x) => [esc(x.option), rupee(x.fare), rupee(x.saving)]))}` : '')
      + `<h4>Fare breakdown</h4>${table(Object.keys(comp).map((k) => k.replace(/_/g, ' ')), [Object.values(comp).map((v) => rupee(v))])}`;
  } catch (e) { fail(out, e); }
});

/* ---------------------------------------------------------------- support chat */
function bubble(role, text, meta) {
  const log = $('#chatLog');
  const div = document.createElement('div');
  div.className = `bubble ${role}`;
  div.innerHTML = esc(text) + (meta ? `<span class="meta">${esc(meta)}</span>` : '');
  log.appendChild(div); log.scrollTop = log.scrollHeight;
}
async function sendChat(message) {
  if (!message.trim()) return;
  bubble('user', message);
  const thinking = document.createElement('div'); thinking.className = 'bubble bot'; thinking.innerHTML = '<span class="loading">Thinking…</span>';
  $('#chatLog').appendChild(thinking);
  try {
    const r = await api('/v1/support/chat', { method: 'POST', body: { message, session_id: 'web' } });
    thinking.remove();
    bubble('bot', r.reply, `Detected ${r.language}, intent ${r.intent} (${pct(r.confidence)}), groundedness ${Number(r.guard.groundedness).toFixed(2)}${r.escalated ? ', handed to a human agent' : ''}`);
    state.chatTurns += 1;
    $('#rating').innerHTML = `How helpful was that? ${[1, 2, 3, 4, 5].map((n) => `<button type="button" data-rate="${n}">${n}</button>`).join('')}`;
  } catch (e) { thinking.remove(); bubble('bot', `Sorry, something went wrong: ${e.message}`); }
}
$('#chatForm').addEventListener('submit', (ev) => { ev.preventDefault(); const i = $('#chatInput'); sendChat(i.value); i.value = ''; });
$$('#quickPrompts button').forEach((b) => b.addEventListener('click', () => sendChat(b.textContent)));
$('#rating').addEventListener('click', async (ev) => {
  const btn = ev.target.closest('button[data-rate]'); if (!btn) return;
  try { await api('/v1/support/feedback', { method: 'POST', body: { session_id: 'web', rating: Number(btn.dataset.rate) } }); $('#rating').textContent = 'Thanks, your rating was recorded.'; } catch (e) { $('#rating').textContent = e.message; }
});

/* ---------------------------------------------------------------- fraud */
const FRAUD_FIELDS = ['requests_per_min', 'avg_inter_click_ms', 'form_fill_seconds', 'captcha_solve_seconds', 'accounts_on_device', 'ips_last_24h', 'bookings_last_24h', 'tatkal_window_hit', 'payment_failures_24h', 'passengers_per_booking', 'account_age_days', 'headless_signal'];
const PRESETS = {
  'Typical passenger': [6, 2200, 45, 7, 1, 1, 0, 0, 0, 2, 700, 0, 'mobile', 'metro'],
  'Bulk booking bot': [120, 120, 3, 1.5, 2, 3, 12, 1, 0, 6, 20, 1, 'desktop', 'metro'],
  'Multi-account tout': [18, 700, 15, 4, 9, 2, 8, 1, 0, 6, 120, 0, 'desktop', 'tier2'],
  'Low-and-slow bot ring': [7, 2100, 41, 7, 1, 1, 2, 1, 0, 5, 380, 0, 'mobile', 'tier2'],
};
function applyPreset(name) {
  const vals = PRESETS[name]; const form = $('#fraudForm');
  $('#fraudFields').innerHTML = FRAUD_FIELDS.map((k, i) => `<label>${esc(k.replace(/_/g, ' '))}<input type="number" step="any" name="${k}" value="${vals[i]}"></label>`).join('');
  $('select[name="device_type"]', form).value = vals[12]; $('select[name="region_segment"]', form).value = vals[13];
}
$('#fraudPreset').innerHTML = Object.keys(PRESETS).map((k) => `<option>${esc(k)}</option>`).join('');
$('#fraudPreset').addEventListener('change', (ev) => applyPreset(ev.target.value));
applyPreset('Typical passenger');
$('#fraudForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const f = formData(ev.target); const out = $('#fraudResult'); busy(out);
  const body = {}; FRAUD_FIELDS.forEach((k) => { body[k] = Number(f[k]); }); body.device_type = f.device_type; body.region_segment = f.region_segment;
  try {
    const r = await api('/v1/fraud/score', { method: 'POST', body, key: opsKey() });
    out.innerHTML = metrics([['Risk score', r.risk_score], ['Decision', r.decision], ['Suspected pattern', r.suspected_pattern || 'none']])
      + `<p>${esc(r.action)}</p>` + (r.reasons.length ? `<ul class="why">${r.reasons.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : '')
      + `<p class="fineprint">Model ${esc(r.model)} ${esc(r.version)}. The low-and-slow ring is a blind spot of v1: inject the novel bot attack in the Watchtower console, approve the retrained model, then score it again.</p>`;
  } catch (e) { fail(out, e); }
});

/* ---------------------------------------------------------------- Watchtower console */
const LAMPS = { green: ['Clear', 1, '#1a9e5c'], double_yellow: ['Attention', 2, '#e3a008'], yellow: ['Caution', 1, '#e3a008'], red: ['Danger', 1, '#c8102e'] };
const lamp = (aspect) => { const [label, n, c] = LAMPS[aspect] || LAMPS.green; return `<span class="aspect" style="color:${c}">${`<i class="lamp" style="background:${c}"></i>`.repeat(n)}<span style="color:inherit">${label}</span></span>`; };
const setStatus = (msg) => { $('#govStatus').textContent = msg; };

async function loadBoard() {
  try {
    const rows = await api('/v1/governance/health', { key: opsKey() });
    $('#board tbody').innerHTML = rows.map((r) => `<tr><td>${esc(r.service)}</td><td>${lamp(r.aspect)}</td><td class="mono">${esc(r.version)}</td><td>${r.requests}</td><td>${r.p95_ms ?? '-'}</td><td>${esc(r.metric)}</td><td>${esc(r.drift)}</td></tr>`).join('');
  } catch (e) { $('#board tbody').innerHTML = `<tr><td colspan="7" class="error">${esc(e.message)}</td></tr>`; }
}
async function loadScenarios() {
  try {
    state.scenarios = await api('/v1/governance/scenarios', { key: opsKey() });
    $('#scenarios').innerHTML = state.scenarios.map((s, i) => `<label title="${esc(s.description)}"><input type="checkbox" value="${esc(s.id)}" ${i < 4 ? 'checked' : ''}>${esc(s.label)}</label>`).join('');
  } catch (e) { $('#scenarios').innerHTML = `<span class="error">${esc(e.message)}</span>`; }
}
async function withButton(btn, msg, fn) {
  btn.disabled = true; setStatus(msg);
  try { await fn(); } catch (e) { setStatus(`Error: ${e.message}`); return; } finally { btn.disabled = false; }
}
$('#btnNormal').addEventListener('click', () => withButton($('#btnNormal'), 'Sending normal traffic (about 15 seconds)…', async () => {
  const r = await api('/v1/governance/simulate', { method: 'POST', body: { scenario: 'normal' }, key: opsKey() });
  setStatus(`Sent ${r.model_requests} model requests, ${r.chats} chats, ${r.searches} searches and ${r.plans} plans.`);
  await loadBoard();
}));
$('#btnInject').addEventListener('click', () => withButton($('#btnInject'), 'Injecting failures…', async () => {
  const chosen = $$('#scenarios input:checked').map((i) => i.value);
  for (const s of chosen) { setStatus(`Injecting ${s}…`); await api('/v1/governance/simulate', { method: 'POST', body: { scenario: s }, key: opsKey() }); }
  setStatus(`Injected ${chosen.length} failure(s). Run a governance cycle to see what Watchtower does.`);
  await loadBoard();
}));
$('#btnCycle').addEventListener('click', () => withButton($('#btnCycle'), 'Governance agents at work (can take 30 seconds when models retrain)…', async () => {
  const c = await api('/v1/governance/cycle', { method: 'POST', key: opsKey() });
  setStatus(`Cycle ${c.cycle_id} finished in ${c.duration_s} s.`);
  renderCycle(c);
  await Promise.all([loadBoard(), loadApprovals(), loadRegistry()]);
}));

function renderCycle(c) {
  const out = $('#cycleOut');
  if (!c.incidents.length) { out.innerHTML = `<div class="notice ok">${esc(c.summary)}</div>`; return; }
  out.innerHTML = `<p>${esc(c.summary.split('. ')[0])}.</p>` + c.incidents.map((inc) => `
    <details class="incident" ${inc.priority === 'P1' ? 'open' : ''}>
      <summary><span class="prio ${esc(inc.priority)}">${esc(inc.priority)}</span><b>${esc(inc.target)}</b><span class="tag ${inc.status === 'resolved' ? 'ok' : ''}">${esc(inc.status)}</span></summary>
      <p><b>Root cause</b> (${pct(inc.rca.confidence)} confidence): ${esc(inc.rca.root_cause)}</p>
      <ul>${inc.findings.map((f) => `<li><b>${esc(f.agent)}</b>: ${esc(f.summary)}</li>`).join('')}</ul>
      <ul>${inc.actions.map((a) => `<li><b>${esc(a.action)}</b>: ${esc(a.status)}${a.verification ? ` <span class="tag ${a.verified ? 'ok' : ''}">verified: ${esc(a.verification)}</span>` : ''}</li>`).join('')}</ul>
      ${inc.rca.blast_radius.length ? `<p class="small muted">Downstream services exposed: ${esc(inc.rca.blast_radius.join(', '))}</p>` : ''}
      <p class="small muted">Hypotheses considered: ${esc(inc.rca.hypotheses.map((h) => `${h.cause} (${Math.round(100 * h.confidence)}%)`).join(' | '))}</p>
    </details>`).join('');
}

async function loadApprovals() {
  const box = $('#approvals');
  try {
    const all = await api('/v1/governance/approvals', { key: opsKey() });
    const pending = all.filter((a) => a.status === 'pending');
    if (!pending.length) { box.innerHTML = '<p class="muted">No approvals pending.</p>'; return; }
    box.innerHTML = pending.map((a) => `<div class="approval">
      <p><b class="mono">${esc(a.approval_id)}</b>: ${esc(a.action)} ${esc(a.target)} ${esc(a.version || '')}. ${esc(a.why)}.</p>
      <p class="small muted">Reason: ${esc(a.reason)}</p>
      <div class="actions"><button class="btn accent" type="button" data-approve="${esc(a.approval_id)}">Approve</button><button class="btn ghost light" type="button" data-reject="${esc(a.approval_id)}">Reject</button></div>
    </div>`).join('');
  } catch (e) { box.innerHTML = `<p class="error">${esc(e.message)}</p>`; }
}
$('#approvals').addEventListener('click', async (ev) => {
  const btn = ev.target.closest('button[data-approve],button[data-reject]'); if (!btn) return;
  const approve = btn.hasAttribute('data-approve'); const id = btn.dataset.approve || btn.dataset.reject;
  btn.disabled = true; setStatus(approve ? 'Promoting the approved model and running a check…' : 'Recording the rejection…');
  try {
    const r = await api(`/v1/governance/approvals/${id}`, { method: 'POST', body: { approve, reviewer: 'web-reviewer', note: approve ? 'approved on the website' : 'rejected on the website' }, key: opsKey() });
    setStatus(`${id} ${r.status}${r.post_decision_check ? `: ${r.post_decision_check.summary}` : ''}.`);
    await Promise.all([loadApprovals(), loadBoard(), loadRegistry()]);
  } catch (e) { setStatus(`Error: ${e.message}`); btn.disabled = false; }
});

async function loadRegistry() {
  const box = $('#tab-registry');
  try {
    const rows = await api('/v1/governance/models', { key: opsKey() });
    box.innerHTML = table(['Model', 'Version', 'Status', 'Parent', 'Created', 'Metrics', 'Notes'], rows.map((r) => [esc(r.model), `<span class="mono">${esc(r.version)}</span>`, esc(r.status), esc(r.parent || '-'), esc(r.created), esc(r.metrics), esc(r.notes)]), 'dark');
  } catch (e) { box.innerHTML = `<p class="error">${esc(e.message)}</p>`; }
}
async function loadAudit() {
  const box = $('#tab-audit'); busy(box);
  try {
    const [entries, ver] = await Promise.all([api('/v1/governance/audit?n=30', { key: opsKey() }), api('/v1/governance/audit/verify', { key: opsKey() })]);
    box.innerHTML = `<div class="notice ${ver.valid ? 'ok' : 'bad'}">${ver.entries} entries; hash chain ${ver.valid ? 'intact' : `broken at entry ${ver.first_broken_seq}`}.</div>`
      + table(['#', 'Time (UTC)', 'Actor', 'Action', 'Resource', 'Severity'], entries.slice().reverse().map((e) => [e.seq, esc(e.ts), esc(e.actor), esc(e.action), esc(e.resource), esc(e.severity)]), 'dark');
  } catch (e) { fail(box, e); }
}
async function loadReport() {
  const box = $('#tab-report'); busy(box);
  try {
    const text = await api('/v1/governance/report', { key: opsKey() });
    box.innerHTML = `<pre>${esc(text)}</pre>`;
  } catch (e) { fail(box, e); }
}
$$('.tabs button').forEach((b) => b.addEventListener('click', () => {
  $$('.tabs button').forEach((x) => x.classList.toggle('active', x === b));
  $$('.tab-body').forEach((x) => x.classList.toggle('hidden', x.id !== `tab-${b.dataset.tab}`));
  if (b.dataset.tab === 'audit') loadAudit();
  if (b.dataset.tab === 'report') loadReport();
}));
$('#opsKey').addEventListener('change', () => { loadBoard(); loadScenarios(); loadApprovals(); loadRegistry(); });

/* ---------------------------------------------------------------- boot */
bindClassSelects();
loadTrains().catch((e) => { $('#planResult').innerHTML = `<p class="error">Could not reach the API: ${esc(e.message)}</p>`; });
loadBoard(); loadScenarios(); loadApprovals(); loadRegistry();
