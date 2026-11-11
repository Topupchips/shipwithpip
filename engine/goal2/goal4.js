// Scenes: walk, shortcut, drone, password, keepups. Loaded after goal3.js (uses FC, SC, helpers).

// ---------------------------------------------------------------- walk
const WGY = AY + 760, WSC = 1.1, WFR = 2; // 2 recorded samples per frame
const WMUSC = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]];
SC.walk = (b, fi, lt) => {
  const d = b.data; const T = d.runs[0].length; const s = Math.min(fi * WFR, T - 1);
  arena('#b6ff3b');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  // leader = max mean x at this sample
  let lead = 0, lx = -1e9; const mx = d.runs.map(r => r[s].reduce((a, p) => a + p[0], 0) / 4); d.runs.forEach((r, i) => { if (mx[i] > lx) { lx = mx[i]; lead = i; } });
  const cam = Math.max(-120, lx - 520 / WSC); const sx = x => AX + (x - cam) * WSC, sy = y => WGY + y * WSC;
  // sky gradient + ground
  const g = X.createLinearGradient(0, AY, 0, WGY); g.addColorStop(0, '#0a1030'); g.addColorStop(1, '#1a2a5a'); X.fillStyle = g; X.fillRect(AX, AY, AW, WGY - AY);
  X.fillStyle = '#132244'; X.fillRect(AX, WGY, AW, AY + AH - WGY); X.save(); X.strokeStyle = LIME; X.lineWidth = 4; X.shadowColor = LIME; X.shadowBlur = 14; X.beginPath(); X.moveTo(AX, WGY); X.lineTo(AX + AW, WGY); X.stroke(); X.restore();
  for (let m = Math.floor(cam / 250) * 250; m < cam + 1000 / WSC + 250; m += 250) { if (m < 0) continue; const x = sx(m); X.save(); X.strokeStyle = 'rgba(255,255,255,.25)'; X.lineWidth = 2; X.beginPath(); X.moveTo(x, WGY); X.lineTo(x, WGY + 24); X.stroke(); X.restore(); txt(`${m}`, x, WGY + 52, 22, { fill: 'rgba(255,255,255,.55)', font: 'Mono' }); }
  // creatures
  const draw = (r, i, isLead) => { const P = r[s]; X.save(); X.globalAlpha = isLead ? 1 : 0.14; X.lineWidth = isLead ? 8 : 3; X.lineCap = 'round';
    WMUSC.forEach(([a, c], k) => { X.strokeStyle = isLead ? (k < 4 ? YEL : ORANGE) : CYAN; X.beginPath(); X.moveTo(sx(P[a][0]), sy(P[a][1])); X.lineTo(sx(P[c][0]), sy(P[c][1])); X.stroke(); });
    P.forEach((p, k) => { const fr = d.fr[i][k]; const col = isLead ? (fr > 0.5 ? MAG : WHITE) : CYAN; if (isLead) glow(sx(p[0]), sy(p[1]), 30, col, 0.5); X.fillStyle = col; X.beginPath(); X.arc(sx(p[0]), sy(p[1]), isLead ? 13 : 5, 0, 7); X.fill(); }); X.restore(); };
  d.runs.forEach((r, i) => { if (i !== lead) draw(r, i, false); }); draw(d.runs[lead], lead, true);
  X.restore();
  { const k = eOutBack(clamp((lt - 0.1) / 0.4)); X.save(); X.translate(AX + 170, AY + 150); X.scale(k, k); const v = Math.max(0, Math.round(lx)); const col = lx > 1500 ? GREEN : YEL; txt(String(v), 0, 0, 110, { fill: col, font: 'Pop', stroke: INK, strokeW: 12 }); txt(String(v), 0, 0, 110, { fill: col }); txt('LEADER DISTANCE', 0, 78, 34, { fill: 'rgba(255,255,255,.75)', font: 'PopM' }); X.restore(); }
  txt(`${(s * 3 * 0.033).toFixed(1)}s`, AX + AW - 40, AY + 60, 38, { fill: WHITE, font: 'Mono', align: 'right' });
};
SC.walk.events = () => [];
FC.walk = d => Math.ceil(d.runs[0].length / WFR);

// ---------------------------------------------------------------- shortcut
const SSTR = 8;
SC.shortcut = (b, fi, lt) => {
  const d = b.data; const T = Math.max(...d.runs.map(r => r.length)); const step = Math.min(fi * SSTR, T - 1);
  arena('#ffd400');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  // asphalt lanes
  X.fillStyle = '#141a3a'; X.fillRect(AX + 20, AY + 20, 960, 960); X.fillStyle = '#0b1026'; X.fillRect(AX + 300, AY + 300, 400, 400);
  // corridor floor
  X.fillStyle = '#141a3a'; X.fillRect(AX + 440, AY + 300, 120, 400);
  // walls
  X.save(); X.strokeStyle = YEL; X.lineWidth = 6; X.shadowColor = YEL; X.shadowBlur = 14; X.strokeRect(AX + 20, AY + 20, 960, 960);
  X.beginPath(); X.moveTo(AX + 300, AY + 300); X.lineTo(AX + 440, AY + 300); X.moveTo(AX + 560, AY + 300); X.lineTo(AX + 700, AY + 300); X.lineTo(AX + 700, AY + 700); X.lineTo(AX + 560, AY + 700); X.moveTo(AX + 440, AY + 700); X.lineTo(AX + 300, AY + 700); X.lineTo(AX + 300, AY + 300); X.stroke();
  X.strokeStyle = MAG; X.shadowColor = MAG; X.beginPath(); X.moveTo(AX + 440, AY + 300); X.lineTo(AX + 440, AY + 700); X.moveTo(AX + 560, AY + 300); X.lineTo(AX + 560, AY + 700); X.stroke(); X.restore();
  // finish line
  X.save(); X.setLineDash([10, 10]); X.strokeStyle = WHITE; X.lineWidth = 4; X.beginPath(); X.moveTo(AX + 620, AY + 20); X.lineTo(AX + 620, AY + 300); X.stroke(); X.restore(); txt('FINISH', AX + 620, AY + 160, 20, { fill: 'rgba(255,255,255,.6)', font: 'PopM' });
  txt('THE GAP', AX + 500, AY + 500, 22, { fill: MAG, font: 'PopM' }); X.save(); X.fillStyle = MAG; X.beginPath(); X.moveTo(AX + 500, AY + 440); X.lineTo(AX + 486, AY + 462); X.lineTo(AX + 514, AY + 462); X.closePath(); X.fill(); X.restore();
  // cars
  let cut = 0; const n = d.runs.length;
  for (let i = n - 1; i >= 0; i--) { const r = d.runs[i]; const st = Math.min(step, r.length - 1); const [x, y, a] = r[st]; const user = d.usedat[i] >= 0 && d.usedat[i] <= step; if (user) cut++;
    const col = user ? MAG : (i === 0 ? WHITE : CYAN);
    X.save(); X.globalAlpha = 0.35; X.strokeStyle = col; X.lineWidth = 3; X.beginPath(); for (let k = 0; k < 12; k++) { const j = st - k * 2; if (j < 0) break; const p = r[j]; if (k === 0) X.moveTo(AX + p[0], AY + p[1]); else X.lineTo(AX + p[0], AY + p[1]); } X.stroke(); X.restore();
    X.save(); X.translate(AX + x, AY + y); X.rotate(a); if (user) glow(0, 0, 36, MAG, 0.5); X.fillStyle = col; X.beginPath(); X.moveTo(20, 0); X.lineTo(-13, 12); X.lineTo(-8, 0); X.lineTo(-13, -12); X.closePath(); X.fill(); X.restore(); }
  X.restore();
  counter(AX + 170, AY + 150, cut, n, 'TOOK THE GAP', cut > 0 ? MAG : WHITE, lt);
  txt(`${(step * 0.033).toFixed(1)}s`, AX + AW - 40, AY + 60, 38, { fill: WHITE, font: 'Mono', align: 'right' });
};
SC.shortcut.events = (b, fi) => { const d = b.data; const step = fi * SSTR; const ev = []; d.usedat.forEach((t, i) => { if (t >= 0 && t <= step && t > step - SSTR) { const r = d.runs[i]; const p = r[Math.min(t, r.length - 1)]; ev.push({ x: AX + p[0], y: AY + p[1], col: MAG, n: 14, sp: 6, size: 5 }); } }); return ev; };
FC.shortcut = d => Math.ceil(Math.max(...d.runs.map(r => r.length)) / SSTR);

// ---------------------------------------------------------------- drone
const RINGS0 = [[250, 700], [500, 380], [780, 600], [820, 220], [480, 150], [180, 320]]; const DRSTR = 4;
function droneSprite(x, y, th, side, col, a, fi) { X.save(); X.translate(x, y); X.globalAlpha = a; X.rotate(side * 0.6); X.fillStyle = col; rr(-16, -5, 32, 10, 4); X.fill(); X.strokeStyle = col; X.lineWidth = 3; X.beginPath(); X.moveTo(-14, -4); X.lineTo(-24, -12); X.moveTo(14, -4); X.lineTo(24, -12); X.stroke(); const sp = (fi * 1.7) % 6.28; X.globalAlpha = a * 0.7; X.beginPath(); X.ellipse(-24, -13, 12, 3 + Math.abs(Math.sin(sp)) * 2, 0, 0, 7); X.ellipse(24, -13, 12, 3 + Math.abs(Math.cos(sp)) * 2, 0, 0, 7); X.fill(); if (th > 0.3) { X.globalAlpha = a * 0.8; X.fillStyle = ORANGE; X.beginPath(); X.moveTo(-6, 6); X.lineTo(0, 6 + th * 40); X.lineTo(6, 6); X.closePath(); X.fill(); } X.restore(); }
SC.drone = (b, fi, lt) => {
  const d = b.data; const T = Math.max(...d.runs.map(r => r.length)); const step = Math.min(fi * DRSTR, T - 1); const small = d.phase === 1; const R = small ? 26 : 46;
  arena(small ? '#ff3fa4' : '#6cf0ff');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  const rnd2 = rnd(7); for (let i = 0; i < 50; i++) dot(AX + rnd2() * 1000, AY + rnd2() * 1000, 1 + rnd2(), WHITE, 0);
  RINGS0.forEach(([x, y], i) => { const col = i === 5 ? LIME : (small ? MAG : CYAN); glow(AX + x, AY + y, R * 2.4, col, 0.3); X.save(); X.strokeStyle = col; X.lineWidth = 7; X.shadowColor = col; X.shadowBlur = 18; X.beginPath(); X.arc(AX + x, AY + y, R, 0, 7); X.stroke(); X.restore(); txt(`${i + 1}`, AX + x, AY + y + 2, 26, { fill: col, font: 'PopM' }); });
  let fin = 0; const n = d.runs.length;
  for (let i = n - 1; i >= 0; i--) { const r = d.runs[i]; const st = Math.min(step, r.length - 1); const [x, y, th, side] = r[st]; const done = d.rings[i] >= 6 && d.passed[i].length >= 6 && d.passed[i][5] <= step; if (done) { fin++; continue; }
    const col = i === 0 ? WHITE : YEL;
    X.save(); X.globalAlpha = 0.3; X.strokeStyle = col; X.lineWidth = 2; X.beginPath(); for (let k = 0; k < 14; k++) { const j = st - k * 2; if (j < 0) break; const p = r[j]; if (k === 0) X.moveTo(AX + p[0], AY + p[1]); else X.lineTo(AX + p[0], AY + p[1]); } X.stroke(); X.restore();
    droneSprite(AX + x, AY + y, th, side, col, i === 0 ? 1 : 0.75, fi); }
  X.restore();
  counter(AX + 170, AY + 880, fin, n, 'FINISHED', fin >= 25 ? GREEN : YEL, lt);
  txt(`${(step * 0.033).toFixed(1)}s`, AX + AW - 40, AY + 60, 38, { fill: WHITE, font: 'Mono', align: 'right' });
};
SC.drone.events = (b, fi) => { const d = b.data; const step = fi * DRSTR; const ev = []; d.passed.forEach((pa, i) => pa.forEach((t, k) => { if (t <= step && t > step - DRSTR) { const [x, y] = RINGS0[k]; ev.push({ x: AX + x, y: AY + y, col: k === 5 ? GREEN : CYAN, n: 6, sp: 4, size: 3 }); } })); return ev; };
FC.drone = d => Math.ceil(Math.max(...d.runs.map(r => r.length)) / DRSTR);

// ---------------------------------------------------------------- password
SC.password = (b, fi, lt) => {
  const d = b.data; const target = D.common.target; const L = target.length;
  arena('#b6ff3b');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  // best guess big
  const best = d.guesses[0]; const cw = 60; const bx0 = AX + 500 - (L * cw) / 2 + cw / 2;
  for (let k = 0; k < L; k++) { const ok = best[k] === target[k]; const settle = clamp((lt - 0.1 - k * 0.04) / 0.25); const ch = settle < 1 ? String.fromCharCode(65 + Math.floor(rnd(fi * 7 + k)() * 26)) : best[k]; const col = settle < 1 ? WHITE : (ok ? LIME : 'rgba(255,255,255,.35)');
    if (settle >= 1 && ok) glow(bx0 + k * cw, AY + 120, 40, LIME, 0.35); txt(ch === ' ' ? '·' : ch, bx0 + k * cw, AY + 120, 64, { fill: col, font: 'Mono' }); }
  X.save(); X.strokeStyle = 'rgba(255,255,255,.2)'; X.lineWidth = 2; X.beginPath(); X.moveTo(AX + 80, AY + 190); X.lineTo(AX + 920, AY + 190); X.stroke(); X.restore();
  // 100 guesses, 4 cols x 25 rows
  const cols = 4, rows = 25; const cellw = 230, rowh = 30; const x0 = AX + 500 - (cols * cellw) / 2 + cellw / 2; const y0 = AY + 230;
  for (let i = 0; i < d.guesses.length; i++) { const c = i % cols, r = Math.floor(i / cols); const gx = x0 + c * cellw, gy = y0 + r * rowh; const appear = clamp((lt - 0.05 - i * 0.004) / 0.2); if (appear <= 0) continue; const gstr = d.guesses[i]; const right = d.right[i];
    X.save(); X.globalAlpha = appear; X.font = '22px Mono'; X.textAlign = 'center'; X.textBaseline = 'middle';
    for (let k = 0; k < L; k++) { const ok = gstr[k] === target[k]; X.fillStyle = ok ? LIME : 'rgba(255,255,255,.3)'; X.fillText(gstr[k] === ' ' ? '·' : gstr[k], gx - (L * 15) / 2 + 7 + k * 15, gy); }
    X.fillStyle = right === L ? LIME : 'rgba(255,255,255,.45)'; X.font = '18px Mono'; X.fillText(`${right}`, gx + (L * 15) / 2 + 24, gy); X.restore(); }
  X.restore();
};
SC.password.events = () => [];
FC.password = d => 90;

// ---------------------------------------------------------------- keepups
const KBW = 180, KBH = 216, KSCL = 0.9, KSTR = 5;
SC.keepups = (b, fi, lt) => {
  const d = b.data; const T = Math.max(...d.runs.map(r => r.length)); const step = Math.min(fi * KSTR, T - 1); const nb = d.nballs;
  arena(nb === 2 ? '#ff3fa4' : '#6cf0ff');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  let alive = 0;
  d.runs.forEach((r, i) => { const c = i % 5, row = Math.floor(i / 5); const bx = AX + 22 + c * (KBW + 14), by = AY + 30 + row * (KBH + 24); const dead = step >= r.length && r.length < d.full; const st = Math.min(step, r.length - 1); const f = r[st]; if (!dead) alive++;
    X.save(); X.translate(bx, by); X.globalAlpha = dead ? 0.35 : 1;
    X.fillStyle = dead ? '#1a0f1f' : '#0e1838'; rr(0, 0, KBW, KBH, 10); X.fill(); X.strokeStyle = dead ? RED : (i === 0 ? WHITE : 'rgba(108,240,255,.6)'); X.lineWidth = dead ? 3 : 2; rr(0, 0, KBW, KBH, 10); X.stroke();
    X.save(); rr(0, 0, KBW, KBH, 10); X.clip();
    // paddle
    const px = f[0] * KSCL; X.fillStyle = dead ? RED : YEL; if (!dead) glow(px, KBH - 12, 30, YEL, 0.3); rr(px - 35 * KSCL, KBH - 16, 70 * KSCL, 8, 4); X.fill();
    for (let k = 0; k < nb; k++) { const x = f[1 + k * 2] * KSCL, y = f[2 + k * 2] * KSCL; const col = k === 0 ? CYAN : MAG; if (!dead) glow(x, y, 24, col, 0.5); X.fillStyle = col; X.beginPath(); X.arc(x, y, 7, 0, 7); X.fill(); }
    X.restore();
    // hits so far
    const h = d.hitTimes[i].filter(t => t <= step).length; txt(`${h}`, KBW - 14, 18, 20, { fill: dead ? RED : LIME, font: 'Mono', align: 'right' });
    X.restore(); });
  X.restore();
  txt(`${alive} / 20 STILL UP`, AX + 500, AY + AH + 34, 30, { fill: alive > 10 ? LIME : (alive === 0 ? RED : YEL), font: 'PopM' });
};
SC.keepups.events = (b, fi) => { const d = b.data; const step = fi * KSTR; const ev = []; d.runs.forEach((r, i) => { if (r.length < d.full && step >= r.length && step - KSTR < r.length) { const c = i % 5, row = Math.floor(i / 5); const f = r[r.length - 1]; ev.push({ x: AX + 22 + c * (KBW + 14) + f[1] * KSCL, y: AY + 30 + row * (KBH + 24) + KBH, col: RED, n: 8, sp: 4, size: 3 }); } }); return ev; };
FC.keepups = d => Math.ceil(Math.max(...d.runs.map(r => r.length)) / KSTR);
