// Five more scenes for the neon sim engine: maze, flappy, moon, dodge, pole. Loaded after goal2.js.
const FC = {};
const lerp3 = (a, b, t) => a + (b - a) * t;

// ---------------------------------------------------------------- maze
const MZN = 14, MCELL = 850 / 14, MX0 = AX + 75, MY0 = AY + 140;
const mcx = c => MX0 + c * MCELL + MCELL / 2, mcy = c => MY0 + c * MCELL + MCELL / 2;
function mazeWalls(walls, col, lw) {
  X.save(); X.strokeStyle = col; X.lineWidth = lw; X.lineCap = 'round'; X.shadowColor = col; X.shadowBlur = 10; X.beginPath();
  for (let y = 0; y < MZN; y++) for (let x = 0; x < MZN; x++) { const w = walls[y][x]; const x0 = MX0 + x * MCELL, y0 = MY0 + y * MCELL;
    if (w.includes(0) && x < MZN - 1) { X.moveTo(x0 + MCELL, y0); X.lineTo(x0 + MCELL, y0 + MCELL); }
    if (w.includes(1) && y < MZN - 1) { X.moveTo(x0, y0 + MCELL); X.lineTo(x0 + MCELL, y0 + MCELL); } }
  X.stroke(); X.strokeStyle = col; X.lineWidth = lw + 2; X.beginPath(); rr(MX0, MY0, MZN * MCELL, MZN * MCELL, 4); X.stroke(); X.restore();
}
SC.maze = (b, fi, lt) => {
  const d = b.data, C = D.common; const sf = fi / 1.2; const walls = d.phase ? C.walls1 : C.walls0;
  arena(d.phase ? '#ff3fa4' : '#5a74ff');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  mazeWalls(walls, d.phase ? 'rgba(255,120,200,.75)' : 'rgba(120,150,255,.75)', 5);
  // start + exit
  glow(mcx(0), mcy(0), 60, CYAN, 0.25); txt('START', mcx(0), mcy(0) + 2, 18, { fill: CYAN, font: 'PopM' });
  const pulse = 0.8 + 0.2 * Math.sin(lt * 6); glow(mcx(MZN - 1), mcy(MZN - 1), 110 * pulse, GREEN, 0.5); dot(mcx(MZN - 1), mcy(MZN - 1), 16, GREEN, 0); txt('EXIT', mcx(MZN - 1), mcy(MZN - 1) + 30, 18, { fill: GREEN, font: 'PopM' });
  // block wall
  if (d.phase && C.block) { const [x, y, nx, ny] = C.block; const k = eOutBack(clamp(lt / 0.5)); const px = (mcx(x) + mcx(nx)) / 2, py = (mcy(y) + mcy(ny)) / 2; const horiz = ny !== y;
    X.save(); X.translate(px, py); X.scale(k, k); glow(0, 0, 70, RED, 0.6); X.strokeStyle = RED; X.lineWidth = 12; X.lineCap = 'round'; X.beginPath(); if (horiz) { X.moveTo(-MCELL / 2, 0); X.lineTo(MCELL / 2, 0); } else { X.moveTo(0, -MCELL / 2); X.lineTo(0, MCELL / 2); } X.stroke(); X.restore(); }
  // agents
  let reached = 0; const n = d.paths.length;
  d.paths.forEach((p, i) => { const col = d.ok[i] ? LIME : (i % 3 === 0 ? CYAN : (i % 3 === 1 ? MAG : YEL)); const last = p.length - 1; const s = Math.min(sf, last); const j = Math.floor(s), f = s - j; const q = p[Math.min(j + 1, last)];
    const x = lerp3(mcx(p[j][0]), mcx(q[0]), f), y = lerp3(mcy(p[j][1]), mcy(q[1]), f);
    if (d.ok[i] && sf >= last) { reached++; return; }
    for (let k = 1; k <= 6; k++) { const jj = j - k; if (jj < 0) break; X.save(); X.globalAlpha = 0.25 * (7 - k) / 6; X.fillStyle = col; X.beginPath(); X.arc(mcx(p[jj][0]), mcy(p[jj][1]), 7, 0, 7); X.fill(); X.restore(); }
    dot(x, y, 9, col, 0.9); });
  X.restore();
  counter(AX + 170, AY + 50, reached, n, 'REACHED', reached >= n * 0.6 ? GREEN : YEL, lt);
};
SC.maze.events = (b, fi) => { const d = b.data; const sf = fi / 1.2, pf = (fi - 1) / 1.2; const ev = []; d.paths.forEach((p, i) => { const last = p.length - 1; if (d.ok[i] && sf >= last && pf < last) ev.push({ x: mcx(MZN - 1), y: mcy(MZN - 1), col: GREEN, n: 8, sp: 5, size: 4 }); }); return ev; };
FC.maze = d => Math.ceil(Math.max(...d.paths.map(p => p.length)) * 1.2);

// ---------------------------------------------------------------- flappy
const FBX = 220, FSTR = 3, FPW = 90, FGAP = 250;
SC.flappy = (b, fi, lt) => {
  const d = b.data, C = D.common; const maxl = Math.max(...d.ys.map(p => p.length)); const step = Math.min(fi * FSTR, maxl - 1); const dist = step * 5;
  arena('#b6ff3b');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  const g = X.createLinearGradient(0, AY, 0, AY + AH); g.addColorStop(0, '#0a1030'); g.addColorStop(1, '#13204a'); X.fillStyle = g; X.fillRect(AX, AY, AW, AH);
  // stars
  const r = rnd(9); for (let i = 0; i < 40; i++) { const sx = ((r() * 2000 - dist * 0.2) % 1000 + 1000) % 1000; dot(AX + sx, AY + r() * 1000, 1.5, WHITE, 0); }
  // pipes
  for (const [px, cy] of C.pipes) { const sx = px - dist; if (sx + FPW < 0 || sx > 1000) continue; const x = AX + sx; const top = AY + cy - FGAP / 2, bot = AY + cy + FGAP / 2;
    X.save(); X.fillStyle = '#2fbf71'; X.strokeStyle = LIME; X.lineWidth = 4; X.shadowColor = LIME; X.shadowBlur = 16;
    rr(x, AY - 20, FPW, top - AY + 20, 8); X.fill(); X.stroke(); rr(x - 8, top - 36, FPW + 16, 36, 8); X.fill(); X.stroke();
    rr(x, bot, FPW, AY + AH - bot + 20, 8); X.fill(); X.stroke(); rr(x - 8, bot, FPW + 16, 36, 8); X.fill(); X.stroke(); X.restore(); }
  // birds
  let alive = 0; const n = d.ys.length;
  for (let i = n - 1; i >= 0; i--) { const ys = d.ys[i]; const sv = ys.length >= d.full; if (step >= ys.length && !sv) continue; alive++; const st2 = Math.min(step, ys.length - 1); const y = AY + ys[st2]; const vy = st2 > 0 ? ys[st2] - ys[st2 - 1] : 0; const x = AX + FBX + (i % 7) * 4 - 12;
    const col = i === 0 ? WHITE : YEL; X.save(); X.translate(x, y); X.rotate(clamp(vy / 12, -1, 1) * 0.6); glow(0, 0, 40, col, 0.35); X.fillStyle = col; X.beginPath(); X.ellipse(0, 0, 18, 13, 0, 0, 7); X.fill(); X.fillStyle = ORANGE; X.beginPath(); X.moveTo(14, -2); X.lineTo(26, 2); X.lineTo(14, 6); X.closePath(); X.fill(); X.fillStyle = INK; X.beginPath(); X.arc(7, -4, 3, 0, 7); X.fill(); const wf = Math.sin(fi * 1.2) * 6; X.fillStyle = 'rgba(0,0,0,.25)'; X.beginPath(); X.ellipse(-4, 2 + wf * 0.3, 9, 5, -0.4, 0, 7); X.fill(); X.restore(); }
  X.restore();
  counter(AX + 170, AY + 900, alive, n, 'ALIVE', alive >= n * 0.4 ? GREEN : (alive === 0 ? RED : YEL), lt);
  const passed = Math.max(0, Math.floor((dist - 600 + FPW) / 320) + 1); txt(`PIPES ${Math.max(0, Math.min(passed, Math.floor((dist - 600 + FPW) / 320) + 1))}`, AX + AW - 40, AY + 60, 38, { fill: WHITE, font: 'Mono', align: 'right' });
};
SC.flappy.events = (b, fi) => { const d = b.data; const step = fi * FSTR; const ev = []; d.ys.forEach((ys, i) => { if (ys.length < d.full && step >= ys.length && step - FSTR < ys.length) ev.push({ x: AX + FBX, y: AY + ys[ys.length - 1], col: YEL, n: 10, sp: 6, size: 4 }); }); return ev; };
FC.flappy = d => Math.ceil(Math.max(...d.ys.map(p => p.length)) / FSTR);

// ---------------------------------------------------------------- moon
const MEARTH = [110, 520], MPLAN = [520, 500], MPR = 80, MMOON = [880, 300], MMR = 34;
SC.moon = (b, fi, lt) => {
  const d = b.data; const step = fi * 2; const heavy = d.phase === 1;
  arena(heavy ? '#ff3fa4' : '#6cf0ff');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  const r = rnd(4); for (let i = 0; i < 70; i++) dot(AX + r() * 1000, AY + r() * 1000, 1 + r() * 1.2, WHITE, 0);
  // earth
  glow(AX + MEARTH[0], AY + MEARTH[1], 90, BLUE, 0.5); dot(AX + MEARTH[0], AY + MEARTH[1], 30, '#3f8cff', 0); X.fillStyle = '#2fbf71'; X.beginPath(); X.arc(AX + MEARTH[0] - 6, AY + MEARTH[1] - 8, 11, 0, 7); X.fill(); txt('EARTH', AX + MEARTH[0], AY + MEARTH[1] + 58, 20, { fill: 'rgba(255,255,255,.7)', font: 'PopM' });
  // planet
  const pc = heavy ? '#ff5a3c' : '#ff8c2b'; glow(AX + MPLAN[0], AY + MPLAN[1], MPR * (heavy ? 3.2 : 2.4), pc, heavy ? 0.6 : 0.45); dot(AX + MPLAN[0], AY + MPLAN[1], MPR, heavy ? '#b03020' : '#c25a12', 0);
  X.save(); X.globalAlpha = 0.35; X.fillStyle = '#ffd39b'; X.beginPath(); X.ellipse(AX + MPLAN[0] - 20, AY + MPLAN[1] - 25, 28, 14, -0.3, 0, 7); X.fill(); X.beginPath(); X.ellipse(AX + MPLAN[0] + 25, AY + MPLAN[1] + 20, 22, 10, 0.2, 0, 7); X.fill(); X.restore();
  X.save(); X.strokeStyle = 'rgba(255,255,255,.35)'; X.lineWidth = 6; X.beginPath(); X.ellipse(AX + MPLAN[0], AY + MPLAN[1], MPR * 1.8, MPR * 0.45, -0.25, 0, 7); X.stroke(); X.restore();
  txt(heavy ? 'GRAVITY x2' : 'PLANET', AX + MPLAN[0], AY + MPLAN[1] + MPR + 40, 22, { fill: heavy ? MAG : 'rgba(255,255,255,.7)', font: 'PopM' });
  // moon
  const pulse = 0.8 + 0.2 * Math.sin(lt * 5); glow(AX + MMOON[0], AY + MMOON[1], 120 * pulse, WHITE, 0.35); dot(AX + MMOON[0], AY + MMOON[1], MMR, '#e8e8f0', 0); X.fillStyle = 'rgba(0,0,0,.18)'; X.beginPath(); X.arc(AX + MMOON[0] + 8, AY + MMOON[1] - 6, 8, 0, 7); X.fill(); X.beginPath(); X.arc(AX + MMOON[0] - 10, AY + MMOON[1] + 9, 5, 0, 7); X.fill();
  X.save(); X.strokeStyle = GREEN; X.lineWidth = 3; X.setLineDash([8, 8]); X.beginPath(); X.arc(AX + MMOON[0], AY + MMOON[1], MMR + 14, 0, 7); X.stroke(); X.restore(); txt('MOON', AX + MMOON[0], AY + MMOON[1] - MMR - 26, 22, { fill: 'rgba(255,255,255,.8)', font: 'PopM' });
  // probes
  let hits = 0;
  d.shots.forEach((pts, i) => { const last = pts.length - 1; const s = Math.min(step, last); const col = d.ok[i] ? LIME : (d.crash[i] ? ORANGE : CYAN);
    if (step > last) { if (d.ok[i]) hits++; if (d.ok[i] || d.crash[i]) { X.save(); X.globalAlpha = d.ok[i] ? 0.35 : 0.12; X.strokeStyle = col; X.lineWidth = 2; X.beginPath(); X.moveTo(AX + pts[0][0], AY + pts[0][1]); for (let k = 1; k <= last; k += 2) X.lineTo(AX + pts[k][0], AY + pts[k][1]); X.stroke(); X.restore(); return; } }
    X.save(); X.lineWidth = 3; X.lineCap = 'round'; for (let k = 0; k < 30; k++) { const j = s - k; if (j < 1) break; X.globalAlpha = 0.5 * (30 - k) / 30; X.strokeStyle = col; X.beginPath(); X.moveTo(AX + pts[j - 1][0], AY + pts[j - 1][1]); X.lineTo(AX + pts[j][0], AY + pts[j][1]); X.stroke(); } X.restore();
    if (step <= last) dot(AX + pts[s][0], AY + pts[s][1], 5, col, 0.9); });
  X.restore();
  counter(AX + 170, AY + 880, hits, d.shots.length, 'HIT THE MOON', hits >= 60 ? GREEN : YEL, lt);
};
SC.moon.events = (b, fi) => { const d = b.data; const step = fi * 2; const ev = []; d.shots.forEach((pts, i) => { const n = pts.length; if (step >= n && step - 2 < n) { const p = pts[n - 1]; if (d.ok[i]) ev.push({ x: AX + p[0], y: AY + p[1], col: GREEN, n: 10, sp: 5, size: 4 }); else if (d.crash[i]) ev.push({ x: AX + p[0], y: AY + p[1], col: ORANGE, n: 14, sp: 7, size: 5 }); } }); return ev; };
FC.moon = d => Math.ceil(Math.max(...d.shots.map(p => p.length)) / 2);

// ---------------------------------------------------------------- dodge
const DSTR = 3, DGY = 900;
function rock(x, y, r, a) { X.save(); X.translate(x, y); X.rotate(a); glow(0, 0, r * 2.6, ORANGE, 0.45); X.fillStyle = '#5a2a1a'; X.strokeStyle = ORANGE; X.lineWidth = 3; X.beginPath(); for (let i = 0; i < 8; i++) { const an = i / 8 * 6.283, rr2 = r * (0.8 + 0.25 * ((i * 7) % 3) / 2); X.lineTo(Math.cos(an) * rr2, Math.sin(an) * rr2); } X.closePath(); X.fill(); X.stroke(); X.fillStyle = 'rgba(0,0,0,.3)'; X.beginPath(); X.arc(-r * 0.3, -r * 0.2, r * 0.25, 0, 7); X.fill(); X.restore(); }
SC.dodge = (b, fi, lt) => {
  const d = b.data; const step = fi * DSTR; const storm = d.phase === 1;
  arena(storm ? '#ff5a3c' : '#6cf0ff');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  const r = rnd(5); for (let i = 0; i < 60; i++) dot(AX + r() * 1000, AY + ((r() * 1000 + step * 2) % 1000), 1 + r(), WHITE, 0);
  X.save(); X.strokeStyle = 'rgba(108,240,255,.35)'; X.lineWidth = 2; X.beginPath(); X.moveTo(AX, AY + DGY + 30); X.lineTo(AX + AW, AY + DGY + 30); X.stroke(); X.restore();
  // rocks
  for (const [t, x, vy, rad] of d.rocks) { const y = (step - t) * vy; if (step < t || y > 1000 + rad) continue; X.save(); X.globalAlpha = 0.25; X.strokeStyle = ORANGE; X.lineWidth = rad; X.lineCap = 'round'; X.beginPath(); X.moveTo(AX + x, AY + y - vy * 8); X.lineTo(AX + x, AY + y - vy * 2); X.stroke(); X.restore(); rock(AX + x, AY + y, rad, (step - t) * 0.03 + x); }
  // ships
  let alive = 0; const n = d.xs.length;
  for (let i = n - 1; i >= 0; i--) { const xs = d.xs[i]; const sv = xs.length >= d.full; if (step >= xs.length && !sv) continue; alive++; const x = AX + xs[Math.min(step, xs.length - 1)], y = AY + DGY; const col = i === 0 ? WHITE : CYAN; glow(x, y, 34, col, 0.3); X.save(); X.translate(x, y); X.fillStyle = col; X.beginPath(); X.moveTo(0, -16); X.lineTo(12, 12); X.lineTo(0, 6); X.lineTo(-12, 12); X.closePath(); X.fill(); X.fillStyle = ORANGE; X.globalAlpha = 0.7; X.beginPath(); X.moveTo(-4, 10); X.lineTo(0, 22 + (fi % 3) * 3); X.lineTo(4, 10); X.closePath(); X.fill(); X.restore(); }
  X.restore();
  counter(AX + 170, AY + 150, alive, n, 'ALIVE', alive >= n * 0.5 ? GREEN : (alive === 0 ? RED : YEL), lt);
};
SC.dodge.events = (b, fi) => { const d = b.data; const step = fi * DSTR; const ev = []; d.xs.forEach((xs, i) => { if (xs.length < d.full && step >= xs.length && step - DSTR < xs.length) ev.push({ x: AX + xs[xs.length - 1], y: AY + DGY, col: CYAN, n: 12, sp: 7, size: 5 }); }); return ev; };
FC.dodge = d => Math.ceil(Math.max(...d.xs.map(p => p.length)) / DSTR);

// ---------------------------------------------------------------- pole
const PSTR = 3, PTY = 780, PLEN = 360, PSC = 180;
SC.pole = (b, fi, lt) => {
  const d = b.data; const step = fi * PSTR; const pushed = d.phase === 1; const pushStep = 220;
  arena(pushed ? '#ff3fa4' : '#ffd400');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  // track
  X.save(); X.strokeStyle = 'rgba(255,212,0,.5)'; X.lineWidth = 6; X.beginPath(); X.moveTo(AX + 30, AY + PTY + 50); X.lineTo(AX + AW - 30, AY + PTY + 50); X.stroke(); X.setLineDash([6, 18]); X.lineWidth = 2; X.strokeStyle = 'rgba(255,255,255,.25)'; X.beginPath(); X.moveTo(AX + 30, AY + PTY + 70); X.lineTo(AX + AW - 30, AY + PTY + 70); X.stroke(); X.restore();
  txt('−2.4', AX + 68, AY + PTY + 105, 22, { fill: 'rgba(255,255,255,.5)', font: 'Mono' }); txt('+2.4', AX + 932, AY + PTY + 105, 22, { fill: 'rgba(255,255,255,.5)', font: 'Mono' });
  // push indicator
  if (pushed && step >= pushStep - 10 && step < pushStep + 25) { const k = clamp((step - pushStep + 10) / 10); X.save(); X.globalAlpha = k * (1 - clamp((step - pushStep) / 25)); X.fillStyle = RED; X.beginPath(); X.moveTo(AX + 120, AY + PTY - PLEN); X.lineTo(AX + 300, AY + PTY - PLEN); X.lineTo(AX + 300, AY + PTY - PLEN - 30); X.lineTo(AX + 380, AY + PTY - PLEN + 10); X.lineTo(AX + 300, AY + PTY - PLEN + 50); X.lineTo(AX + 300, AY + PTY - PLEN + 20); X.lineTo(AX + 120, AY + PTY - PLEN + 20); X.closePath(); X.fill(); X.restore(); }
  let alive = 0; const n = d.runs.length; let lead = -1;
  for (let i = n - 1; i >= 0; i--) { const run = d.runs[i]; if (step >= run.length && run.length < d.full) continue; alive++; lead = i; }
  for (let i = n - 1; i >= 0; i--) { const run = d.runs[i]; if (step >= run.length && run.length < d.full) continue; const [x, th] = run[Math.min(step, run.length - 1)]; const cx = AX + 500 + x * PSC, cy = AY + PTY; const col = i === lead ? WHITE : CYAN;
    X.save(); X.globalAlpha = i === lead ? 1 : 0.45; const tx = cx + Math.sin(th) * PLEN, ty = cy - Math.cos(th) * PLEN;
    X.strokeStyle = i === lead ? YEL : ORANGE; X.lineWidth = i === lead ? 14 : 8; X.lineCap = 'round'; X.beginPath(); X.moveTo(cx, cy - 16); X.lineTo(tx, ty); X.stroke();
    rr(cx - 50, cy - 18, 100, 46, 10); X.fillStyle = col; X.fill(); X.fillStyle = INK; X.beginPath(); X.arc(cx - 28, cy + 36, 12, 0, 7); X.arc(cx + 28, cy + 36, 12, 0, 7); X.fill();
    if (i === lead) glow(tx, ty, 40, YEL, 0.6); dot(tx, ty, i === lead ? 10 : 6, i === lead ? YEL : ORANGE, i === lead ? 1 : 0); X.restore(); }
  X.restore();
  counter(AX + 170, AY + 150, alive, n, 'BALANCING', alive >= n * 0.5 ? GREEN : (alive === 0 ? RED : YEL), lt);
  txt(`${(step * 0.02).toFixed(1)}s`, AX + AW - 40, AY + 60, 38, { fill: WHITE, font: 'Mono', align: 'right' });
};
SC.pole.events = (b, fi) => { const d = b.data; const step = fi * PSTR; const ev = []; d.runs.forEach((run, i) => { if (run.length < d.full && step >= run.length && step - PSTR < run.length) { const [x, th] = run[run.length - 1]; ev.push({ x: AX + 500 + x * PSC + Math.sin(th) * PLEN, y: AY + PTY - Math.cos(th) * PLEN, col: ORANGE, n: 10, sp: 6, size: 4 }); } }); return ev; };
FC.pole = d => Math.ceil(Math.max(...d.runs.map(p => p.length)) / PSTR);

// hook into frameCount
const _fc = window.frameCount;
window.frameCount = (i) => { const b = beats[i]; if (b.kind === 'end') return 0; if (FC[D.reel]) return FC[D.reel](b.data); return _fc(i); };
