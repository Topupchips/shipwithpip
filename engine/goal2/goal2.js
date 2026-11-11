// Neon sim reels: many AIs, one goal. Uses primitives from lib.js (txt, rr, sfx, burst, speedLines...).
const AX = 40, AY = 470, AW = 1000, AH = 1000; // arena box
const ORANGE = '#ff8c2b', MAG = '#ff3fa4', LIME = '#b6ff3b', WHITE = '#ffffff';
const NCOL = { RED, YEL, BLUE, GREEN, CYAN, ORANGE, MAG, LIME, WHITE, INK };
const NC = n => NCOL[n] || n || WHITE;
let D = null, beats = [], T = 0, DUR = [];

// ---------------------------------------------------------------- glow sprites + particles
const glowCache = {};
function glowSprite(col) {
  if (glowCache[col]) return glowCache[col];
  const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d');
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64); gr.addColorStop(0, col); gr.addColorStop(0.35, col); gr.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128); return glowCache[col] = c;
}
function glow(x, y, r, col, a = 0.6) { X.save(); X.globalAlpha = a; X.globalCompositeOperation = 'lighter'; X.drawImage(glowSprite(col), x - r, y - r, r * 2, r * 2); X.restore(); }
function dot(x, y, r, col, g = 1) { if (g) glow(x, y, r * 3.2, col, 0.35 * g); X.beginPath(); X.arc(x, y, r, 0, 7); X.fillStyle = col; X.fill(); }
let parts = [], lastKey = -2;
function spawn(x, y, col, n = 18, sp = 9, life = 26, size = 5) { const r = rnd(Math.floor(x * 7 + y * 13 + n)); for (let i = 0; i < n; i++) { const a = r() * 6.283, v = sp * (0.3 + r()); parts.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - 2, col, life, t: life * (0.6 + r() * 0.4), size: size * (0.5 + r()) }); } }
function stepParts() { for (const p of parts) { p.x += p.vx; p.y += p.vy; p.vy += 0.25; p.vx *= 0.96; p.vy *= 0.96; p.t--; } parts = parts.filter(p => p.t > 0); }
function drawParts() { for (const p of parts) { const a = clamp(p.t / p.life); dot(p.x, p.y, p.size * a + 1, p.col, 0.6); } }

// ---------------------------------------------------------------- chrome
function bg(t) {
  const g = X.createLinearGradient(0, 0, 0, H); g.addColorStop(0, '#07091c'); g.addColorStop(0.5, '#0d1230'); g.addColorStop(1, '#070818'); X.fillStyle = g; X.fillRect(0, 0, W, H);
  X.save(); X.globalAlpha = 0.5; X.fillStyle = dotPattern(1.2, 'rgba(120,140,255,.35)', 28); X.fillRect(0, 0, W, H); X.restore();
  X.save(); X.globalCompositeOperation = 'lighter'; const gg = X.createRadialGradient(W / 2, 980, 100, W / 2, 980, 900); gg.addColorStop(0, 'rgba(60,80,220,.22)'); gg.addColorStop(1, 'rgba(0,0,0,0)'); X.fillStyle = gg; X.fillRect(0, 0, W, H); X.restore();
}
function headline(lines, lt) {
  const k = eOutBack(clamp(lt / 0.45)); if (!lines) return;
  X.save(); X.translate(W / 2, 230); X.scale(k, k);
  const s0 = lines[0].length > 16 ? 72 : 88, s1 = lines[1].length > 16 ? 72 : 88;
  txt(lines[0], 0, -52, s0, { fill: WHITE, stroke: INK, strokeW: 14 }); txt(lines[0], 0, -52, s0, { fill: WHITE });
  txt(lines[1], 0, 48, s1, { fill: YEL, stroke: INK, strokeW: 14 }); txt(lines[1], 0, 48, s1, { fill: YEL });
  X.restore();
}
function chip(x, y, label, col, fcol, lt, size = 44) {
  const k = eOutBack(clamp((lt - 0.15) / 0.4)); if (k <= 0) return;
  X.save(); X.font = `${size}px Pop`; const w = X.measureText(label).width + 60; X.restore();
  X.save(); X.translate(x, y); X.scale(k, k); glow(0, 0, w * 0.7, col, 0.25); rr(-w / 2, -40, w, 80, 24); X.fillStyle = col; X.fill(); X.save(); X.lineWidth = 6; X.strokeStyle = INK; X.stroke(); X.restore(); txt(label, 0, 3, size, { fill: fcol }); X.restore();
  return w;
}
function chips(b, lt) {
  const l = b.label || ''; const s = b.stat;
  X.save(); X.font = '44px Pop'; const wl = X.measureText(l).width + 60; const ws = s ? X.measureText(s.text).width + 60 : 0; X.restore();
  const gap = 24; const tot = wl + (s ? gap + ws : 0); let x = W / 2 - tot / 2 + wl / 2;
  chip(x, 392, l, YEL, INK, lt); if (s) chip(x + wl / 2 + gap + ws / 2, 392, s.text, NC(s.col || 'INK'), s.col ? WHITE : WHITE, lt + 0.08);
}
function arena(col = '#5a74ff') {
  X.save(); rr(AX, AY, AW, AH, 26); X.fillStyle = 'rgba(6,8,24,.85)'; X.fill(); X.lineWidth = 4; X.strokeStyle = col; X.shadowColor = col; X.shadowBlur = 24; X.stroke(); X.restore();
}
function capDark(text, lt, dur) {
  if (!text) return;
  const size = 48, font = `${size}px Pop`; const lines = wrap(text, 900, font); const lh = 60; const h = lines.length * lh + 40; const y = 1640 - h / 2;
  const pop = clamp(lt / 0.3);
  X.save(); X.translate(W / 2, y + h / 2); X.scale(1, eOutBack(pop)); X.translate(-W / 2, -y - h / 2);
  rr(70, y, 940, h, 20); X.fillStyle = 'rgba(10,12,32,.92)'; X.fill(); X.lineWidth = 4; X.strokeStyle = 'rgba(120,140,255,.5)'; X.stroke();
  const words = text.split(' '); const total = words.length; const prog = clamp((lt - 0.1) / Math.max(0.5, dur - 0.6)) * total; let wi = 0;
  lines.forEach((l, i) => { const ws = l.split(' '); X.font = font; const widths = ws.map(w => X.measureText(w + ' ').width); const tw = widths.reduce((a, b) => a + b, 0) - X.measureText(' ').width; let x = W / 2 - tw / 2; ws.forEach((w, j) => { const hi = wi < prog; txt(w, x, y + 20 + lh * i + lh / 2, size, { align: 'left', fill: hi ? YEL : WHITE }); x += widths[j]; wi++; }); });
  X.restore();
}
function handleDark() { txt('@shipwithpip', W / 2, 1840, 34, { font: 'PopM', fill: 'rgba(255,255,255,.6)' }); }
function counter(x, y, n, total, label, col, lt) {
  const k = eOutBack(clamp((lt - 0.1) / 0.4)); X.save(); X.translate(x, y); X.scale(k, k);
  txt(String(n), 0, 0, 110, { fill: col, font: 'Pop', stroke: INK, strokeW: 12 }); txt(String(n), 0, 0, 110, { fill: col });
  txt(`/ ${total} ${label}`, 0, 78, 34, { fill: 'rgba(255,255,255,.75)', font: 'PopM' }); X.restore();
}

// ---------------------------------------------------------------- scenes
const SC = {};
// hoops: world 1000x1000 -> arena
const hwx = x => AX + 100 + x * 0.8, hwy = y => AY + 40 + (y + 250) * 0.8;
SC.hoops = (b, fi, lt) => {
  const d = b.data, C = D.common; const step = fi * 2; const hx = hwx(d.hoop[0]), hy = hwy(d.hoop[1]); const FL = hwy(905);
  arena('#ff8c2b');
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip();
  X.save(); X.beginPath(); X.moveTo(AX + 20, FL); X.lineTo(AX + AW - 20, FL); X.lineWidth = 5; X.strokeStyle = 'rgba(255,255,255,.35)'; X.stroke(); X.restore();
  const BBX = hwx(C.HOOP[0] + 50), BBY = hwy(C.HOOP[1] - 220), BBH = 340 * 0.8;
  X.save(); X.fillStyle = 'rgba(255,255,255,.12)'; X.fillRect(BBX, BBY, 12, BBH); X.strokeStyle = 'rgba(255,255,255,.5)'; X.lineWidth = 3; X.strokeRect(BBX, BBY, 12, BBH); X.fillStyle = 'rgba(255,255,255,.25)'; X.fillRect(BBX + 12, BBY, 9, FL - BBY); X.restore();
  const RIM = C.RIM * 0.8;
  glow(hx, hy, 100, ORANGE, 0.35);
  X.save(); X.strokeStyle = 'rgba(255,255,255,.7)'; X.lineWidth = 2; for (let i = 0; i < 7; i++) { const u = i / 6; X.beginPath(); X.moveTo(hx - RIM + u * 2 * RIM, hy); X.lineTo(hx - RIM * 0.6 + u * 1.2 * RIM, hy + 56); X.stroke(); } X.beginPath(); X.moveTo(hx - RIM * 0.6, hy + 56); X.lineTo(hx + RIM * 0.6, hy + 56); X.stroke(); X.restore();
  X.save(); X.beginPath(); X.ellipse(hx, hy, RIM, 10, 0, 0, 7); X.lineWidth = 8; X.strokeStyle = ORANGE; X.stroke(); X.restore();
  if (d.phase === 1) { X.save(); X.strokeStyle = 'rgba(255,255,255,.3)'; X.setLineDash([10, 10]); X.lineWidth = 3; X.beginPath(); X.moveTo(hx, hy + 56); X.lineTo(hx, FL); X.stroke(); X.restore(); }
  const sx = hwx(C.SHOOTER[0]), sy = hwy(C.SHOOTER[1]);
  X.save(); X.strokeStyle = WHITE; X.lineWidth = 7; X.lineCap = 'round'; X.beginPath(); X.arc(sx - 26, sy - 6, 22, 0, 7); X.stroke(); X.beginPath(); X.moveTo(sx - 26, sy + 16); X.lineTo(sx - 30, sy + 72); X.moveTo(sx - 30, sy + 72); X.lineTo(sx - 50, sy + 120); X.moveTo(sx - 30, sy + 72); X.lineTo(sx - 8, sy + 120); X.moveTo(sx - 26, sy + 36); X.lineTo(sx + 8, sy - 6); X.stroke(); X.restore();
  let made = 0;
  d.shots.forEach((pts, i) => { const idx = Math.min(step, pts.length - 1); const done = step >= pts.length - 1; const sw = d.sw[i] && done; if (sw) made++;
    const col = sw ? GREEN : (d.sw[i] ? '#ffd27a' : ORANGE);
    for (let k = 6; k >= 1; k--) { const j = idx - k * 2; if (j < 0) continue; const p = pts[j]; X.save(); X.globalAlpha = 0.12 * (7 - k) / 6; X.beginPath(); X.arc(hwx(p[0]), hwy(p[1]), 10 - k, 0, 7); X.fillStyle = col; X.fill(); X.restore(); }
    const p = pts[idx]; if (done && !sw) return;
    dot(hwx(p[0]), hwy(p[1]), 11, col, sw ? 1.4 : 0.6);
  });
  X.restore();
  counter(AX + 160, AY + 190, made, d.shots.length, 'SWISHES', made >= d.shots.length * 0.9 ? GREEN : YEL, lt);
};
SC.hoops.events = (b, fi) => { const d = b.data; const step = fi * 2; const ev = []; d.shots.forEach((pts, i) => { if (d.sw[i] && step >= pts.length - 1 && step - 2 < pts.length - 1) ev.push({ x: hwx(d.hoop[0]), y: hwy(d.hoop[1]) + 10, col: GREEN, n: 10, sp: 6 }); }); return ev; };

// lander
SC.lander = (b, fi, lt) => {
  const d = b.data, C = D.common; const step = fi * 4;
  arena('#6cf0ff');
  // stars
  const r = rnd(9); X.save(); for (let i = 0; i < 90; i++) { X.globalAlpha = 0.25 + r() * 0.6; X.fillStyle = WHITE; const sx = AX + 20 + r() * 960, sy = AY + 20 + r() * 850; X.fillRect(sx, sy, 2 + r() * 2, 2 + r() * 2); } X.restore();
  // ground + pad
  const gy = AY + C.PADY; X.save(); X.fillStyle = '#1a1f4a'; rr(AX + 4, gy, AW - 8, AH - C.PADY - 4, 0); X.fill(); X.beginPath(); X.moveTo(AX + 6, gy); X.lineTo(AX + AW - 6, gy); X.lineWidth = 4; X.strokeStyle = 'rgba(120,140,255,.6)'; X.stroke(); X.restore();
  glow(AX + 500, gy, 160, GREEN, 0.3); X.save(); rr(AX + C.PADX[0], gy - 10, C.PADX[1] - C.PADX[0], 14, 6); X.fillStyle = GREEN; X.fill(); X.restore();
  txt('PAD', AX + 500, gy + 36, 28, { fill: GREEN, font: 'Mono' });
  let landed = 0;
  d.runs.forEach((fr, i) => { const n = fr.length; if (step >= n) { if (d.ok[i]) { landed++; const p = fr[n - 1]; rocket(AX + p[0], AY + p[1], 0, 0, GREEN, 1); } else if (!d.crash[i] && n >= 460) { const p = fr[n - 1]; rocket(AX + p[0], AY + p[1] + Math.sin(fi * 0.3) * 3, p[2], p[3], WHITE, 0.85); } return; }
    const p = fr[step]; rocket(AX + p[0], AY + p[1], p[2], p[3], WHITE, 0.85);
    for (let k = 5; k >= 1; k--) { const j = step - k * 3; if (j < 0) continue; const q = fr[j]; X.save(); X.globalAlpha = 0.1 * (6 - k) / 5; X.beginPath(); X.arc(AX + q[0], AY + q[1], 4, 0, 7); X.fillStyle = CYAN; X.fill(); X.restore(); }
  });
  counter(AX + 170, AY + 150, landed, d.runs.length, 'LANDED', landed >= d.runs.length * 0.5 ? GREEN : YEL, lt);
};
function rocket(x, y, th, side, col, a) {
  X.save(); X.globalAlpha = a; X.translate(x, y + 30);
  if (th > 0.03) { const L = 20 + th * 170 + Math.random() * 12; glow(0, 10 + L * 0.4, L * 0.8, ORANGE, 0.4); X.beginPath(); X.moveTo(-9, 0); X.lineTo(0, L); X.lineTo(9, 0); X.closePath(); X.fillStyle = YEL; X.fill(); X.beginPath(); X.moveTo(-5, 0); X.lineTo(0, L * 0.55); X.lineTo(5, 0); X.closePath(); X.fillStyle = WHITE; X.fill(); }
  if (Math.abs(side) > 0.015) { const s = Math.sign(side); const L = 14 + Math.abs(side) * 260; X.beginPath(); X.moveTo(s * -12, -30); X.lineTo(s * -12 - s * L, -26); X.lineTo(s * -12, -22); X.closePath(); X.fillStyle = CYAN; X.fill(); }
  X.beginPath(); X.moveTo(0, -60); X.lineTo(12, -30); X.lineTo(12, 0); X.lineTo(-12, 0); X.lineTo(-12, -30); X.closePath(); X.fillStyle = col; X.fill();
  X.beginPath(); X.moveTo(-12, -10); X.lineTo(-20, 4); X.lineTo(-12, 0); X.moveTo(12, -10); X.lineTo(20, 4); X.lineTo(12, 0); X.fillStyle = RED; X.fill();
  X.beginPath(); X.moveTo(0, -60); X.lineTo(12, -30); X.lineTo(-12, -30); X.closePath(); X.fillStyle = RED; X.fill();
  X.restore();
}
SC.lander.events = (b, fi) => { const d = b.data; const step = fi * 4; const ev = []; d.runs.forEach((fr, i) => { const n = fr.length; if (step >= n && step - 4 < n) { const p = fr[n - 1]; if (d.crash[i]) ev.push({ x: AX + p[0], y: AY + p[1] + 20, col: ORANGE, n: 22, sp: 9, size: 7 }); else if (d.ok[i]) ev.push({ x: AX + p[0], y: AY + p[1] + 30, col: GREEN, n: 8, sp: 4, size: 4 }); } }); return ev; };

// snake: 16 boards
const SNC = [CYAN, MAG, LIME, YEL, ORANGE, '#7cf', '#f7c', '#9f9', CYAN, MAG, LIME, YEL, ORANGE, '#7cf', '#f7c', '#9f9'];
SC.snake = (b, fi, lt) => {
  const d = b.data; const step = fi * 2; const GRID = 12; const BS = 232, GAP = 12; const cell = BS / GRID;
  arena('#b6ff3b'); let total = 0;
  d.games.forEach((fr, i) => { const bx = AX + 14 + (i % 4) * (BS + GAP), by = AY + 14 + Math.floor(i / 4) * (BS + GAP); const idx = Math.min(step, fr.length - 1); const [body, food, dead] = fr[idx]; const col = SNC[i]; const score = body.length - 3; total += score;
    X.save(); rr(bx, by, BS, BS, 10); X.fillStyle = dead ? 'rgba(14,10,24,.9)' : 'rgba(10,14,40,.9)'; X.fill(); X.lineWidth = dead ? 3 : 2; X.strokeStyle = dead ? 'rgba(232,52,47,.8)' : 'rgba(120,140,255,.35)'; X.stroke();
    if (!dead || idx < fr.length - 1 || (fi % 10) < 5) {
      X.beginPath(); body.forEach((c, k) => { const x = bx + c[0] * cell + cell / 2, y = by + c[1] * cell + cell / 2; k ? X.lineTo(x, y) : X.moveTo(x, y); }); X.lineWidth = cell * 0.7; X.lineCap = 'round'; X.lineJoin = 'round'; X.strokeStyle = dead ? 'rgba(255,80,80,.7)' : col; X.shadowColor = col; X.shadowBlur = dead ? 0 : 12; X.stroke(); X.shadowBlur = 0;
      const h = body[0]; X.beginPath(); X.arc(bx + h[0] * cell + cell / 2, by + h[1] * cell + cell / 2, cell * 0.42, 0, 7); X.fillStyle = WHITE; X.fill();
    }
    const fp = 0.8 + 0.2 * Math.sin(fi * 0.5); X.beginPath(); X.arc(bx + food[0] * cell + cell / 2, by + food[1] * cell + cell / 2, cell * 0.32 * fp, 0, 7); X.fillStyle = RED; X.shadowColor = RED; X.shadowBlur = 10; X.fill();
    X.restore();
    txt(String(score), bx + BS - 8, by + 18, 26, { fill: dead ? RED : WHITE, font: 'Mono', align: 'right' });
  });
};
SC.snake.events = (b, fi) => { const d = b.data; const step = fi * 2; const BS = 232, GAP = 12, cell = BS / 12; const ev = []; d.games.forEach((fr, i) => { const idx = Math.min(step, fr.length - 1); if (idx >= 1 && step <= fr.length - 1) { const a = fr[idx][0].length, p = fr[Math.max(0, idx - 2)][0].length; if (a > p) { const bx = AX + 14 + (i % 4) * (BS + GAP), by = AY + 14 + Math.floor(i / 4) * (BS + GAP); const h = fr[idx][0][0]; ev.push({ x: bx + h[0] * cell + cell / 2, y: by + h[1] * cell + cell / 2, col: SNC[i], n: 6, sp: 3, size: 3 }); } if (fr[idx][2] && !fr[Math.max(0, idx - 2)][2]) { const bx = AX + 14 + (i % 4) * (BS + GAP), by = AY + 14 + Math.floor(i / 4) * (BS + GAP); const h = fr[idx][0][0]; ev.push({ x: bx + h[0] * cell + cell / 2, y: by + h[1] * cell + cell / 2, col: RED, n: 12, sp: 5, size: 4 }); } } }); return ev; };

// dino
SC.dino = (b, fi, lt) => {
  const d = b.data, C = D.common; const step = fi * 2; const dist = C.dist[Math.min(step, C.dist.length - 1)];
  const oy = AY + 60; // world y offset so GY=760 -> screen ~ AY+820
  arena('#b6ff3b');
  // sky gradient inside arena
  X.save(); rr(AX + 2, AY + 2, AW - 4, AH - 4, 24); X.clip(); const g = X.createLinearGradient(0, AY, 0, AY + AH); g.addColorStop(0, '#0a0d2a'); g.addColorStop(1, '#141a48'); X.fillStyle = g; X.fillRect(AX, AY, AW, AH);
  // moon + clouds
  glow(AX + 820, AY + 150, 120, '#ffe9a8', 0.35); dot(AX + 820, AY + 150, 44, '#ffe9a8', 0);
  const gy = oy + C.GY; X.fillStyle = '#1b2050'; X.fillRect(AX, gy, AW, AY + AH - gy); X.beginPath(); X.moveTo(AX, gy); X.lineTo(AX + AW, gy); X.lineWidth = 4; X.strokeStyle = LIME; X.stroke();
  X.save(); X.strokeStyle = 'rgba(182,255,59,.35)'; X.lineWidth = 3; const off = dist % 80; for (let x = -off; x < AW; x += 80) { X.beginPath(); X.moveTo(AX + x, gy + 30); X.lineTo(AX + x + 30, gy + 30); X.stroke(); } X.restore();
  // cacti
  for (const [cx, cw, ch] of C.cacti) { const sx = cx - dist; if (sx < -100 || sx > AW + 50) continue; cactus(AX + sx, gy, cw, ch); }
  // dinos
  let alive = 0; const n = d.ys.length;
  for (let i = n - 1; i >= 0; i--) { const ys = d.ys[i]; if (step >= ys.length) continue; alive++; const y = ys[step]; const x = AX + 60 + (i % 10) * 7 - 30; dino(x, oy + y, i === 0 ? WHITE : LIME, 0.5, fi, 1.5); }
  X.restore();
  counter(AX + 170, AY + 150, alive, n, 'ALIVE', alive >= n * 0.5 ? GREEN : YEL, lt);
  txt(`SCORE ${Math.floor(dist / 10)}`, AX + AW - 40, AY + 60, 38, { fill: WHITE, font: 'Mono', align: 'right' });
};
function cactus(x, gy, w, h) { w *= 1.5; h *= 1.5; X.save(); X.fillStyle = '#2fbf71'; X.strokeStyle = '#0b3'; X.lineWidth = 3; rr(x + w * 0.3, gy - h, w * 0.4, h, 6); X.fill(); X.stroke(); rr(x, gy - h * 0.7, w * 0.3, h * 0.35, 5); X.fill(); X.stroke(); rr(x + w * 0.7, gy - h * 0.8, w * 0.3, h * 0.4, 5); X.fill(); X.stroke(); X.restore(); }
function dino(x, gy, col, a, fi, sc = 1) {
  X.save(); X.globalAlpha = a; X.translate(x, gy); X.scale(sc, sc); X.fillStyle = col;
  // body (44 wide, 50 tall) blocky t-rex
  X.beginPath(); X.moveTo(0, 0); X.lineTo(0, -28); X.lineTo(12, -40); X.lineTo(22, -50); X.lineTo(44, -50); X.lineTo(44, -38); X.lineTo(30, -38); X.lineTo(30, -28); X.lineTo(26, -28); X.lineTo(26, -12); X.lineTo(20, -12); X.lineTo(20, 0); X.lineTo(14, 0); X.lineTo(14, -8); X.lineTo(8, -8); X.lineTo(8, 0); X.closePath(); X.fill();
  X.fillStyle = INK; X.fillRect(32, -47, 4, 4); X.restore();
}
SC.dino.events = (b, fi) => { const d = b.data; const step = fi * 2; const oy = AY + 60; const ev = []; d.ys.forEach((ys, i) => { if (step >= ys.length && step - 2 < ys.length) ev.push({ x: AX + 60 + 10, y: oy + ys[ys.length - 1] - 20, col: RED, n: 8, sp: 5, size: 4 }); }); return ev; };

// hunt
SC.hunt = (b, fi, lt) => {
  const d = b.data; const step = Math.min(fi * 2, d.frames.length - 1); const f = d.frames[step];
  arena(b.data.prey_learning ? '#6cf0ff' : '#ff3fa4');
  // trails
  for (let k = 6; k >= 1; k--) { const j = step - k * 2; if (j < 0) continue; const q = d.frames[j]; X.save(); X.globalAlpha = 0.14 * (7 - k) / 6; q.p.forEach((p, i) => { if (!q.al[i]) return; X.beginPath(); X.arc(AX + p[0], AY + p[1], 5, 0, 7); X.fillStyle = CYAN; X.fill(); }); q.d.forEach(p => { X.beginPath(); X.arc(AX + p[0], AY + p[1], 8, 0, 7); X.fillStyle = RED; X.fill(); }); X.restore(); }
  f.p.forEach((p, i) => { if (!f.al[i]) return; const x = AX + p[0], y = AY + p[1]; glow(x, y, 26, CYAN, 0.35); X.save(); X.translate(x, y); X.rotate(p[2]); X.beginPath(); X.moveTo(12, 0); X.lineTo(-8, 7); X.lineTo(-8, -7); X.closePath(); X.fillStyle = CYAN; X.fill(); X.restore(); });
  f.d.forEach(p => { const x = AX + p[0], y = AY + p[1]; glow(x, y, 60, RED, 0.45); X.save(); X.translate(x, y); X.rotate(p[2]); X.beginPath(); X.moveTo(22, 0); X.lineTo(-14, 13); X.lineTo(-8, 0); X.lineTo(-14, -13); X.closePath(); X.fillStyle = RED; X.fill(); X.lineWidth = 3; X.strokeStyle = WHITE; X.stroke(); X.restore(); });
  counter(AX + 170, AY + 150, f.c, f.p.length, 'CAUGHT', f.c >= 20 ? RED : CYAN, lt);
};
SC.hunt.events = (b, fi) => { const d = b.data; const step = Math.min(fi * 2, d.frames.length - 1); if (step < 2 || fi * 2 > d.frames.length - 1) return []; const f = d.frames[step], q = d.frames[step - 2]; const ev = []; f.al.forEach((a, i) => { if (!a && q.al[i]) ev.push({ x: AX + q.p[i][0], y: AY + q.p[i][1], col: RED, n: 16, sp: 7, size: 5 }); }); return ev; };

// end card
SC.end = (b, fi, lt) => {
  const k = eOutBack(clamp((lt - 0.2) / 0.5));
  X.save(); X.translate(W / 2, 760); X.scale(Math.max(0, k), Math.max(0, k));
  (b.lines || []).forEach((l, i) => txt(l, 0, -90 + i * 95, i === 2 ? 84 : 76, { fill: i === 2 ? YEL : WHITE, stroke: INK, strokeW: 12 }));
  (b.lines || []).forEach((l, i) => txt(l, 0, -90 + i * 95, i === 2 ? 84 : 76, { fill: i === 2 ? YEL : WHITE }));
  X.restore();
  if (lt > 1.0) { X.save(); X.translate(W / 2, 1120); X.scale(eOutBack((lt - 1.0) / 0.4), eOutBack((lt - 1.0) / 0.4)); glow(0, 0, 300, YEL, 0.3); rr(-280, -55, 560, 110, 55); X.fillStyle = YEL; X.fill(); X.lineWidth = 6; X.strokeStyle = INK; X.stroke(); txt('follow @shipwithpip', 0, 4, 42, { fill: INK }); X.restore(); }
};

// ---------------------------------------------------------------- timeline
window.setData = (data) => { D = data; beats = D.beats; if (D.reel === 'dino') { const C = D.common; const dist = [0]; let s = 0, sp = 7; for (let i = 1; i < 3200; i++) { s += sp; sp = Math.min(11.5, 7 + i * 0.002); dist.push(s); } C.dist = dist; } };
function layout(durs) { DUR = durs; T = 0; beats.forEach((b, i) => { b.t0 = T; b.t1 = T + DUR[i]; T += DUR[i]; }); window.TL = { T, beats: beats.map((b, i) => ({ i, t0: b.t0, t1: b.t1, kind: b.kind })) }; return window.TL; }
window.setDur = layout;
window.frameCount = (i) => { const b = beats[i]; if (b.kind === 'end') return 0; const d = b.data; const r = D.reel; if (r === 'hoops') return Math.ceil(Math.max(...d.shots.map(p => p.length)) / 2); if (r === 'lander') return Math.ceil(Math.max(...d.runs.map(p => p.length)) / 4); if (r === 'snake') return Math.ceil(Math.max(...d.games.map(p => p.length)) / 2); if (r === 'dino') return Math.ceil(Math.max(...d.ys.map(p => p.length)) / 2); if (r === 'hunt') return Math.ceil(d.frames.length / 2); return 0; };

window.setFrame = (t) => {
  t = Math.min(t, T - 0.001);
  const bi = beats.findIndex(q => t >= q.t0 && t < q.t1); const b = beats[bi < 0 ? beats.length - 1 : bi]; const lt = t - b.t0, d = b.t1 - b.t0;
  const fi = Math.max(0, Math.floor((lt - (b.delay || 0.35)) * 30));
  const key = (bi < 0 ? beats.length - 1 : bi) * 1000000 + fi;
  if (key !== lastKey) { if (key !== lastKey + 1) parts = []; if (b.kind !== 'end' && SC[D.reel].events) for (const e of SC[D.reel].events(b, fi)) spawn(e.x, e.y, e.col, e.n, e.sp, 26, e.size || 5); lastKey = key; }
  stepParts();
  X.setTransform(1, 0, 0, 1, 0, 0); bg(t);
  const sh = lt < 0.2 ? (0.2 - lt) * 30 : 0; const r = rnd(Math.floor(t * 60)); X.save(); X.translate((r() - .5) * sh, (r() - .5) * sh);
  if (b.kind === 'end') { SC.end(b, fi, lt); }
  else { SC[D.reel](b, fi, lt); drawParts(); headline(b.head, lt); chips(b, lt); }
  if (b.sfx && lt > b.sfx.t) sfx(b.sfx.text, b.sfx.x || W / 2, b.sfx.y || 1000, b.sfx.size || 120, NC(b.sfx.col || 'YEL'), { pop: (lt - b.sfx.t) / 0.35, rot: b.sfx.rot || -0.1, burst: NC(b.sfx.burst || 'RED') });
  if (b.sfx && lt > b.sfx.t && lt < b.sfx.t + 0.5) speedLines(b.sfx.x || W / 2, b.sfx.y || 1000, 30, 200, 700, 17, WHITE, 4, fade(lt, b.sfx.t, b.sfx.t + 0.05, b.sfx.t + 0.3, b.sfx.t + 0.5));
  X.restore();
  if (lt < 0.15) { X.fillStyle = `rgba(255,255,255,${(0.15 - lt) * 1.6})`; X.fillRect(0, 0, W, H); }
  capDark(b.cap, lt, d); handleDark();
  X.fillStyle = YEL; X.fillRect(0, 0, W * t / T, 10);
};
Promise.all(['700 50px Pop', '44px Hand', '40px Mono', '40px PopM'].map(f => document.fonts.load(f))).then(() => { window.ready = true; });
