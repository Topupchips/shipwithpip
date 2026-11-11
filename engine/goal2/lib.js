// Comic-book reel engine: inked panels, halftone, bubbles, SFX lettering, camera.
const W = 1080, H = 1920;
const cv = document.getElementById('c'), X = cv.getContext('2d');
const INK = '#17142a', PAPER = '#f4ecd8', RED = '#e8342f', YEL = '#ffd23f', BLUE = '#2f6df6', GREEN = '#2fbf71', CYAN = '#4fd1ff';
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const lerp = (a, b, u) => a + (b - a) * u;
const eOutBack = x => { x = clamp(x); const c = 1.70158, d = c + 1; return 1 + d * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2); };
const eOutExpo = x => { x = clamp(x); return x >= 1 ? 1 : 1 - Math.pow(2, -10 * x); };
const eOutCubic = x => { x = clamp(x); return 1 - Math.pow(1 - x, 3); };
const eInOut = x => { x = clamp(x); return x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const eElastic = x => { x = clamp(x); return x === 0 || x === 1 ? x : Math.pow(2, -10 * x) * Math.sin((x * 10 - 0.75) * 2.0944) + 1; };
function rnd(seed) { let s = seed || 1; return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; }; }
const fade = (x, a, b, c, d) => clamp((x - a) / (b - a)) * clamp((d - x) / (d - c));

// ---------------------------------------------------------------- textures
const paperTex = (() => {
  const c = document.createElement('canvas'); c.width = W; c.height = H; const g = c.getContext('2d');
  g.fillStyle = PAPER; g.fillRect(0, 0, W, H);
  const r = rnd(5); const id = g.getImageData(0, 0, W, H); const d = id.data;
  for (let i = 0; i < d.length; i += 4) { const n = (r() - .5) * 18; d[i] += n; d[i + 1] += n; d[i + 2] += n * 0.8; }
  g.putImageData(id, 0, 0);
  for (let i = 0; i < 900; i++) { g.fillStyle = `rgba(90,70,40,${0.02 + r() * 0.05})`; g.beginPath(); g.arc(r() * W, r() * H, 1 + r() * 3, 0, 7); g.fill(); }
  return c;
})();
function dotPattern(rad, col, cell = 18) {
  const c = document.createElement('canvas'); c.width = c.height = cell; const g = c.getContext('2d');
  g.fillStyle = col; g.beginPath(); g.arc(cell / 2, cell / 2, rad, 0, 7); g.fill(); g.beginPath(); g.arc(0, 0, rad, 0, 7); g.fill(); g.beginPath(); g.arc(cell, cell, rad, 0, 7); g.fill(); g.beginPath(); g.arc(cell, 0, rad, 0, 7); g.fill(); g.beginPath(); g.arc(0, cell, rad, 0, 7); g.fill();
  return X.createPattern(c, 'repeat');
}
const HT = [dotPattern(2.2, 'rgba(23,20,42,.55)'), dotPattern(3.6, 'rgba(23,20,42,.55)'), dotPattern(5.2, 'rgba(23,20,42,.6)')];
const HTRED = dotPattern(4, 'rgba(232,52,47,.75)', 16), HTBLUE = dotPattern(3.5, 'rgba(47,109,246,.6)', 16);

// ---------------------------------------------------------------- primitives
function rr(x, y, w, h, r) { X.beginPath(); X.moveTo(x + r, y); X.arcTo(x + w, y, x + w, y + h, r); X.arcTo(x + w, y + h, x, y + h, r); X.arcTo(x, y + h, x, y, r); X.arcTo(x, y, x + w, y, r); X.closePath(); }
function ink(w = 8) { X.lineWidth = w; X.strokeStyle = INK; X.lineJoin = 'round'; X.lineCap = 'round'; X.stroke(); }
function wobblyRect(x, y, w, h, seed = 1, amp = 3) {
  const r = rnd(seed); X.beginPath(); const n = 14;
  const pts = []; for (let i = 0; i <= n; i++) pts.push([x + w * i / n, y + (r() - .5) * amp]);
  for (let i = 1; i <= n; i++) pts.push([x + w + (r() - .5) * amp, y + h * i / n]);
  for (let i = n - 1; i >= 0; i--) pts.push([x + w * i / n, y + h + (r() - .5) * amp]);
  for (let i = n - 1; i >= 1; i--) pts.push([x + (r() - .5) * amp, y + h * i / n]);
  X.moveTo(pts[0][0], pts[0][1]); pts.forEach(p => X.lineTo(p[0], p[1])); X.closePath();
}
// text with outline
function txt(s, x, y, size, opt = {}) {
  X.save(); X.font = `${opt.weight || ''} ${size}px ${opt.font || 'Pop'}`; X.textAlign = opt.align || 'center'; X.textBaseline = opt.base || 'middle';
  if (opt.rot) { X.translate(x, y); X.rotate(opt.rot); x = 0; y = 0; }
  if (opt.shadow) { X.fillStyle = opt.shadow; X.fillText(s, x + (opt.sdx || 10), y + (opt.sdy || 10)); }
  if (opt.stroke) { X.lineJoin = 'round'; X.lineWidth = opt.strokeW || size * 0.18; X.strokeStyle = opt.stroke; X.strokeText(s, x, y); }
  X.fillStyle = opt.fill || INK; X.fillText(s, x, y); X.restore();
}
function wrap(s, maxW, font) { X.save(); X.font = font; const out = []; let line = ''; for (const w of s.split(' ')) { const t = line ? line + ' ' + w : w; if (X.measureText(t).width > maxW && line) { out.push(line); line = w; } else line = t; } if (line) out.push(line); X.restore(); return out; }

// burst (starburst polygon) for SFX
function burst(x, y, r1, r2, n, col, seed = 3, rot = 0) {
  const r = rnd(seed); X.beginPath();
  for (let i = 0; i < n * 2; i++) { const a = rot + i * Math.PI / n; const rad = (i % 2 ? r1 : r2) * (0.85 + r() * 0.3); X.lineTo(x + Math.cos(a) * rad, y + Math.sin(a) * rad); }
  X.closePath(); X.fillStyle = col; X.fill(); ink(9);
}
// speed lines radiating from a point
function speedLines(cx, cy, n, inner, outer, seed, col = INK, lw = 4, alpha = 1) {
  const r = rnd(seed); X.save(); X.globalAlpha = alpha; X.strokeStyle = col; X.lineCap = 'round';
  for (let i = 0; i < n; i++) { const a = r() * 6.283, i0 = inner * (0.7 + r() * 0.6), o = outer * (0.8 + r() * 0.5); X.lineWidth = lw * (0.5 + r()); X.beginPath(); X.moveTo(cx + Math.cos(a) * i0, cy + Math.sin(a) * i0); X.lineTo(cx + Math.cos(a) * o, cy + Math.sin(a) * o); X.stroke(); }
  X.restore();
}
// horizontal motion lines
function motionLines(x, y, w, n, seed, dir = 1) { const r = rnd(seed); X.save(); X.strokeStyle = INK; X.lineCap = 'round'; for (let i = 0; i < n; i++) { const yy = y + (r() - .5) * 120, len = w * (0.4 + r() * 0.6); X.lineWidth = 3 + r() * 4; X.beginPath(); X.moveTo(x, yy); X.lineTo(x + dir * len, yy); X.stroke(); } X.restore(); }

function bubble(text, x, y, w, tailTo, opt = {}) {
  const size = opt.size || 44, font = `${size}px Hand`; const lines = wrap(text, opt.shout ? w * 0.56 : w - 70, font); const lh = size * 1.12; const h = lines.length * lh + 50;
  X.save(); if (opt.pop !== undefined) { const k = eOutBack(opt.pop); X.translate(x + w / 2, y + h / 2); X.scale(k, k); X.translate(-x - w / 2, -y - h / 2); }
  X.beginPath();
  if (opt.thought) { rr(x, y, w, h, 60); }
  else if (opt.shout) { const r = rnd(7); const cx = x + w / 2, cy = y + h / 2; for (let i = 0; i < 28; i++) { const a = i / 28 * 6.283; const rad = (i % 2 ? 1.0 : 1.22) * (0.5 + r() * 0.1); X.lineTo(cx + Math.cos(a) * w * 0.58 * rad * 1.1, cy + Math.sin(a) * h * 0.75 * rad * 1.25); } X.closePath(); }
  else rr(x, y, w, h, 34);
  X.fillStyle = '#fff'; X.fill(); ink(7);
  if (tailTo && !opt.thought) { const bx = clamp(tailTo[0], x + 60, x + w - 60), by = tailTo[1] > y ? y + h : y; X.beginPath(); X.moveTo(bx - 26, by); X.lineTo(tailTo[0], tailTo[1]); X.lineTo(bx + 26, by); X.closePath(); X.fillStyle = '#fff'; X.fill(); X.save(); X.lineWidth = 7; X.strokeStyle = INK; X.beginPath(); X.moveTo(bx - 26, by); X.lineTo(tailTo[0], tailTo[1]); X.lineTo(bx + 26, by); X.stroke(); X.restore(); }
  if (tailTo && opt.thought) { for (let i = 0; i < 3; i++) { const u = (i + 1) / 4; X.beginPath(); X.arc(lerp(x + w / 2, tailTo[0], u), lerp(y + h, tailTo[1], u), 16 - i * 4, 0, 7); X.fillStyle = '#fff'; X.fill(); ink(5); } }
  lines.forEach((l, i) => txt(l, x + w / 2, y + 30 + lh * i + lh / 2 - 4, size, { font: 'Hand', fill: opt.col || INK }));
  X.restore(); return h;
}
function sfx(s, x, y, size, col, opt = {}) {
  const k = opt.pop === undefined ? 1 : eOutBack(opt.pop); if (k <= 0.01) return;
  X.save(); X.translate(x, y); X.rotate(opt.rot || -0.12); X.scale(k, k);
  if (opt.burst) burst(0, 0, size * 1.1, size * 1.6, 12, opt.burst, 9, 0.2);
  txt(s, 0, 0, size, { stroke: INK, strokeW: size * 0.22, fill: col, shadow: 'rgba(23,20,42,.9)', sdx: size * 0.09, sdy: size * 0.09 });
  txt(s, 0, 0, size, { fill: col });
  X.restore();
}

// ---------------------------------------------------------------- character: the student
// moods: tired, shock, rage, dead, stare, aha, relief, smug
function student(cx, cy, s, mood, t, opt = {}) {
  X.save(); X.translate(cx, cy); X.scale(s, s);
  const breathe = Math.sin(t * 2.2) * 3;
  const sh = opt.shake || 0; X.translate((Math.random() - .5) * sh, (Math.random() - .5) * sh);
  // hoodie body
  X.beginPath(); X.moveTo(-150, 330); X.quadraticCurveTo(-170, 150, -110, 120 + breathe); X.quadraticCurveTo(0, 70 + breathe, 110, 120 + breathe); X.quadraticCurveTo(170, 150, 150, 330); X.closePath();
  X.fillStyle = '#3a3ea8'; X.fill(); X.save(); X.clip(); X.fillStyle = HT[1]; X.fillRect(40, 100, 140, 240); X.restore(); ink(9);
  // hood rim
  X.beginPath(); X.moveTo(-90, 125 + breathe); X.quadraticCurveTo(0, 175 + breathe, 90, 125 + breathe); ink(8);
  // strings
  X.beginPath(); X.moveTo(-28, 150 + breathe); X.lineTo(-34, 240 + breathe); X.moveTo(28, 150 + breathe); X.lineTo(34, 240 + breathe); ink(5);
  // neck
  X.fillStyle = '#e8b48a'; rr(-34, 70, 68, 60, 20); X.fill(); ink(7);
  // head
  const bob = Math.sin(t * 2.2 + 1) * 2 + (mood === 'dead' ? 18 : 0);
  X.translate(0, bob);
  const tilt = mood === 'dead' ? 0.14 : mood === 'aha' ? -0.05 : 0; X.rotate(tilt);
  X.beginPath(); X.moveTo(-108, -40); X.quadraticCurveTo(-112, 90, -40, 108); X.quadraticCurveTo(0, 120, 40, 108); X.quadraticCurveTo(112, 90, 108, -40); X.quadraticCurveTo(100, -140, 0, -145); X.quadraticCurveTo(-100, -140, -108, -40); X.closePath();
  X.fillStyle = '#e8b48a'; X.fill(); X.save(); X.clip(); X.fillStyle = HT[0]; X.fillRect(50, -60, 70, 180); X.restore(); ink(9);
  // ears
  for (const sx of [-1, 1]) { X.beginPath(); X.ellipse(sx * 110, 10, 16, 24, 0, 0, 7); X.fillStyle = '#e8b48a'; X.fill(); ink(6); }
  // screen glow on face (blue light from laptop below)
  if (opt.glow) { X.save(); X.globalCompositeOperation = 'screen'; const g = X.createRadialGradient(0, 120, 20, 0, 60, 230); g.addColorStop(0, `rgba(80,160,255,${0.55 * opt.glow})`); g.addColorStop(1, 'rgba(80,160,255,0)'); X.fillStyle = g; X.beginPath(); X.ellipse(0, 10, 112, 130, 0, 0, 7); X.fill(); X.restore(); }
  // hair: messy spikes
  X.beginPath(); X.moveTo(-112, -40); X.quadraticCurveTo(-120, -120, -70, -150); X.lineTo(-85, -195); X.lineTo(-40, -160); X.lineTo(-30, -215); X.lineTo(5, -165); X.lineTo(35, -210); X.lineTo(50, -160); X.lineTo(95, -190); X.lineTo(75, -140); X.quadraticCurveTo(118, -115, 112, -40); X.quadraticCurveTo(80, -70, 60, -95); X.quadraticCurveTo(20, -75, -20, -100); X.quadraticCurveTo(-60, -70, -112, -40); X.closePath();
  X.fillStyle = '#1d1a2a'; X.fill(); ink(9);
  X.beginPath(); X.moveTo(-60, -160); X.quadraticCurveTo(-20, -175, 20, -165); X.strokeStyle = '#4a4466'; X.lineWidth = 7; X.lineCap = 'round'; X.stroke();
  // eyes
  const bl = opt.blink ? 0.08 : 1;
  const E = { tired: { open: 0.42, pup: 1, brow: 0.05, browA: 0.1, bags: 1 }, shock: { open: 1.35, pup: 0.55, brow: -0.5, browA: 0, bags: 0.6 }, rage: { open: 0.8, pup: 0.7, brow: 0.3, browA: 0.55, bags: 0.4 }, dead: { open: 0.25, pup: 0.9, brow: 0.1, browA: 0.05, bags: 1.2 }, stare: { open: 0.7, pup: 0.75, brow: 0.0, browA: 0.25, bags: 1 }, aha: { open: 1.2, pup: 0.7, brow: -0.6, browA: -0.1, bags: 0.6 }, relief: { open: 0.55, pup: 0.9, brow: -0.25, browA: -0.15, bags: 0.5 }, smug: { open: 0.6, pup: 0.8, brow: -0.1, browA: 0.3, bags: 0.3 } }[mood] || {};
  const look = opt.look || [0, 0];
  for (const sx of [-1, 1]) {
    const ex = sx * 48, ey = -10;
    if (E.bags) { X.save(); X.globalAlpha = 0.35 * E.bags; X.strokeStyle = '#7a4a8a'; X.lineWidth = 6; X.beginPath(); X.arc(ex, ey + 14, 30, 0.3, 2.84); X.stroke(); X.restore(); }
    X.beginPath(); X.ellipse(ex, ey, 34, 34 * E.open * bl, 0, 0, 7); X.fillStyle = '#fff'; X.fill(); ink(7);
    if (bl > 0.2) { X.save(); X.beginPath(); X.ellipse(ex, ey, 34, 34 * E.open, 0, 0, 7); X.clip(); X.beginPath(); X.arc(ex + look[0] * 14 + 4, ey + look[1] * 12 + 4, 17 * E.pup, 0, 7); X.fillStyle = INK; X.fill(); X.beginPath(); X.arc(ex + look[0] * 14 - 3, ey + look[1] * 12 - 4, 6, 0, 7); X.fillStyle = '#fff'; X.fill(); X.restore(); }
    // brow
    const by = ey - 48 + E.brow * 40; X.beginPath(); X.moveTo(ex - 36 * sx, by + E.browA * 22 * 1); X.quadraticCurveTo(ex, by - 10, ex + 36 * sx, by - E.browA * 22); X.strokeStyle = '#1d1a2a'; X.lineWidth = 11; X.lineCap = 'round'; X.stroke();
  }
  // nose
  X.beginPath(); X.moveTo(6, 10); X.quadraticCurveTo(14, 40, -2, 44); ink(6);
  // mouth
  X.beginPath();
  if (mood === 'shock') { X.ellipse(0, 76, 22, 34, 0, 0, 7); X.fillStyle = '#5a1020'; X.fill(); ink(7); X.beginPath(); X.ellipse(0, 62, 18, 10, 0, 0, 7); X.fillStyle = '#fff'; X.fill(); }
  else if (mood === 'rage') { rr(-36, 62, 72, 30, 8); X.fillStyle = '#5a1020'; X.fill(); ink(7); X.save(); X.strokeStyle = INK; X.lineWidth = 4; for (let i = -24; i <= 24; i += 12) { X.beginPath(); X.moveTo(i, 62); X.lineTo(i, 92); X.stroke(); } X.restore(); X.fillStyle = '#fff'; X.fillRect(-32, 64, 64, 10); }
  else if (mood === 'dead') { X.moveTo(-30, 80); X.lineTo(30, 78); ink(7); }
  else if (mood === 'aha') { X.ellipse(0, 78, 26, 20, 0, 0, 7); X.fillStyle = '#5a1020'; X.fill(); ink(7); }
  else if (mood === 'relief' || mood === 'smug') { X.moveTo(-34, 70); X.quadraticCurveTo(0, 100, 34, 68); ink(7); }
  else { X.moveTo(-26, 82); X.quadraticCurveTo(0, 72, 26, 82); ink(7); }
  // cheek lines (tired)
  if (mood === 'tired' || mood === 'dead') { X.save(); X.globalAlpha = 0.5; X.beginPath(); X.moveTo(-70, 40); X.lineTo(-60, 60); X.moveTo(70, 40); X.lineTo(60, 60); ink(4); X.restore(); }
  X.restore();
}
function laptop(cx, cy, s, t, opt = {}) {
  X.save(); X.translate(cx, cy); X.scale(s, s);
  // base
  X.beginPath(); X.moveTo(-230, 0); X.lineTo(230, 0); X.lineTo(260, 40); X.lineTo(-260, 40); X.closePath(); X.fillStyle = '#5a5f7a'; X.fill(); ink(8);
  // lid (seen from behind: dark)
  rr(-210, -300, 420, 300, 18); X.fillStyle = '#2b2d44'; X.fill(); ink(8);
  if (opt.back) { X.beginPath(); X.arc(0, -150, 22, 0, 7); X.fillStyle = CYAN; X.fill(); ink(5); }
  else { rr(-192, -284, 384, 268, 10); X.fillStyle = opt.screen || '#0e1326'; X.fill(); if (opt.draw) { X.save(); X.beginPath(); rr(-192, -284, 384, 268, 10); X.clip(); opt.draw(t); X.restore(); } }
  X.restore();
}
function clockCard(x, y, s, label, opt = {}) {
  X.save(); X.translate(x, y); X.scale(s, s); if (opt.rot) X.rotate(opt.rot);
  rr(-190, -70, 380, 140, 22); X.fillStyle = INK; X.fill(); ink(8);
  txt(label, 0, 2, 92, { font: 'Mono', fill: opt.col || RED });
  X.restore();
}
function panel(x, y, w, h, opt, draw) {
  const p = opt.pop === undefined ? 1 : opt.pop; const k = p >= 1 ? 1 : lerp(1.3, 1, eOutBack(p)); const rot = p >= 1 ? 0 : (1 - eOutCubic(p)) * (opt.rot || -0.06);
  if (p <= 0) return;
  X.save(); X.translate(x + w / 2, y + h / 2); X.rotate(rot); X.scale(k, k); X.translate(-w / 2, -h / 2);
  // drop shadow
  X.save(); X.translate(14, 18); wobblyRect(0, 0, w, h, opt.seed || 1, 4); X.fillStyle = 'rgba(23,20,42,.9)'; X.fill(); X.restore();
  wobblyRect(0, 0, w, h, opt.seed || 1, 4); X.fillStyle = opt.bg || '#fff'; X.fill();
  X.save(); X.clip(); draw(w, h); X.restore();
  wobblyRect(0, 0, w, h, opt.seed || 1, 4); ink(11);
  if (p < 0.25) { X.fillStyle = `rgba(255,255,255,${(0.25 - p) * 3})`; X.fillRect(-20, -20, w + 40, h + 40); }
  X.restore();
}
function caption(text, lt, dur, opt = {}) {
  if (!text) return;
  const size = 50, font = `${size}px Pop`; const lines = wrap(text, 900, font); const lh = 62; const h = lines.length * lh + 44; const y = (opt.y || 1560) - h / 2;
  const pop = clamp(lt / 0.3);
  X.save(); X.translate(W / 2, y + h / 2); X.scale(1, eOutBack(pop)); X.translate(-W / 2, -y - h / 2);
  X.save(); X.translate(8, 10); rr(70, y, 940, h, 20); X.fillStyle = INK; X.fill(); X.restore();
  rr(70, y, 940, h, 20); X.fillStyle = '#fff'; X.fill(); ink(8);
  const words = text.split(' '); const total = words.length; const prog = clamp((lt - 0.1) / Math.max(0.5, dur - 0.5)) * total;
  let wi = 0;
  lines.forEach((l, i) => { const ws = l.split(' '); X.font = font; const widths = ws.map(w => X.measureText(w + ' ').width); const tw = widths.reduce((a, b) => a + b, 0) - X.measureText(' ').width; let x = W / 2 - tw / 2; ws.forEach((w, j) => { const hi = wi < prog; txt(w, x, y + 22 + lh * i + lh / 2, size, { align: 'left', fill: hi ? (opt.hi || RED) : INK }); x += widths[j]; wi++; }); });
  X.restore();
}
function code(lines, x, y, size, opt = {}) {
  X.save(); X.font = `${size}px Mono`; X.textAlign = 'left'; X.textBaseline = 'middle';
  lines.forEach((l, i) => { const yy = y + i * size * 1.5; if (opt.num) { X.fillStyle = '#6a7090'; X.fillText(String(opt.num + i).padStart(2, ' '), x, yy); } let xx = x + (opt.num ? size * 2.2 : 0); for (const seg of l) { X.fillStyle = seg[1] || '#dfe6ff'; X.fillText(seg[0], xx, yy); xx += X.measureText(seg[0]).width; } });
  X.restore();
}
function vignette(a = 0.35) { const g = X.createRadialGradient(W / 2, H / 2, 500, W / 2, H / 2, 1300); g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, `rgba(10,5,20,${a})`); X.fillStyle = g; X.fillRect(0, 0, W, H); }
function handle() { txt('@shipwithpip', W / 2, 1840, 34, { font: 'PopM', fill: 'rgba(23,20,42,.75)' }); }

