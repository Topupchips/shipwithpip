"""'What if' reels: an escalating chain of hypotheticals, every number computed from sourced inputs.

Format (Ansh, Nov 16 2026, after a viral 'what if you swung a hammer at light speed' reel): a 2D stylized engine,
big bold visuals, word-by-word captions, a red stamp on the payoff numbers, a reality-check ending.

usage:
  python3 whatif.py stats  script.json                 prints inputs + every computed value (the only numbers a script may use)
  python3 whatif.py check  script.json                 every number in hook/head/chip/stamp/say/speak/labels/caption must be in stats
  python3 whatif.py sheet  script.json out.png --model DIR
  python3 whatif.py render script.json out.mp4 --model DIR

Script keys: id, title, caption, hashtags, sources (URLs, each opened in the run that wrote it),
  "inputs": {name: number}  sourced numbers only (each listed in "notes" with its source),
  "calc": [[name, "python expression over earlier names and math"], ...]  every derived number, evaluated here,
  "beats": [{"say", "speak"?, "key": [...], "hook": [3 lines] (first beat) | "head": [2 lines], "chip"?, "stamp"?: {"text", "at"},
             "vis": {"scene": stop|equator|bars|lat|daynight|slow|float|hammer|rain|jump|elevator|tunnel|bolt, ...}}],
  last beat {"say": "Follow Ship With Pip for more.", "nocap": true}.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import minimal as M  # noqa: E402
from minimal import W, H, FPS, SR, font, al, tc, clamp, ease, F_BOLD, F_SEMI, F_MED  # noqa: E402

TH = dict(M.THEMES["dark"]); TH["bg"] = (7, 9, 18); TH["grid"] = (15, 18, 32)
YEL = (255, 214, 64); RED = (255, 64, 72); CYAN = (90, 220, 255); LIME = (182, 255, 59); WHITE = (255, 255, 255)
COLS = dict(YEL=YEL, RED=RED, CYAN=CYAN, LIME=LIME, WHITE=WHITE, MUTED=(150, 156, 180))
CAP_Y = 1560
GX, GY, GR = 540, 880, 330  # globe placement

_LAND = None


def land():
    global _LAND
    if _LAND is None:
        _LAND = np.unpackbits(np.load(os.path.join(HERE, "assets", "landmask_1024x512.npy")))[:1024 * 512].reshape(512, 1024).astype(bool)
    return _LAND


# ============================================================ globe
_grid = {}


def globe(img, cx, cy, r, rot, tilt=18.0, light=(-0.55, -0.45, 0.70), night=0.22, eq=0.0, lats=(), glow=1.0):
    """Orthographic Earth with real coastlines. rot: degrees of longitude at the centre. light: unit-ish vector (x right, y down, z to viewer)."""
    r = int(r)
    if r not in _grid:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
        m = xx ** 2 + yy ** 2 <= r * r
        z = np.sqrt(np.clip(r * r - xx ** 2 - yy ** 2, 0, None))
        _grid[r] = (xx / r, yy / r, z / r, m)
    x, y, z, m = _grid[r]
    tl = math.radians(tilt)
    # rotate view about the x axis by tilt (north pole toward viewer when tilt > 0)
    y2 = y * math.cos(tl) - z * math.sin(tl); z2 = y * math.sin(tl) + z * math.cos(tl)
    lat = np.degrees(np.arcsin(np.clip(-y2, -1, 1)))
    lon = (np.degrees(np.arctan2(x, z2)) + rot + 180) % 360 - 180
    L = land(); li = np.clip(((90 - lat) / 180 * 512).astype(int), 0, 511); lj = np.clip(((lon + 180) / 360 * 1024).astype(int), 0, 1023)
    isl = L[li, lj]
    ocean = np.array([22, 70, 150], np.float32); ocean2 = np.array([12, 40, 100], np.float32)
    landc = np.array([70, 170, 95], np.float32); ice = np.array([225, 235, 245], np.float32)
    base = ocean2 + (ocean - ocean2) * z[..., None]
    col = np.where(isl[..., None], landc, base)
    col = np.where((np.abs(lat) > 72)[..., None] & isl[..., None] | (lat < -66)[..., None], ice, col)
    lv = np.array(light, np.float32); lv = lv / np.linalg.norm(lv)
    lam = np.clip(x * lv[0] + y * lv[1] + z * lv[2], 0, 1)
    shade = night + (1 - night) * (lam ** 0.8)
    col = col * shade[..., None]
    # rim light
    rim = (1 - z) ** 3
    col = col + np.array([80, 160, 255], np.float32) * rim[..., None] * 0.55 * glow
    def ring(lat0, c, a, wdt=0.012):
        band = np.abs(lat - lat0) < (wdt * 90) if lat0 != 90 else lat > 88.5
        band &= z2 > -0.05
        return np.where(band[..., None], col * (1 - a) + np.array(c, np.float32) * a, col)
    if eq > 0: col = ring(0, YEL, 0.9 * eq, 0.016)
    for la, c, a in lats: col = ring(la, c, a)
    rgba = np.zeros(m.shape + (4,), np.uint8); rgba[..., :3] = col.clip(0, 255); rgba[..., 3] = (m * 255).astype(np.uint8)
    g = Image.fromarray(rgba, "RGBA")
    # atmosphere halo
    halo = Image.new("RGBA", (2 * r + 120, 2 * r + 120), (0, 0, 0, 0)); hd = ImageDraw.Draw(halo)
    hd.ellipse([40, 40, 2 * r + 80, 2 * r + 80], fill=(70, 150, 255, int(110 * glow))); halo = halo.filter(ImageFilter.GaussianBlur(26))
    img.alpha_composite(halo, (int(cx - r - 60), int(cy - r - 60)))
    img.alpha_composite(g, (int(cx - r), int(cy - r)))


def surface_xy(cx, cy, r, lat, lon_rel, tilt=18.0):
    """Screen position of a surface point (lon relative to the view centre). Returns (x, y, z)."""
    la, lo, tl = math.radians(lat), math.radians(lon_rel), math.radians(tilt)
    x = math.cos(la) * math.sin(lo); yv = -math.sin(la); z = math.cos(la) * math.cos(lo)
    y2 = yv * math.cos(tl) + z * math.sin(tl); z2 = -yv * math.sin(tl) + z * math.cos(tl)
    return cx + x * r, cy + y2 * r, z2


def person(d, x, y, s, ang, col=WHITE, a=1.0):
    """Stick figure with feet at (x, y), standing along angle ang (radians, 0 = up)."""
    def P(u, v):  # u along body (up), v sideways
        return (x + math.sin(ang) * u * s + math.cos(ang) * v * s, y - math.cos(ang) * u * s + math.sin(ang) * v * s)
    c = al(col, a); w = max(3, int(s * 0.16))
    d.line([P(0, -0.35), P(1.0, 0), P(0, 0.35)], fill=c, width=w, joint="curve")
    d.line([P(1.0, 0), P(1.9, 0)], fill=c, width=w)
    d.line([P(1.65, -0.55), P(1.55, 0), P(1.65, 0.55)], fill=c, width=w, joint="curve")
    hx, hy = P(2.3, 0); hr = s * 0.38
    d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=c)


def bigtext(img, xy, text, size, col, angle=0.0, a=1.0, scale=1.0, outline=None):
    f = font(F_BOLD, max(8, int(size * scale)))
    tmp = Image.new("RGBA", (int(len(text) * size * scale * 0.75) + 80, int(size * scale * 1.5) + 40), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    if outline: td.text((tmp.width / 2, tmp.height / 2), text, font=f, fill=al(outline, a), anchor="mm", stroke_width=max(2, int(size * scale / 14)), stroke_fill=al(outline, a))
    td.text((tmp.width / 2, tmp.height / 2), text, font=f, fill=al(col, a), anchor="mm")
    if angle: tmp = tmp.rotate(angle, resample=Image.BICUBIC, expand=True)
    img.alpha_composite(tmp, (int(xy[0] - tmp.width / 2), int(xy[1] - tmp.height / 2)))


def fmt_num(v, dec=0):
    return f"{v:,.{dec}f}"


# ============================================================ scenes  (each: fn(img, d, vis, p, t, ctx), p = beat progress 0..1)
def spin_rot(t, rate=26.0): return (-t * rate) % 360 - 40  # Earth turns west to east: the front surface moves right


def sc_stop(img, d, v, p, t, ctx):
    stop = v.get("stop_at", 0.30); ps = clamp(p / stop)
    # spinning, decelerating to a dead stop at `stop` (rotation frozen afterwards)
    rot = 60 - 120 * (1 - (1 - ps) ** 2)
    shake = 0
    if p > stop: shake = 14 * math.exp(-(p - stop) * 18) * math.sin(p * 400)
    globe(img, GX + shake, GY, GR, rot)
    d = ImageDraw.Draw(img, "RGBA")
    # person on the right limb at the equator, then launched to the right
    x0, y0, _ = surface_xy(GX + shake, GY, GR, 0, 89.5)
    if p <= stop:
        person(d, x0, y0, 46, math.pi / 2, WHITE)
        tc(d, (x0 + 60, y0 - 110), "YOU", font(F_BOLD, 52), al(YEL, 1))
    else:
        q = (p - stop) / (1 - stop); dx = 900 * ease(min(q * 1.6, 1)); dy = -180 * q
        for k in range(14):
            tr = max(0, dx - k * 40)
            d.line([(x0 + tr, y0 + dy * tr / max(dx, 1)), (x0 + tr - 40, y0 + dy * (tr - 40) / max(dx, 1))], fill=al(YEL, 0.5 * (1 - k / 14)), width=6)
        person(d, x0 + dx, y0 + dy, 46, math.pi / 2 + q * 9, WHITE)
        if q < 0.25: d.ellipse([GX - GR - 60, GY - GR - 60, GX + GR + 60, GY + GR + 60], outline=al(WHITE, 0.6 * (1 - q * 4)), width=int(30 * (1 - q * 4)) + 1)
    # speed counter
    val = v["value"]; g = clamp((p - stop) / 0.25) if p > stop else 0
    if p > stop:
        bigtext(img, (540, 1330), fmt_num(val) + " " + v.get("unit", ""), 120, YEL, a=g, scale=1 + 0.5 * (1 - ease(g)), outline=(40, 30, 0))  # no tweened numbers: only the real value is ever shown


def sc_equator(img, d, v, p, t, ctx):
    globe(img, GX, GY, GR, spin_rot(t), eq=clamp(p / 0.15))
    d = ImageDraw.Draw(img, "RGBA")
    # a glowing dot running along the front of the equator
    for k in range(6):
        lo = ((t * 60 - k * 6) % 180) - 90
        x, y, z = surface_xy(GX, GY, GR, 0, lo)
        rr = 14 - k * 2
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=al(WHITE, 0.9 - k * 0.14))
    labs = v.get("labels", []); n = len(labs)
    for i, lb in enumerate(labs):
        a = clamp((p - (0.12 + i * 0.7 / max(n, 1))) / 0.08)
        if a > 0: bigtext(img, (540, 1262 + i * 84), lb, 72 if i < n - 1 else 88, YEL if i == n - 1 else WHITE, a=a)


def sc_bars(img, d, v, p, t, ctx):
    rows = v["rows"]; mx = max(r["value"] for r in rows) * 1.05
    y0 = 600; gap = min(260, 800 / len(rows))
    globe(img, 540, 1310, 120, spin_rot(t) if not v.get("still") else 20, glow=0.6) if v.get("globe") else None
    d = ImageDraw.Draw(img, "RGBA")
    for i, r in enumerate(rows):
        st = 0.08 + i * 0.22; g = ease(clamp((p - st) / 0.28))
        if p < st: continue
        y = y0 + i * gap; c = COLS.get(r.get("col", "WHITE"), WHITE)
        tc(d, (90, y), r["label"], font(F_BOLD, 54), al(WHITE, clamp((p - st) / 0.06)), anchor="lm")
        wfull = 900 * r["value"] / mx
        d.rounded_rectangle([90, y + 44, 90 + max(wfull * g, 24), y + 124], radius=26, fill=al(c, 0.92))
        txt = fmt_num(r["value"], r.get("dec", 0)) + " " + r.get("unit", "") if g > 0.98 else ""  # value appears once the bar is full
        tw = d.textlength(txt, font=font(F_BOLD, 56))
        tx = 90 + wfull * g + 20
        if tx + tw > 1010: tc(d, (90 + wfull * g - 20, y + 84), txt, font(F_BOLD, 56), (12, 12, 20, 255), anchor="rm")
        else: tc(d, (tx, y + 84), txt, font(F_BOLD, 56), al(WHITE, 1), anchor="lm")


def sc_lat(img, d, v, p, t, ctx):
    tilt = v.get("tilt", 32); rows = v["rows"]; n = len(rows)
    shown = [i for i in range(n) if p > 0.08 + i * 0.75 / n]
    lats = [(r["lat"], COLS.get(r.get("col", "YEL"), YEL), 0.95) for i, r in enumerate(rows) if i in shown]
    rot = spin_rot(t) if not v.get("still") else 10
    globe(img, GX, GY, GR, rot, tilt=tilt, lats=lats)
    d = ImageDraw.Draw(img, "RGBA")
    for i in shown:
        r = rows[i]; c = COLS.get(r.get("col", "YEL"), YEL); a = clamp((p - (0.08 + i * 0.75 / n)) / 0.08)
        # moving dots on the ring: speed proportional to cos(lat)
        if r["lat"] < 89:
            for k in range(5):
                lo = ((t * 70 * math.cos(math.radians(r["lat"])) + k * 36) % 180) - 90
                x, y, z = surface_xy(GX, GY, GR, r["lat"], lo, tilt)
                if z > 0: d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=al(WHITE, 0.9 * a))
        else:
            x, y, _ = surface_xy(GX, GY, GR, 90, 0, tilt)
            person(d, x, y, 26, 0, WHITE, a)
        tag = r["text"]; ty = 1262 + i * 72
        tc(d, (540, ty), tag, font(F_BOLD, 60), al(c, a))


def sc_daynight(img, d, v, p, t, ctx):
    # the sun appears to crawl across the sky: the lit side sweeps slowly over the planet
    ang = math.radians(-70 + 140 * p)
    lv = (math.sin(ang) * -1, -0.1, math.cos(ang) * 0.6)
    sx = 540 - 430 * math.sin(ang); sy = 430
    sun = Image.new("RGBA", (420, 420), (0, 0, 0, 0)); sd = ImageDraw.Draw(sun)
    sd.ellipse([60, 60, 360, 360], fill=(255, 200, 60, 140)); sun = sun.filter(ImageFilter.GaussianBlur(40))
    img.alpha_composite(sun, (int(sx - 210), int(sy - 210)))
    d0 = ImageDraw.Draw(img, "RGBA"); d0.ellipse([sx - 70, sy - 70, sx + 70, sy + 70], fill=(255, 226, 120, 255))
    globe(img, GX, GY + 60, GR - 30, 20, light=lv, night=0.08)
    d = ImageDraw.Draw(img, "RGBA")
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - 0.2 - i * 0.3) / 0.1)
        if a > 0: bigtext(img, (540, 1330 + i * 90), lb, 76, YEL if i == 0 else WHITE, a=a)


def sc_slow(img, d, v, p, t, ctx):
    globe(img, GX, GY, GR, (-t * 6) % 360)
    d = ImageDraw.Draw(img, "RGBA")
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - 0.15 - i * 0.25) / 0.1)
        if a > 0: bigtext(img, (540, 1320 + i * 90), lb, 76, WHITE if i else YEL, a=a)


def sc_float(img, d, v, p, t, ctx):
    """Globe spins faster and faster; the person on the equator limb lifts off (float) or is flung away (fling)."""
    fling = v.get("mode") == "fling"
    ctx_rate = 26 + (260 if fling else 170) * ease(clamp(p / 0.5))
    rot = (-26 * t - (900 if fling else 600) * p * p) % 360 - 40
    globe(img, GX, GY, GR, rot)
    d = ImageDraw.Draw(img, "RGBA")
    # speed streaks along the equator front
    for k in range(10):
        lo = ((t * ctx_rate * 0.9 - k * 18) % 180) - 90
        x, y, z = surface_xy(GX, GY, GR, 0, lo)
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=al(WHITE, 0.55))
    x0, y0, _ = surface_xy(GX, GY, GR, 0, 89.5)
    lift = clamp((p - 0.4) / 0.6)
    if fling:
        q = ease(clamp((p - 0.45) / 0.4)); dx = 150 * q; dy = -430 * q
        for k in range(12):
            tr = max(0.0, q - k * 0.04)
            d.line([(x0 + 150 * tr, y0 - 430 * tr), (x0 + 150 * max(0, tr - 0.04), y0 - 430 * max(0, tr - 0.04))], fill=al(YEL, 0.5 * (1 - k / 12)), width=6)
        person(d, x0 + dx, y0 + dy, 46 * (1 - 0.5 * q), math.pi / 2 + q * 7, WHITE)
    else:
        dx = 70 * ease(lift) + 10 * math.sin(t * 2.2) * lift; dy = 14 * math.sin(t * 1.7) * lift
        person(d, x0 + dx, y0 + dy, 46, math.pi / 2 + 0.35 * math.sin(t * 1.3) * lift, WHITE)
        if lift > 0: tc(d, (x0 - 30, y0 - 120), "YOU", font(F_BOLD, 52), al(YEL, 1))
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - 0.5 - i * 0.2) / 0.1)
        if a > 0: bigtext(img, (540, 1300 + i * 90), lb, 80, YEL if i == 0 else WHITE, a=a)


SCENES = dict(stop=sc_stop, equator=sc_equator, bars=sc_bars, lat=sc_lat, daynight=sc_daynight, slow=sc_slow, float=sc_float)



# ============================================================ hammer chain (Nov 18 2026, Ansh's nail reel model: one object, one change per beat)
GROUND = 1250; NAIL_X = 650; PIV = (300, 905); HREACH = 392  # shoulder pivot, pivot to hammer face
BROWN = (150, 96, 52); BROWN_D = (96, 58, 30); STEEL = (190, 200, 215); ORANGE = (255, 140, 40)
_TOWN = None


def town_houses():
    global _TOWN
    if _TOWN is None:
        rs = np.random.RandomState(7); hs = []; x = -20
        while x < 1100:
            w = int(rs.randint(60, 120)); h = int(rs.randint(70, 190)); hs.append((x, w, h, int(rs.randint(0, 3)))); x += w + int(rs.randint(4, 16))
        _TOWN = hs
    return _TOWN


def draw_town(d, wave_x=None, flat=0.0, glow=0.0):
    """Skyline on the horizon. Houses the shockwave (|x - NAIL_X| < wave_x) has passed lean and collapse; flat=1 flattens everything."""
    for x, w, h, k in town_houses():
        cx = x + w / 2; hit = 0.0
        if wave_x is not None and abs(cx - NAIL_X) < wave_x: hit = clamp((wave_x - abs(cx - NAIL_X)) / 160)
        hit = max(hit, flat); hh = h * (1 - 0.9 * hit); lean = (1 if cx > NAIL_X else -1) * 60 * hit
        base = (34, 40, 66) if k else (44, 50, 80); c = tuple(int(base[i] + (ORANGE[i] - base[i]) * 0.35 * glow) for i in range(3))
        top = GROUND - 60 - hh
        d.polygon([(x, GROUND - 60), (x + w, GROUND - 60), (x + w + lean, top), (x + lean, top)], fill=c + (255,))
        if hit < 0.5 and k != 1: d.polygon([(x + lean - 6, top), (x + w + lean + 6, top), (x + w / 2 + lean, top - 34)], fill=c + (255,))
        if hit < 0.3:
            for wy in range(int(top) + 22, GROUND - 80, 42):
                for wx in range(int(x) + 14, int(x + w) - 20, 30): d.rectangle([wx + lean * 0.3, wy, wx + 12 + lean * 0.3, wy + 16], fill=al(YEL, 0.35))
        if hit > 0.5:  # rubble
            rs = np.random.RandomState(x + 1000)
            for _ in range(5):
                rx = x + rs.rand() * w + lean; rw = 10 + rs.rand() * 22
                d.rectangle([rx, GROUND - 60 - rw * 0.6, rx + rw, GROUND - 60], fill=c + (255,))
    d.rectangle([0, GROUND - 60, W, GROUND - 52], fill=(28, 32, 52, 255))


def draw_hammer(d, ang, glow=0.0, blur=0.0, a=1.0):
    """ang: radians from straight down-right toward the nail (0 = striking the nail head)."""
    tx, ty = NAIL_X - PIV[0], (GROUND - 70 - 96) - PIV[1]; base = math.atan2(ty, tx); th = base - ang
    ux, uy = math.cos(th), math.sin(th); hand = (PIV[0] + ux * 170, PIV[1] + uy * 170); face = (PIV[0] + ux * HREACH, PIV[1] + uy * HREACH)
    if blur > 0:  # motion arc: earlier hammer positions, fading
        for k in range(1, 9):
            t2 = th - k * blur * 0.09; px, py = PIV[0] + math.cos(t2) * HREACH, PIV[1] + math.sin(t2) * HREACH
            d.ellipse([px - 14, py - 14, px + 14, py + 14], fill=al(ORANGE if glow > 0.3 else WHITE, 0.30 * (1 - k / 9) * a))
    d.line([PIV, hand], fill=al(WHITE, a), width=20)  # arm
    hc = tuple(int(BROWN[i] + (ORANGE[i] - BROWN[i]) * glow) for i in range(3))
    d.line([hand, (PIV[0] + ux * (HREACH - 10), PIV[1] + uy * (HREACH - 10))], fill=hc + (int(255 * a),), width=18)  # handle
    px, py = -uy, ux; hx, hy = face[0] - ux * 22, face[1] - uy * 22
    hcol = tuple(int(STEEL[i] + (ORANGE[i] + 0 - STEEL[i]) * glow) for i in range(3))
    poly = [(hx + px * 62 + ux * 22, hy + py * 62 + uy * 22), (hx + px * 62 - ux * 22, hy + py * 62 - uy * 22),
            (hx - px * 70 - ux * 14, hy - py * 70 - uy * 14), (hx - px * 70 + ux * 14, hy - py * 70 + uy * 14)]
    if glow > 0:
        g = max(1, int(34 * glow)); d.ellipse([face[0] - g, face[1] - g, face[0] + g, face[1] + g], fill=al(ORANGE, 0.35 * glow * a))
    d.polygon(poly, fill=hcol + (int(255 * a),))
    d.ellipse([hand[0] - 16, hand[1] - 16, hand[0] + 16, hand[1] + 16], fill=al(WHITE, a))


def draw_worker(d, a=1.0):
    x, y = PIV[0] - 40, GROUND - 60; c = al(WHITE, a)
    d.line([(x - 50, y), (x, y - 160), (x + 40, y)], fill=c, width=20, joint="curve")  # legs
    d.line([(x, y - 160), (PIV[0], PIV[1])], fill=c, width=22)  # torso to shoulder
    hx, hy = PIV[0] - 6, PIV[1] - 70; d.ellipse([hx - 44, hy - 44, hx + 44, hy + 44], fill=c)
    d.chord([hx - 52, hy - 70, hx + 52, hy + 20], 180, 360, fill=al(YEL, a)); d.rectangle([hx - 62, hy - 26, hx + 62, hy - 18], fill=al(YEL, a))  # hard hat


def draw_plank(d, sink, crack=0.0, gone=0.0):
    if gone < 1:
        a = 1 - gone
        if crack <= 0:
            d.rounded_rectangle([420, GROUND - 120, 900, GROUND - 60], radius=6, fill=al(BROWN, a)); d.line([(430, GROUND - 98), (890, GROUND - 98)], fill=al(BROWN_D, 0.6 * a), width=3)
        else:  # split in two at the nail, halves tilting apart
            o = 40 * crack
            d.polygon([(420 - o, GROUND - 120 + o * 0.5), (NAIL_X - 6 - o * 0.4, GROUND - 120 - o * 0.6), (NAIL_X - 18 - o * 0.4, GROUND - 60), (420 - o, GROUND - 60)], fill=al(BROWN, a))
            d.polygon([(NAIL_X + 6 + o * 0.4, GROUND - 120 - o * 0.6), (900 + o, GROUND - 120 + o * 0.5), (900 + o, GROUND - 60), (NAIL_X + 18 + o * 0.4, GROUND - 60)], fill=al(BROWN, a))
    top = GROUND - 120 - 96 * (1 - sink)
    d.line([(NAIL_X, top), (NAIL_X, top + 96)], fill=al(STEEL, 1), width=8)
    d.rectangle([NAIL_X - 18, top - 6, NAIL_X + 18, top + 2], fill=al(STEEL, 1))


def sc_hammer(img, d, v, p, t, ctx):
    mode = v.get("mode", "sink"); hit = v.get("hit", 0.38); before = p < hit; q = clamp((p - hit) / (1 - hit)) if not before else 0
    big = mode in ("flash", "mega", "after")
    # sky glow for the big ones
    if big and not before or mode == "after":
        sky = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sd = ImageDraw.Draw(sky)
        sd.ellipse([NAIL_X - 900, GROUND - 900, NAIL_X + 900, GROUND + 700], fill=al(ORANGE, 0.35 if mode != "after" else 0.15)); img.alpha_composite(sky.filter(ImageFilter.GaussianBlur(120)))
        d = ImageDraw.Draw(img, "RGBA")
    if big or v.get("town"):
        wave = None if before or mode == "after" else 1400 * ease(clamp((q - 0.08) / 0.7))
        draw_town(d, wave, flat=1.0 if mode == "after" else 0.0, glow=0.0 if before else 1 - q * 0.5)
    d.rectangle([0, GROUND - 60, W, H], fill=(16, 18, 32, 255)); d.line([(0, GROUND - 60), (W, GROUND - 60)], fill=al(LIME, 0.5), width=4)
    # swing angle: wind up, then strike at `hit`
    wind = math.radians(150)
    if mode == "frozen":  # the hammer creeps toward the nail and never arrives
        ang = math.radians(3 + 40 * math.exp(-p * 3.5))
    elif before:
        u = p / hit; ang = wind * (math.sin(u * math.pi / 0.75) if u < 0.75 * 0.5 else max(0, (1 - (u - 0.375) / 0.625)) ** 2)
    else:
        ang = math.radians(2) + (math.radians(25) * ease(clamp(q * 3)) if mode == "sink" else 0)
    fast = mode in ("crack", "fire", "flash", "mega", "hook")
    glow = 1.0 if mode in ("fire", "flash", "mega") else 0.0
    sink = {"sink": 0.55, "crack": 1.0}.get(mode, 1.0) * (0 if before else 1)
    crack = ease(clamp(q * 4)) if mode == "crack" and not before else 0
    gone = 1.0 if (mode in ("fire", "flash", "mega") and not before) or mode == "after" else 0.0
    if mode in ("fire", "flash", "mega") and not before:  # crater
        cr = 160 if mode == "fire" else 260
        d.ellipse([NAIL_X - cr, GROUND - 90, NAIL_X + cr, GROUND - 30], fill=(6, 6, 10, 255))
    if mode == "after": d.ellipse([NAIL_X - 260, GROUND - 90, NAIL_X + 260, GROUND - 30], fill=(6, 6, 10, 255))
    draw_plank(d, sink if mode != "frozen" else 0, crack, gone)
    draw_worker(d)
    draw_hammer(d, ang, glow=glow if (before or mode == "frozen") else glow * (1 - q), blur=(1.0 if fast and before and p / hit > 0.5 else 0) + (0.25 if mode == "frozen" else 0))
    if not before and mode != "frozen":
        if mode == "sink" and q < 0.2:
            for k in range(6):
                an = -math.pi / 2 + (k - 2.5) * 0.4; r = 30 + 120 * q * 5
                d.line([(NAIL_X + math.cos(an) * r * 0.5, GROUND - 216 + math.sin(an) * r * 0.5), (NAIL_X + math.cos(an) * r, GROUND - 216 + math.sin(an) * r)], fill=al(YEL, 1 - q * 5), width=5)
        if mode == "crack":
            r = 60 + 260 * ease(clamp(q * 1.5)); a = 1 - clamp(q * 1.5)
            d.ellipse([NAIL_X - r, GROUND - 130 - r, NAIL_X + r, GROUND - 130 + r], outline=al(WHITE, 0.8 * a), width=10)
            rs = np.random.RandomState(3)
            for _ in range(14):
                an = -math.pi * rs.rand(); sp = 200 + 500 * rs.rand(); dist = sp * clamp(q * 1.3)
                x0 = NAIL_X + math.cos(an) * dist; y0 = GROUND - 120 + math.sin(an) * dist + 900 * clamp(q * 1.3) ** 2 * 0.4
                d.line([(x0, y0), (x0 + 22 * math.cos(an * 3), y0 + 10)], fill=al(BROWN, 1 - clamp(q * 1.2)), width=9)
        if mode in ("fire", "flash", "mega"):
            R = {"fire": 300, "flash": 760, "mega": 1250}[mode] * ease(clamp(q * 2.2)); fa = 1 - clamp((q - 0.3) / 0.7) * (0.7 if mode == "fire" else 0.35)
            fb = Image.new("RGBA", (W, H), (0, 0, 0, 0)); fd = ImageDraw.Draw(fb)
            fd.ellipse([NAIL_X - R, GROUND - 120 - R * 0.8, NAIL_X + R, GROUND - 120 + R * 0.5], fill=al((255, 190, 90), 0.85 * fa))
            fd.ellipse([NAIL_X - R * 0.55, GROUND - 120 - R * 0.5, NAIL_X + R * 0.55, GROUND - 120 + R * 0.3], fill=al((255, 250, 220), 0.95 * fa))
            fb = fb.filter(ImageFilter.GaussianBlur(30)); al_ = np.array(fb); al_[GROUND - 50:, :, 3] = 0; ramp = np.clip(0.3 + 0.7 * (np.arange(H) - 300) / 400, 0.3, 1.0)[:, None]; al_[..., 3] = (al_[..., 3] * ramp).astype(np.uint8); img.alpha_composite(Image.fromarray(al_))
            d = ImageDraw.Draw(img, "RGBA")
            draw_worker(d); draw_hammer(d, ang)  # you and the hammer are unbreakable: still standing in the blast
            if mode != "fire" and v.get("whiteout", True):
                wl = 0.9 * (1 - clamp(q * 6))  # white-out flash
                if wl > 0: d.rectangle([0, 0, W, H], fill=al(WHITE, wl))
    if mode == "after": draw_plank(d, 1.0, 0, 1.0)
    # readouts: real values only, never tweened
    if v.get("speed"): bigtext(img, (540, 470), v["speed"], 64, CYAN, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
    if "value" in v and (not before or mode in ("frozen", "after")):
        g = clamp(q / 0.15) if mode not in ("frozen", "after") else clamp(p / 0.15)
        bigtext(img, (540, 1340), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 104), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    y0 = 1430 if "value" in v else 1350
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a > 0: bigtext(img, (540, y0 - (len(v["labels"]) - 1 - i) * 76), lb, 58, WHITE if i else YEL, a=a)


SCENES["hammer"] = sc_hammer


# ============================================================ rain chain (Nov 18 2026, Ansh: "highly animated like the hammer one, a completely different topic")
RAIN_BLUE = (120, 190, 255); UMB = (232, 60, 80); UMB_D = (150, 30, 50); CLOUD = (52, 58, 82); CLOUD_D = (36, 40, 60)
PX = 540; CANOPY_Y = GROUND - 60 - 330; CANOPY_R = 160; ROOF_X0, ROOF_X1 = 330, 750


def draw_cloud(d, y=470, dark=0.0, t=0.0):
    rs = np.random.RandomState(11)
    for k in range(16):
        cx = -60 + k * 80 + 18 * math.sin(t * 0.6 + k); r = 80 + rs.randint(0, 70); cy = y + rs.randint(-30, 30)
        c = tuple(int(CLOUD[i] * (1 - 0.35 * dark)) for i in range(3))
        d.ellipse([cx - r, cy - r * 0.7, cx + r, cy + r * 0.7], fill=c + (255,))
    for k in range(10):
        cx = -20 + k * 120; r = 70 + rs.randint(0, 40)
        d.ellipse([cx - r, y + 10, cx + r, y + 10 + r * 0.9], fill=CLOUD_D + (255,))


def canopy_y(x):
    dx = (x - PX) / CANOPY_R
    return None if abs(dx) >= 1 else CANOPY_Y + 6 - CANOPY_R * 0.62 * math.sqrt(1 - dx * dx)


def draw_umbrella(d, dmg=0.0, t=0.0):
    """dmg 0..1: holes, then the canopy shreds away panel by panel (8 panels)."""
    hx, hy = PX + 34, GROUND - 60 - 175
    d.line([(PX + 34, CANOPY_Y), (hx, hy + 30)], fill=al((210, 210, 220), 1), width=7)
    d.arc([hx - 22, hy + 14, hx + 2, hy + 44], 0, 180, fill=al((210, 210, 220), 1), width=6)
    n = 8; rs = np.random.RandomState(5); order = rs.permutation(n)
    for i in range(n):
        lost = clamp((dmg - 0.35 - 0.08 * list(order).index(i)) / 0.12)
        if lost >= 1: continue
        x0 = PX - CANOPY_R + 2 * CANOPY_R * i / n; x1 = PX - CANOPY_R + 2 * CANOPY_R * (i + 1) / n
        pts = [(x0 + (x1 - x0) * u / 6, canopy_y(x0 + (x1 - x0) * u / 6 + (0.5 if u == 0 else -0.5 if u == 6 else 0)) or CANOPY_Y) for u in range(7)]
        sag = 30 * lost
        pts = [(x, y + sag) for x, y in pts] + [(x1 - 4, CANOPY_Y + 6 + sag), (x0 + 4, CANOPY_Y + 6 + sag)]
        d.polygon(pts, fill=al(UMB if i % 2 else UMB_D, 1 - lost))
    for k in range(int(40 * clamp(dmg / 0.35))):  # holes punched through
        x = PX - CANOPY_R * 0.9 + rs.rand() * CANOPY_R * 1.8; y = (canopy_y(x) or CANOPY_Y) + 8 + rs.rand() * 30
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=TH["bg"] + (255,))
    if dmg > 0.35:  # scraps flying off
        for k in range(10):
            ph = (t * 1.3 + k * 0.37) % 1.0; ang = -math.pi * (0.15 + 0.7 * rs.rand())
            x = PX + math.cos(ang) * (CANOPY_R + 420 * ph); y = CANOPY_Y + math.sin(ang) * 220 * ph + 500 * ph * ph
            d.polygon([(x, y), (x + 22, y + 6), (x + 8, y + 20)], fill=al(UMB, 1 - ph))


def draw_house(d, dmg=0.0, t=0.0):
    wall_top = GROUND - 330; apex = GROUND - 520 + 190 * ease(clamp((dmg - 0.6) / 0.3))
    d.rectangle([ROOF_X0 + 30, wall_top, ROOF_X1 - 30, GROUND - 60], fill=(70, 78, 112, 255))
    d.rectangle([PX - 45, GROUND - 200, PX + 45, GROUND - 60], fill=(40, 44, 66, 255))
    for wx in (ROOF_X0 + 75, ROOF_X1 - 145): d.rectangle([wx, wall_top + 60, wx + 70, wall_top + 130], fill=al(YEL, 0.55 * (1 - dmg)))
    d.polygon([(ROOF_X0, wall_top + 4), (PX, apex), (ROOF_X1, wall_top + 4)], fill=(150, 70, 60, 255))
    rs = np.random.RandomState(9)
    for k in range(int(70 * clamp(dmg / 0.6))):
        u = rs.rand(); x = ROOF_X0 + 40 + u * (ROOF_X1 - ROOF_X0 - 80); top = apex + (wall_top - apex) * abs(x - PX) / (PX - ROOF_X0)
        y = top + 12 + rs.rand() * (wall_top - top - 20); r = 5 + rs.rand() * 9
        d.ellipse([x - r, y - r, x + r, y + r], fill=(14, 14, 22, 255))


def draw_rain(d, t, fast, n=110, umbrella=True, roof=False, ytop=560):
    """Drops fall straight down; they stop on the umbrella canopy, the roof, or the ground, where they splash."""
    rs = np.random.RandomState(21); spd = 3600 if fast else 900; ln = 150 if fast else 34
    for k in range(n):
        x = rs.rand() * W; ph = rs.rand(); span = GROUND - 60 - ytop
        stop = GROUND - 60
        if umbrella and canopy_y(x) is not None: stop = canopy_y(x)
        if roof and ROOF_X0 < x < ROOF_X1: stop = GROUND - 330 - (190 * (1 - abs(x - PX) / (PX - ROOF_X0)))
        y = ytop + ((ph * span + t * spd) % span)
        if y > stop: continue
        y0 = max(ytop, y - ln)
        d.line([(x, y0), (x, y)], fill=al(RAIN_BLUE if not fast else (200, 230, 255), 0.75), width=3 if not fast else 4)
    rs2 = np.random.RandomState(int(t * 30) + 7)  # splashes this frame
    for k in range(26 if fast else 10):
        x = rs2.rand() * W; y = GROUND - 60
        if umbrella and canopy_y(x) is not None: y = canopy_y(x)
        if roof and ROOF_X0 < x < ROOF_X1: y = GROUND - 330 - (190 * (1 - abs(x - PX) / (PX - ROOF_X0)))
        if fast:
            for j in range(5):
                an = -math.pi * (0.1 + 0.8 * j / 4); r = 16 + rs2.rand() * 26
                d.line([(x, y), (x + math.cos(an) * r, y + math.sin(an) * r)], fill=al((255, 240, 200), 0.9), width=3)
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=al(WHITE, 0.9))
        else:
            d.arc([x - 14, y - 10, x + 14, y + 6], 200, 340, fill=al(RAIN_BLUE, 0.8), width=3)


def sc_rain(img, d, v, p, t, ctx):
    mode = v.get("mode", "real"); hit = v.get("hit", 0.2); q = clamp((p - hit) / (1 - hit)) if p >= hit else 0.0
    fast = mode in ("fast", "shred", "roof", "storm")
    if v.get("town") or mode == "storm":
        wave = 1400 * ease(clamp(q / 0.85)) if mode == "storm" else None
        draw_town(d, wave, glow=0.0)
    d.rectangle([0, GROUND - 60, W, H], fill=(16, 18, 32, 255)); d.line([(0, GROUND - 60), (W, GROUND - 60)], fill=al(RAIN_BLUE, 0.5), width=4)
    if mode == "fall":  # one drop, 12 km column, free fall (y grows with time squared), no air to stop it
        top, bot = 600, GROUND - 60; x = 600
        draw_cloud(d, y=470, t=t)
        d.line([(170, top), (170, bot)], fill=al(TH["muted"], 0.8), width=4)
        for y in (top, bot): d.line([(150, y), (190, y)], fill=al(TH["muted"], 0.8), width=4)
        if v.get("axis"): bigtext(img, (170, (top + bot) / 2), v["axis"], 50, WHITE, angle=90, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
        u = clamp(p / 0.75); y = top + (bot - top) * u * u
        for k in range(14):  # trail: where the drop was in the last moments
            u2 = max(0.0, u - k * 0.012); y2 = top + (bot - top) * u2 * u2
            d.ellipse([x - 12 + k * 0.6, y2 - 12, x + 12 - k * 0.6, y2 + 12], fill=al(RAIN_BLUE, 0.5 * (1 - k / 14)))
        if u < 1:
            d.polygon([(x, y - 34), (x - 16, y), (x + 16, y)], fill=al((200, 235, 255), 1)); d.ellipse([x - 16, y - 16, x + 16, y + 16], fill=al((200, 235, 255), 1))
        else:
            qq = clamp((p - 0.75) / 0.25); r = 30 + 380 * ease(qq)
            d.ellipse([x - r, bot - r * 0.35, x + r, bot + r * 0.35], outline=al(WHITE, 1 - qq), width=8)
            for j in range(9):
                an = -math.pi * (0.08 + 0.84 * j / 8); rr = 40 + 200 * ease(qq)
                d.line([(x, bot), (x + math.cos(an) * rr, bot + math.sin(an) * rr)], fill=al((255, 240, 200), 1 - qq), width=5)
        person(d, 820, GROUND - 60, 62, 0.0)
    else:
        draw_cloud(d, y=470, dark=1.0 if fast else 0.0, t=t)
        if fast:  # lightning-white flicker on impact
            fl = 0.22 * clamp(1 - (t % 2.7) / 0.12)  # one brief flash every 2.7 s
            if fl > 0.01: img.alpha_composite(Image.new("RGBA", (W, H), al(WHITE, fl))); d = ImageDraw.Draw(img, "RGBA")
        if mode == "roof":
            draw_house(d, dmg=q, t=t); draw_rain(d, t, True, umbrella=False, roof=True)
        elif mode == "storm":
            draw_rain(d, t, True, n=170, umbrella=False)
            dust = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dd = ImageDraw.Draw(dust); a = 0.55 * ease(clamp(q / 0.6))
            dd.ellipse([-300, GROUND - 420, W + 300, GROUND + 100], fill=al((150, 120, 90), a)); img.alpha_composite(dust.filter(ImageFilter.GaussianBlur(60)))
            d = ImageDraw.Draw(img, "RGBA")
        else:
            dmg = 0.0
            if mode == "shred": dmg = ease(clamp(q / 0.8))
            person(d, PX, GROUND - 60, 74, 0.0)
            draw_umbrella(d, dmg=dmg, t=t); draw_rain(d, t, fast)
    if v.get("speed"): bigtext(img, (540, 590), v["speed"], 64, CYAN, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
    show = p >= (0.75 if mode == "fall" else hit)
    if "value" in v and show:
        g = clamp((p - (0.75 if mode == "fall" else hit)) / 0.12)
        bigtext(img, (540, 1340), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 104), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    y0 = 1430 if "value" in v else 1350
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a > 0: bigtext(img, (540, y0 - (len(v["labels"]) - 1 - i) * 76), lb, 58, WHITE if i else YEL, a=a)


SCENES["rain"] = sc_rain

# ============================================================ jump chain (Nov 19 2026, Ansh: "more like the rain and earth ones")
JGY = 1150  # ground line for the jump scene
WORLD_COL = dict(earth=((58, 112, 72), (38, 78, 50)), moon=((150, 150, 160), (104, 104, 116)), pluto=((214, 192, 160), (160, 138, 112)),
                 phobos=((124, 104, 92), (84, 70, 62)), deimos=((154, 134, 112), (106, 90, 76)), comet=((84, 84, 92), (56, 56, 64)))
GOLD = (255, 196, 60)


def draw_stars(d, n=110, seed=4, a=0.75, ymax=1250):
    rs = np.random.RandomState(seed)
    for _ in range(n):
        x, y, r = rs.rand() * W, rs.rand() * ymax, 1 + rs.rand() * 2.2
        d.ellipse([x - r, y - r, x + r, y + r], fill=al(WHITE, a * (0.25 + 0.75 * rs.rand())))


def draw_body(d, kind, cx, top, R):
    """A world seen from the side: circle (or two-lobed comet) of radius R px whose top touches y = top."""
    c1, c2 = WORLD_COL[kind]
    if kind == "comet":
        r2 = R * 0.72; d.ellipse([cx + R * 0.35 - r2, top + R * 1.05, cx + R * 0.35 + r2, top + R * 1.05 + 1.7 * r2], fill=c2 + (255,))
        d.ellipse([cx - R, top, cx + R, top + 1.6 * R], fill=c1 + (255,))
    else:
        if kind == "earth": d.ellipse([cx - R - 26, top - 26, cx + R + 26, top + 2 * R + 26], fill=al((80, 160, 255), 0.25))
        d.ellipse([cx - R, top, cx + R, top + 2 * R], fill=c1 + (255,))
    rs = np.random.RandomState(len(kind) * 7)
    if kind != "earth":
        for _ in range(26):
            x = cx + (rs.rand() * 2 - 1) * R * 0.95; y = top + 14 + rs.rand() * min(R, 420)
            if (x - cx) ** 2 + (y - top - R) ** 2 > (R * 0.97) ** 2: continue
            r = 6 + rs.rand() * min(40, R * 0.12); d.ellipse([x - r, y - r * 0.45, x + r, y + r * 0.45], fill=c2 + (255,))


def draw_lander(d, x, y, s=1.0, ang=0.0):
    """Small box lander with three legs and a solar-panel lid; (x, y) = feet."""
    def P(u, v): return (x + (u * math.cos(ang) - v * math.sin(ang)) * s, y + (u * math.sin(ang) + v * math.cos(ang)) * s)
    d.polygon([P(-30, -30), P(30, -30), P(30, -66), P(-30, -66)], fill=al(GOLD, 1))
    d.polygon([P(-38, -66), P(38, -66), P(30, -76), P(-30, -76)], fill=al((70, 90, 160), 1))
    for u in (-26, 0, 26): d.line([P(u * 0.6, -30), P(u * 1.5, 0)], fill=al(STEEL, 1), width=max(2, int(4 * s)))


def fade_bottom(img, y0=1190, y1=1450):
    """Fade everything below y0 into the background so the caption band stays readable."""
    a = np.zeros((H, 1), np.float32); a[y0:y1, 0] = np.linspace(0, 1, y1 - y0); a[y1:, 0] = 1
    ov = np.zeros((H, W, 4), np.uint8); ov[..., :3] = TH["bg"]; ov[..., 3] = (np.repeat(a, W, axis=1) * 255).astype(np.uint8)
    img.alpha_composite(Image.fromarray(ov))


def fade_top(img, y0=300, y1=420):
    """Fade the scene out above y1 so the two-line head stays readable."""
    a = np.zeros((H, 1), np.float32); a[:y0, 0] = 1; a[y0:y1, 0] = np.linspace(1, 0, y1 - y0)
    ov = np.zeros((H, W, 4), np.uint8); ov[..., :3] = TH["bg"]; ov[..., 3] = (np.repeat(a, W, axis=1) * 255).astype(np.uint8)
    img.alpha_composite(Image.fromarray(ov))


def sc_jump(img, d, v, p, t, ctx):
    """Same jump (same take-off speed) on different worlds. h = peak height in m sets the camera scale; R = world radius in m."""
    mode = v.get("mode", "jump"); kind = v.get("world", "earth"); h = float(v.get("h", 0.5)); Rm = float(v.get("R", 6.371e6))
    draw_stars(d, a=0.35 if kind == "earth" else 0.8)
    u = clamp((p - 0.14) / 0.7)
    if mode == "philae":
        R = 380; top = 1000; draw_body(d, "comet", 540, top, R)
        x = 250 + 560 * u; arc = 520 * 4 * u * (1 - u); y = top + 30 - arc
        for k in range(1, 12):  # dotted path behind it
            u2 = max(0.0, u - k * 0.03); x2 = 250 + 560 * u2; y2 = top + 30 - 520 * 4 * u2 * (1 - u2); d.ellipse([x2 - 5, y2 - 5, x2 + 5, y2 + 5], fill=al(GOLD, 0.5 * (1 - k / 12)))
        fade_bottom(img); d = ImageDraw.Draw(img, "RGBA")
        draw_lander(d, x, y, 1.0, ang=0.6 * math.sin(t * 1.3))
        if u >= 1: d.arc([x - 60, y - 16, x + 60, y + 16], 200, 340, fill=al(WHITE, 0.6), width=4)
    else:
        pxpm = min(560.0 / max(h, 1e-6), 114.0); s = 1.75 * pxpm / 2.7; Rpx = min(Rm * pxpm, 12000.0)
        if mode == "escape":  # you never come back: the comet shrinks below you
            Rc = 300 * (1 - 0.88 * ease(u)); top = JGY + 700 * ease(u)
            draw_body(d, "comet", 540, top, Rc); fade_bottom(img); d = ImageDraw.Draw(img, "RGBA"); fy = JGY - 30 - 520 * ease(clamp(u * 1.4)); s = max(10.0, 46 * (1 - 0.5 * u))
            for k in range(1, 10):
                d.ellipse([540 - 6, fy + k * 26 - 6, 540 + 6, fy + k * 26 + 6], fill=al(CYAN, 0.45 * (1 - k / 10)))
            person(d, 540, fy, s, 0.0, col=WHITE)
        else:
            draw_body(d, kind, 540, JGY, Rpx); fade_bottom(img); d = ImageDraw.Draw(img, "RGBA")
            rise = h * 4 * u * (1 - u) * pxpm if 0 < u < 1 else 0.0
            if p < 0.14 or u >= 1: rise = 0.0
            fy = JGY - rise
            if 0 < u < 1:  # trail
                for k in range(1, 10):
                    u2 = max(0.0, u - k * 0.025); y2 = JGY - h * 4 * u2 * (1 - u2) * pxpm - (s * 1.3 if s >= 8 else 0)
                    d.ellipse([540 - 7, y2 - 7, 540 + 7, y2 + 7], fill=al(CYAN, 0.4 * (1 - k / 10)))
            if s >= 8: person(d, 540, fy, s, 0.0)
            else:
                d.ellipse([540 - 22, fy - 22, 540 + 22, fy + 22], fill=al(CYAN, 0.25)); d.ellipse([540 - 7, fy - 7, 540 + 7, fy + 7], fill=al(WHITE, 1))
                bigtext(img, (540 + 92, fy - 10), "YOU", 40, WHITE, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
            # ruler: ground to the real peak height on this scale
            top_y = JGY - h * pxpm; a = clamp((p - 0.3) / 0.1)
            if a > 0:
                d.line([(820, JGY), (820, top_y)], fill=al(YEL, 0.8 * a), width=5)
                for y in (JGY, top_y): d.line([(796, y), (844, y)], fill=al(YEL, 0.8 * a), width=5)
            if u >= 1 and p < 0.98:  # landing dust
                q = clamp((p - 0.84) / 0.14); r = 30 + 140 * ease(q)
                d.arc([540 - r, JGY - 18, 540 + r, JGY + 18], 180, 360, fill=al(c := WORLD_COL[kind][0], 1 - q), width=6)
    if v.get("speed"): bigtext(img, (540, 470), v["speed"], 60, CYAN, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
    at = v.get("value_at", 0.5)
    if "value" in v and p >= at:
        g = clamp((p - at) / 0.12)
        bigtext(img, (540, 1340), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 104), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    y0 = 1430 if "value" in v else 1350
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a > 0: bigtext(img, (540, y0 - (len(v["labels"]) - 1 - i) * 76), lb, 58, WHITE if i else YEL, a=a)


SCENES["jump"] = sc_jump


# ============================================================ elevator chain (Nov 19 2026)
SH_X0, SH_X1 = 330, 750; CAR_W, CAR_H = 300, 360; CAR_TOP = 600; FLOOR_GAP = 250


def sc_elevator(img, d, v, p, t, ctx):
    """Inside a shaft, camera on the car. mode: ride | fall | jump | impact | coil | brake. hit = beat fraction where the cable snaps / brake bites;
    land = fraction where the pit floor arrives (impact, coil, jump)."""
    mode = v.get("mode", "fall"); hit = v.get("hit", 0.12); land = v.get("land", 0.8)
    snapped = mode != "ride" and p >= hit; q = (p - hit) if snapped else 0.0
    landed = mode in ("impact", "coil", "jump") and p >= land
    # how far the car has moved (px, for scrolling the shaft): steady ride, or free fall (grows with time squared)
    if mode == "ride": off = 900 * p
    elif mode == "brake": off = 60 * ease(clamp(q / 0.06)) if snapped else 0.0
    else: off = 9000 * q * q if not landed else 9000 * (land - hit) ** 2
    shake = 0.0
    if landed: shake = 18 * max(0.0, 1 - (p - land) / 0.12) * (0.4 if mode == "coil" else 1.0)
    sx = shake * math.sin(t * 90); sy = shake * math.cos(t * 77)
    # shaft walls and floors scrolling up past the car
    d.rectangle([SH_X0 - 40 + sx, 0, SH_X0 + sx, 1470], fill=(30, 34, 54, 255)); d.rectangle([SH_X1 + sx, 0, SH_X1 + 40 + sx, 1470], fill=(30, 34, 54, 255))
    for k in range(-2, 12):
        y = (k * FLOOR_GAP - off) % (FLOOR_GAP * 8) - FLOOR_GAP + sy
        if y > 1470: continue
        d.rectangle([SH_X0 - 40 + sx, y, SH_X0 + 10 + sx, y + 14], fill=(60, 66, 96, 255)); d.rectangle([SH_X1 - 10 + sx, y, SH_X1 + 40 + sx, y + 14], fill=(60, 66, 96, 255))
        d.rectangle([SH_X1 + 4 + sx, y - 150, SH_X1 + 34 + sx, y], outline=(70, 76, 110, 255), width=3)  # landing door edge
    for x in (SH_X0 + 24, SH_X1 - 24): d.line([(x + sx, 0), (x + sx, 1470)], fill=al(STEEL, 0.35), width=6)  # guide rails
    cx0, cx1 = 540 - CAR_W / 2 + sx, 540 + CAR_W / 2 + sx; ctop = CAR_TOP + sy; cbot = ctop + CAR_H
    if landed and mode == "impact": cbot_c = cbot; ctop = ctop + 70 * ease(clamp((p - land) / 0.05))  # crumple
    # pit floor and coiled cable
    if mode in ("impact", "coil", "jump"):
        floor_y = cbot + max(0.0, (land - p) * 3200) if not landed else cbot
        if floor_y < 1470:
            d.rectangle([SH_X0 - 40 + sx, floor_y, SH_X1 + 40 + sx, 1470], fill=(22, 24, 40, 255)); d.line([(SH_X0 - 40, floor_y), (SH_X1 + 40, floor_y)], fill=al(RED, 0.6), width=4)
            if mode == "coil":
                for k in range(7):  # a spring of coiled cable on the pit floor
                    yy = floor_y - 10 - k * 12; d.ellipse([440 + sx, yy - 10, 640 + sx, yy + 10], outline=al(STEEL, 0.9), width=5)
    # cable
    if not snapped or mode == "brake":
        d.line([(540 + sx, 0), (540 + sx, ctop)], fill=al(STEEL, 1), width=8)
    elif mode != "brake":
        whip = clamp(q / 0.25)
        d.line([(540 + sx, max(0, ctop - 1500 * whip - 400)), (540 + sx, 0)], fill=al(STEEL, 0.9), width=8)
        pts = [(540 + sx + 60 * math.sin(k * 0.9 + t * 9) * (k / 8), ctop - k * 40) for k in range(9)]; d.line(pts, fill=al(STEEL, 0.9), width=7, joint="curve")
        if q < 0.08: bigtext(img, (540, ctop - 120), "SNAP", 90, RED, angle=-8, outline=(60, 0, 0)); d = ImageDraw.Draw(img, "RGBA")
    if mode == "brake" and snapped:  # safety brake jaws bite the rails
        for x in (SH_X0 + 24, SH_X1 - 24):
            for k in range(4): d.polygon([(x - 18 + sx, ctop + 20 + k * 24), (x + 18 + sx, ctop + 32 + k * 24), (x - 18 + sx, ctop + 44 + k * 24)], fill=al(YEL, 1))
        if q < 0.15:
            rs = np.random.RandomState(int(t * 30))
            for _ in range(14):
                x = (SH_X0 + 24 if rs.rand() < 0.5 else SH_X1 - 24) + sx; an = rs.rand() * math.pi * 2; r = 20 + rs.rand() * 50
                d.line([(x, ctop + 60), (x + math.cos(an) * r, ctop + 60 + math.sin(an) * r)], fill=al(ORANGE, 0.9), width=3)
    # the car
    d.rectangle([cx0, ctop, cx1, cbot], fill=(52, 58, 88, 255), outline=al(WHITE, 0.9), width=6)
    d.rectangle([cx0 + 14, ctop + 14, cx1 - 14, cbot - 14], fill=(26, 30, 50, 255))
    d.line([(540 + sx, ctop + 14), (540 + sx, cbot - 14)], fill=al(WHITE, 0.15), width=3)
    # the rider: stands, floats in free fall, kicks up for the last-second jump
    floating = snapped and mode not in ("brake",) and not landed
    fl = ease(clamp(q / 0.3)) if floating else 0.0
    rot = 0.35 * math.sin(t * 2.2) * fl
    fy = cbot - 18 - 70 * fl
    if mode == "jump" and land - 0.1 < p < land: fy = cbot - 18 - 70 * fl - 40 * clamp((p - land + 0.1) / 0.05)
    if landed: fy = cbot - 18
    person(d, 540 + sx, fy, 52 if not (landed and mode == "impact") else 40, rot)
    if floating and mode != "ride":
        for k in range(10):  # speed lines rushing up past the car
            x = SH_X0 + 40 + (k * 67) % (SH_X1 - SH_X0 - 80); y = ((k * 211 + t * 2600) % 1500)
            if not (cx0 < x < cx1 and ctop < y < cbot): d.line([(x, y), (x, y - 90 * min(1.0, q * 6))], fill=al(WHITE, 0.25), width=3)
    if landed and mode in ("impact", "jump"):  # dust burst
        qq = clamp((p - land) / 0.3); r = 60 + 420 * ease(qq)
        dust = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dd = ImageDraw.Draw(dust)
        dd.ellipse([540 - r, cbot - r * 0.5, 540 + r, cbot + r * 0.5], fill=al((170, 150, 130), 0.7 * (1 - qq))); img.alpha_composite(dust.filter(ImageFilter.GaussianBlur(24)))
        d = ImageDraw.Draw(img, "RGBA")
        if p - land < 0.03: img.alpha_composite(Image.new("RGBA", (W, H), al(WHITE, 0.3))); d = ImageDraw.Draw(img, "RGBA")
    fade_top(img); d = ImageDraw.Draw(img, "RGBA")
    if v.get("speed"): bigtext(img, (540, 470), v["speed"], 60, CYAN, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
    at = v.get("value_at", land if mode in ("impact", "coil", "jump") else 0.5)
    if "value" in v and p >= at:
        g = clamp((p - at) / 0.12)
        bigtext(img, (540, 1340), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 104), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    y0 = 1430 if "value" in v else 1350
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a > 0: bigtext(img, (540, y0 - (len(v["labels"]) - 1 - i) * 76), lb, 58, WHITE if i else YEL, a=a)


SCENES["elevator"] = sc_elevator


# ============================================================ tunnel through a world (Nov 19 2026)
TCX, TCY, TR = 540, 830, 330


def sc_tunnel(img, d, v, p, t, ctx):
    """Cross-section with a straight tunnel; a faller does simple harmonic motion (uniform density). mode: drop | back | chord | kola.
    world: earth | moon."""
    mode = v.get("mode", "drop"); world = v.get("world", "earth"); R = TR if world == "earth" else 250
    draw_stars(d, n=70, a=0.5)
    if world == "earth":
        layers = [(1.0, (70, 110, 160)), (0.985, (196, 88, 48)), (0.55, (238, 148, 52)), (0.19, (255, 222, 120))]
    else:
        layers = [(1.0, (150, 150, 160)), (0.97, (118, 118, 130)), (0.2, (170, 120, 90))]
    for f, c in layers: d.ellipse([TCX - R * f, TCY - R * f, TCX + R * f, TCY + R * f], fill=c + (255,))
    u = clamp((p - 0.08) / 0.8)
    def runner(a0, a1, uu, col):
        x0, y0 = TCX + R * math.sin(a0), TCY - R * math.cos(a0); x1, y1 = TCX + R * math.sin(a1), TCY - R * math.cos(a1)
        d.line([(x0, y0), (x1, y1)], fill=(10, 10, 16, 255), width=26)
        s = (1 - math.cos(math.pi * uu)) / 2  # simple harmonic motion: slow at the ends, fastest in the middle
        for k in range(1, 9):
            u2 = max(0.0, uu - k * 0.02); s2 = (1 - math.cos(math.pi * u2)) / 2; xx, yy = x0 + (x1 - x0) * s2, y0 + (y1 - y0) * s2
            d.ellipse([xx - 9, yy - 9, xx + 9, yy + 9], fill=al(col, 0.45 * (1 - k / 9)))
        x, y = x0 + (x1 - x0) * s, y0 + (y1 - y0) * s
        d.ellipse([x - 26, y - 26, x + 26, y + 26], fill=al(col, 0.3)); d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=al(WHITE, 1))
    if mode == "kola":
        d.line([(TCX, TCY - R), (TCX, TCY - R + 2)], fill=(10, 10, 16, 255), width=4)  # the real hole: under 1 px at this scale
        a = clamp(p / 0.2); bx0, by0, bx1, by1 = 170, 1010, 910, 1250
        d.line([(TCX, TCY - R), ((bx0 + bx1) / 2, by0)], fill=al(YEL, 0.6 * a), width=3)
        d.ellipse([TCX - 18, TCY - R - 18, TCX + 18, TCY - R + 18], outline=al(YEL, a), width=4)
        d.rectangle([bx0, by0, bx1, by1], fill=al((14, 16, 28), a), outline=al(YEL, a), width=4)
        d.rectangle([bx0 + 4, by0 + 70, bx1 - 4, by1 - 4], fill=al((70, 110, 160), a))
        hd = (by1 - by0 - 74) * clamp((p - 0.2) / 0.4) * 0.5  # inset spans twice the hole depth
        d.rectangle([(bx0 + bx1) / 2 - 6, by0 + 70, (bx0 + bx1) / 2 + 6, by0 + 70 + hd], fill=(8, 8, 12, 255))
    elif mode == "chord":
        runner(0.0, math.pi, u, CYAN); runner(-2.2, 1.0, u, LIME)
    elif mode == "back":
        uu = clamp(u * 2) if u < 0.5 else 1 - clamp((u - 0.5) * 2)
        runner(0.0, math.pi, uu, CYAN)
    else:
        runner(0.0, math.pi, u, CYAN)
    if v.get("speed"): bigtext(img, (540, 1240), v["speed"], 60, CYAN, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
    at = v.get("value_at", 0.5)
    if "value" in v and p >= at:
        g = clamp((p - at) / 0.12)
        bigtext(img, (540, 1340), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 104), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    y0 = 1430 if "value" in v else 1350
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a > 0: bigtext(img, (540, y0 - (len(v["labels"]) - 1 - i) * 76), lb, 58, WHITE if i else YEL, a=a)


SCENES["tunnel"] = sc_tunnel

# ============================================================ bolt chain (Nov 21 2026: catch one lightning bolt, stakes climb bulb -> house -> country -> planet)
JAR_X = 200; BOLT_Y0 = 560; BGND = 1190


def bolt_path(x0, y0, x1, y1, seed, n=11, jag=70):
    rs = np.random.RandomState(seed); pts = [(x0, y0)]
    for i in range(1, n):
        u = i / n; pts.append((x0 + (x1 - x0) * u + (rs.rand() - 0.5) * jag * 2 * (1 - u * 0.6), y0 + (y1 - y0) * u + (rs.rand() - 0.5) * 18))
    pts.append((x1, y1)); return pts


def draw_bolt(img, x1, y1, seed, a=1.0, x0=None, w=10):
    """A jagged bolt from the cloud base to (x1, y1), drawn as a soft glow plus a white core."""
    x0 = x1 + 120 if x0 is None else x0; pts = bolt_path(x0, BOLT_Y0, x1, y1, seed)
    gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(gl)
    gd.line(pts, fill=al((150, 190, 255), 0.9 * a), width=w * 4, joint="curve")
    rs = np.random.RandomState(seed + 99)  # side branches
    for k in range(3):
        i = 2 + rs.randint(0, len(pts) - 4); bx, by = pts[i]; b2 = bolt_path(bx, by, bx + (rs.rand() - 0.5) * 360, by + 120 + rs.rand() * 160, seed + k, n=5, jag=40)
        gd.line(b2, fill=al((150, 190, 255), 0.6 * a), width=w * 2, joint="curve")
    img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(img, "RGBA"); d.line(pts, fill=al(WHITE, a), width=w, joint="curve")
    rs = np.random.RandomState(seed + 99)
    for k in range(3):
        i = 2 + rs.randint(0, len(pts) - 4); bx, by = pts[i]; b2 = bolt_path(bx, by, bx + (rs.rand() - 0.5) * 360, by + 120 + rs.rand() * 160, seed + k, n=5, jag=40)
        d.line(b2, fill=al(WHITE, 0.8 * a), width=max(3, w // 2), joint="curve")


def draw_jar(d, fill, glow=0.0, x=JAR_X):
    """A big glass capture jar with a lightning rod on top; fill 0..1 = stored charge (glowing yellow)."""
    top, bot = BGND - 300, BGND
    d.line([(x, top - 160), (x, top)], fill=al(STEEL, 1), width=10); d.polygon([(x - 12, top - 160), (x + 12, top - 160), (x, top - 200)], fill=al(STEEL, 1))
    if glow > 0: d.rounded_rectangle([x - 120, top - 20, x + 120, bot + 20], radius=40, fill=al(YEL, 0.18 * glow))
    d.rounded_rectangle([x - 95, top, x + 95, bot], radius=30, fill=(20, 26, 46, 255), outline=al((170, 200, 240), 0.9), width=6)
    if fill > 0:
        fy = bot - 12 - (bot - top - 24) * fill
        d.rounded_rectangle([x - 82, fy, x + 82, bot - 12], radius=22, fill=al(YEL, 0.85))
        for k in range(4): d.line([(x - 60 + k * 40, fy + 20), (x - 40 + k * 40, fy + 60), (x - 50 + k * 40, fy + 60), (x - 30 + k * 40, fy + 100)], fill=al(WHITE, 0.7), width=4)
    d.rectangle([x - 70, top - 14, x + 70, top + 4], fill=al(STEEL, 1))


def draw_cable(d, x0, x1, y=BGND - 20, live=0.0, t=0.0):
    d.line([(x0, y), (x1, y)], fill=(60, 64, 90, 255), width=10)
    if live > 0:
        for k in range(8):
            u = ((t * 1.6 + k / 8) % 1.0); x = x0 + (x1 - x0) * u
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=al(YEL, live))


def draw_bulb(img, x, y, on=0.0):
    if on > 0:
        gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(gl); r = 260
        gd.ellipse([x - r, y - r, x + r, y + r], fill=al(YEL, 0.55 * on)); img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(60)))
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([x - 110, y - 120, x + 110, y + 100], fill=al((255, 236, 150), 1) if on > 0.5 else (60, 66, 92, 255), outline=al(WHITE, 0.8), width=5)
    d.polygon([(x - 60, y + 80), (x + 60, y + 80), (x + 45, y + 140), (x - 45, y + 140)], fill=al((255, 236, 150), 1) if on > 0.5 else (60, 66, 92, 255))
    for k in range(4): d.rectangle([x - 48, y + 140 + k * 22, x + 48, y + 154 + k * 22], fill=al(STEEL, 1))
    d.line([(x, y + 228), (x, BGND - 20)], fill=(60, 64, 90, 255), width=10)
    if on > 0.5: d.line([(x - 40, y - 10), (x - 15, y + 30), (x + 15, y - 10), (x + 40, y + 30)], fill=al(ORANGE, 1), width=6)


def draw_lit_globe(img, t, p, n_rate=40.0, seed=3):
    """Earth with flashes popping up at random places, mostly between 40 S and 40 N, at n_rate per second of video."""
    globe(img, GX, GY - 40, GR, spin_rot(t, rate=8.0), tilt=12.0, night=0.18, light=(-0.3, -0.2, 0.9), glow=0.8)
    d = ImageDraw.Draw(img, "RGBA"); rs = np.random.RandomState(seed); N = 2000
    lat = (rs.rand(N) - 0.5) * 80; lon = (rs.rand(N) - 0.5) * 170; ts = rs.rand(N) * (N / n_rate)  # N flashes spread at n_rate per second
    for la, lo, t0 in zip(lat, lon, ts):
        dt = (t % (N / n_rate)) - t0
        if 0 <= dt < 0.25:
            x, y, z = surface_xy(GX, GY - 40, GR, la, lo, tilt=12.0)
            if z <= 0.05: continue
            a = 1 - dt / 0.25; r = 10 + 26 * (1 - a)
            d.ellipse([x - r, y - r, x + r, y + r], fill=al((200, 225, 255), 0.5 * a)); d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=al(WHITE, a))


def sc_bolt(img, d, v, p, t, ctx):
    mode = v.get("mode", "bulb"); hit = v.get("hit", 0.2); after = p >= hit
    q = clamp((p - hit) / (1 - hit)) if after else 0.0
    if mode in ("earth", "clock"):
        draw_stars(d, a=0.6); draw_lit_globe(img, t, p); d = ImageDraw.Draw(img, "RGBA")
        if mode == "clock":  # stopwatch ring filling over the beat
            cx, cy, r = 880, 1200 - 40, 90; d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(14, 16, 30, 255), outline=al(WHITE, 0.9), width=8)
            d.pieslice([cx - r + 14, cy - r + 14, cx + r - 14, cy + r - 14], -90, -90 + 360 * ease(clamp(p / 0.8)), fill=al(RED, 0.85))
            d.rectangle([cx - 16, cy - r - 30, cx + 16, cy - r - 6], fill=al(WHITE, 0.9))
    else:
        dark = mode in ("why", "bulb", "house", "town", "miss")
        draw_cloud(d, y=470, dark=1.0 if dark else 0.0, t=t)
        if mode == "town":
            draw_town(d, None, glow=0.0)
        d.rectangle([0, BGND, W, H], fill=(16, 18, 32, 255)); d.line([(0, BGND), (W, BGND)], fill=al(CYAN, 0.4), width=4)
        if mode == "town" and not after:  # the grid is dark until the bolt lands
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); od = ImageDraw.Draw(ov); od.rectangle([0, 700, W, BGND - 2], fill=TH["bg"] + (150,)); img.alpha_composite(ov); d = ImageDraw.Draw(img, "RGBA")
        sa = 0.0
        if after:
            dt = (p - hit) * v.get("_len", 6.0)
            sa = 1.0 if any(abs(dt - s) < 0.06 for s in (0.0, 0.15, 0.3, 0.44)) else (0.3 if dt < 0.5 else 0.0)
        fill = ease(clamp(q * 3)) if after and mode != "miss" else 0.0
        if mode == "bulb":
            draw_cable(d, JAR_X + 95, 760, live=fill, t=t); draw_jar(d, fill, glow=fill); draw_bulb(img, 760, 830, on=fill); d = ImageDraw.Draw(img, "RGBA")
        elif mode == "why":
            draw_jar(d, fill, glow=fill)
        elif mode == "house":
            draw_house(d, 0.0, t)
            if not after:
                wt = GROUND - 330
                for wx in (ROOF_X0 + 75, ROOF_X1 - 145): d.rectangle([wx, wt + 60, wx + 70, wt + 130], fill=(24, 26, 40, 255))
            else:
                gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(gl); gd.ellipse([300, 700, 780, 1180], fill=al(YEL, 0.25 * fill)); img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(50))); d = ImageDraw.Draw(img, "RGBA")
            draw_cable(d, JAR_X + 95, ROOF_X0 + 30, live=fill, t=t); draw_jar(d, fill, glow=fill)
        elif mode == "town":
            draw_jar(d, fill, glow=fill)
        elif mode == "miss":
            draw_jar(d, 0.0)
        if mode == "miss":  # bolts land everywhere except the rod
            for k, (hx, ph) in enumerate([(620, 0.06), (900, 0.22), (420, 0.38), (760, 0.52), (520, 0.68), (960, 0.84)]):
                if p >= ph:
                    dt = (p - ph) * v.get("_len", 6.0); a = 1.0 if dt < 0.12 else (0.3 if dt < 0.35 else 0.0)
                    if a > 0: draw_bolt(img, hx, BGND, 40 + k, a=a, x0=hx + 80 - k * 60); d = ImageDraw.Draw(img, "RGBA")
        elif sa > 0:
            if sa >= 1: img.alpha_composite(Image.new("RGBA", (W, H), al(WHITE, 0.25)))
            draw_bolt(img, JAR_X, BGND - 500, 7 + int(v.get("seed", 0)), a=sa, x0=JAR_X + 260); d = ImageDraw.Draw(img, "RGBA")
        fade_bottom(img, 1260, 1300); d = ImageDraw.Draw(img, "RGBA")
    if v.get("speed"): bigtext(img, (540, 650), v["speed"], 60, CYAN, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
    at = v.get("value_at", hit + 0.08)
    if "value" in v and p >= at:
        g = clamp((p - at) / 0.12)
        bigtext(img, (540, 1340), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 100), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    y0 = 1430 if "value" in v else 1350
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a <= 0: continue
        if mode == "why": bigtext(img, (680, 800 + i * 110), lb, 62, YEL if i else WHITE, a=a, outline=(5, 12, 24))  # beside the jar, clear of the value
        else: bigtext(img, (540, y0 - (len(v["labels"]) - 1 - i) * 76), lb, 58, WHITE if i else YEL, a=a)


SCENES["bolt"] = sc_bolt


# ============================================================ facts + check
WORDNUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
           "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def stats(S):
    env = {"math": math}; out = {}
    for k, val in S.get("inputs", {}).items(): env[k] = val; out[k] = val
    for name, expr in S.get("calc", []):
        val = eval(expr, {"__builtins__": {}, "math": math, "min": min, "max": max, "round": round, "abs": abs}, env)
        env[name] = val; out[name] = val
    return out


def numbers_in(text):
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text)]


def spoken(text):
    """'one thousand six hundred and seventy four' -> 1674, 'one point seven' -> 1.7."""
    toks = re.findall(r"[a-z]+|[.,:;!?]", text.lower()); out = []; i = 0
    while i < len(toks):
        if toks[i] in WORDNUM or toks[i] in ("hundred", "thousand"):
            total, cur = 0, 0
            while i < len(toks) and (toks[i] in WORDNUM or toks[i] in ("hundred", "thousand", "and")):
                w = toks[i]
                if w == "and":
                    if i + 1 < len(toks) and toks[i + 1] in WORDNUM: i += 1; continue
                    break
                if w in WORDNUM: cur += WORDNUM[w]
                elif w == "hundred": cur = (cur or 1) * 100
                else: total += (cur or 1) * 1000; cur = 0
                i += 1
            val = float(total + cur)
            if i + 1 < len(toks) and toks[i] == "point" and toks[i + 1] in WORDNUM:
                i += 1; ds = ""
                while i < len(toks) and toks[i] in WORDNUM and WORDNUM[toks[i]] < 10: ds += str(WORDNUM[toks[i]]); i += 1
                val = float(f"{int(val)}.{ds}")
            out.append(val)
        else: i += 1
    return [n for n in out if n not in (1, 2)]


def pool(st):
    p = set()
    for x in st.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            p.update({float(x), float(round(x)), round(float(x), 1), round(float(x), 2), float(int(x))})
    return p


def vis_texts(v):
    out = list(v.get("labels", []))
    for r in v.get("rows", []):
        out += [r.get("label", ""), r.get("text", "")]
        if "value" in r: out.append(fmt_num(r["value"], r.get("dec", 0)))
    if "value" in v: out.append(fmt_num(v["value"], v.get("dec", 0)))
    if v.get("speed"): out.append(v["speed"])
    if v.get("axis"): out.append(v["axis"])
    return out


def check(S):
    st = stats(S); ok = pool(st) | {float(c) for c in S.get("constants", [])}
    bad = []
    def chk(where, txt, speak=False):
        if "—" in txt or "–" in txt: bad.append(f"{where}: em/en dash")
        for n in numbers_in(txt):
            if n not in ok: bad.append(f"{where}: number {n:g} not in stats/constants")
        if speak:
            for n in spoken(txt):
                if float(n) not in ok: bad.append(f"{where}: spoken number {n:g} not in stats/constants")
    for i, b in enumerate(S["beats"]):
        chk(f"beat {i} speak", b.get("speak", b["say"]), True); chk(f"beat {i} say", b["say"])
        for f in ("hook", "head"):
            for ln in b.get(f, []): chk(f"beat {i} {f}", ln)
        if b.get("chip"): chk(f"beat {i} chip", b["chip"])
        if b.get("stamp"): chk(f"beat {i} stamp", b["stamp"]["text"])
        for tx in vis_texts(b.get("vis", {})): chk(f"beat {i} vis", tx)
    for f in ("title", "caption"): chk(f, S.get(f, ""))
    if not S.get("sources"): bad.append("no sources")
    if S["beats"][-1].get("nocap") is not True: bad.append("last beat must be the nocap follow line")
    print("\n".join(bad) if bad else "CHECK OK"); return not bad


# ============================================================ render
def frame(t, S, tl, bg, total):
    img = bg.copy().convert("RGBA"); d = ImageDraw.Draw(img, "RGBA")
    bi = 0
    for i, s in enumerate(tl):
        if t >= s["start"] - 0.1: bi = i
    seg = tl[bi]; b = seg["beat"]
    vj = next((j for j in range(bi, -1, -1) if "vis" in tl[j]["beat"]), 0)
    v_end = next((tl[j]["start"] for j in range(vj + 1, len(tl)) if "vis" in tl[j]["beat"]), total)
    p = clamp((t - tl[vj]["start"] + 0.1) / max(v_end - tl[vj]["start"] - 0.15, 0.1))
    vis = tl[vj]["beat"].get("vis", {})
    if vis.get("scene"):
        vis = dict(vis); vis["_len"] = max(v_end - tl[vj]["start"], 0.5)
        SCENES[vis["scene"]](img, d, vis, p, t, {})
    d = ImageDraw.Draw(img, "RGBA")
    head = next((tl[j]["beat"]["head"] for j in range(bi, -1, -1) if tl[j]["beat"].get("head")), None)
    if bi == 0 and b.get("hook"):
        fh = font(F_BOLD, 100)
        wmax = max(d.textlength(ln, font=fh) for ln in b["hook"])
        if wmax > 980: fh = font(F_BOLD, int(100 * 980 / wmax))
        for i, ln in enumerate(b["hook"]):
            g = clamp((t - i * 0.35) / 0.2)
            tc(d, (540, 150 + i * 108), ln, fh, al(YEL if i == 2 else TH["ink"], g))
    elif head:
        fh = font(F_BOLD, 86)
        for i, ln in enumerate(head):
            f2 = fh if d.textlength(ln, font=fh) < 980 else font(F_BOLD, int(86 * 980 / d.textlength(ln, font=fh)))
            tc(d, (540, 160 + i * 98), ln, f2, al(YEL if i else TH["ink"], 1))
    chip = b.get("chip")
    if chip:
        fc = font(F_BOLD, 46); w = d.textlength(chip, font=fc) + 60
        d.rounded_rectangle([540 - w / 2, 330, 540 + w / 2, 394], radius=32, fill=al(YEL, 1)); tc(d, (540, 362), chip, fc, (20, 16, 8, 255))
    stp = b.get("stamp")
    if stp:
        ts = seg["start"] + stp.get("at", 0.5) * (seg["end"] - seg["start"]); q = (t - ts) / 0.22
        if q > 0:
            sc = 1.0 + 0.6 * max(0, 1 - q)
            bigtext(img, (stp.get("x", 780), stp.get("y", 560)), stp["text"], stp.get("size", 92), RED, angle=stp.get("angle", 12), a=clamp(q), scale=sc, outline=(60, 0, 0))
            d = ImageDraw.Draw(img, "RGBA")
    if not b.get("nocap"):
        M.caption(d, seg, min(t, seg["end"]), TH)
    else:
        tc(d, (540, CAP_Y), "Follow @shipwithpip", font(F_BOLD, 80), al(TH["ink"], clamp((t - seg["start"]) / 0.3)))
    tc(d, (540, 1850), "@shipwithpip", font(F_MED, 34), al(TH["muted"], 0.8))
    d.rectangle([0, 0, W * clamp(t / total), 8], fill=al(YEL, 0.9))
    return img


def prepare(S, model_dir):
    from scale import patch_lexicon
    patch_lexicon(model_dir)
    S = json.loads(json.dumps(S))
    for b in S["beats"]:
        b["_cap"] = b["say"]
        if b.get("speak"): b["say"] = b["speak"]
    tl, vox, total = M.build_audio(S, model_dir)
    for b in S["beats"]: b["say"] = b.pop("_cap"); b.setdefault("cap_y", CAP_Y)
    return S, tl, vox, total


def render(S, out, model_dir, only=None):
    S, tl, vox, total = prepare(S, model_dir); bg = M.make_bg(TH)
    if only is not None:
        ts = only if only else [0.3, 2.0] + [x["start"] + (x["end"] - x["start"]) * 0.8 for x in tl]
        return [frame(t, S, tl, bg, total) for t in ts], tl, total
    audio = M.mix(vox, total, tl, [s["start"] for s in tl], sum(map(ord, S["id"])))
    wav = out[:-4] + ".wav"; sf.write(wav, audio, SR)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-af", "loudnorm=I=-14:TP=-1.2:LRA=11", "-c:a", "aac", "-b:a", "192k", "-shortest",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for fr in range(int(total * FPS)):
        if fr % 150 == 0: print(f"frame {fr} / {int(total * FPS)}", flush=True)
        ff.stdin.write(frame(fr / FPS, S, tl, bg, total).convert("RGB").tobytes())
    ff.stdin.close(); ff.wait(); os.remove(wav)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("script"); ap.add_argument("out", nargs="?")
    ap.add_argument("--model"); a = ap.parse_args(); S = json.load(open(a.script, encoding="utf-8"))
    if a.cmd == "stats":
        for k, val in stats(S).items(): print(f"{k:>16} = {val}")
    elif a.cmd == "check": sys.exit(0 if check(S) else 1)
    elif a.cmd == "sheet":
        frames, tl, total = render(S, a.out, a.model, only=[])
        sh = Image.new("RGB", (270 * 5, 480 * math.ceil(len(frames) / 5)))
        for i, f in enumerate(frames): sh.paste(f.convert("RGB").resize((270, 480)), ((i % 5) * 270, (i // 5) * 480))
        sh.save(a.out); print(a.out, f"{total:.1f}s")
    elif a.cmd == "render":
        total = render(S, a.out, a.model); print(f"done {a.out}: {total:.1f}s")
