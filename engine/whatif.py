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
             "vis": {"scene": stop|equator|bars|lat|daynight|slow|float|hammer|rain|jump|elevator|tunnel|bolt|dive|crash|hair|lift, ...}}],
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


# ============================================================ dive chain (Nov 21 2026, w9: one balloon going down, the pressure climbs every beat)
SEA_TOP = 430; SEA_BOT = 1470; OBJ = (560, 840)
WATER = [(0, (30, 110, 185)), (12, (22, 90, 165)), (1000, (8, 34, 78)), (3810, (4, 14, 38)), (11000, (2, 4, 14))]


def water_col(depth):
    for (d0, c0), (d1, c1) in zip(WATER, WATER[1:]):
        if depth <= d1:
            u = clamp((depth - d0) / max(d1 - d0, 1e-6)); return tuple(int(c0[i] + (c1[i] - c0[i]) * u) for i in range(3))
    return WATER[-1][1]


def draw_balloon(d, x, y, r, a=1.0):
    d.line([(x, y + r * 1.12), (x + 10, y + r * 1.12 + 90), (x - 6, y + r * 1.12 + 170)], fill=al((230, 230, 240), 0.8 * a), width=4, joint="curve")
    d.ellipse([x - r, y - r * 1.12, x + r, y + r * 1.12], fill=al(UMB, a))
    d.polygon([(x - 12, y + r * 1.12 + 14), (x + 12, y + r * 1.12 + 14), (x, y + r * 1.12 - 4)], fill=al(UMB_D, a))
    d.ellipse([x - r * 0.55, y - r * 0.8, x - r * 0.15, y - r * 0.2], fill=al(WHITE, 0.35 * a))


def draw_cup(d, x, y, s, a=1.0, ghost=False):
    w0, w1, h = 120 * s, 80 * s, 200 * s
    pts = [(x - w0, y - h / 2), (x + w0, y - h / 2), (x + w1, y + h / 2), (x - w1, y + h / 2)]
    if ghost: d.polygon(pts, outline=al(WHITE, 0.45 * a), width=4); return
    d.polygon(pts, fill=al((238, 238, 230), a)); d.ellipse([x - w0, y - h / 2 - 16 * s, x + w0, y - h / 2 + 16 * s], fill=al((205, 205, 198), a))
    for k, c in enumerate((UMB, CYAN, LIME)):
        yy = y - h / 2 + h * (0.3 + 0.2 * k); ww = w0 + (w1 - w0) * (0.3 + 0.2 * k)
        d.line([(x - ww * 0.8, yy), (x + ww * 0.8, yy)], fill=al(c, a), width=max(3, int(8 * s)))


def press_arrows(d, cx, cy, rad, level, a=1.0, phase=0.0):
    n = 12; L = 40 + 60 * level
    for k in range(n):
        an = 2 * math.pi * k / n + 0.13; wob = 8 * math.sin(phase * 6 + k)
        r1 = rad + 30 + L + wob; r0 = rad + 26 + wob
        x1, y1 = cx + math.cos(an) * r1, cy + math.sin(an) * r1; x0, y0 = cx + math.cos(an) * r0, cy + math.sin(an) * r0
        d.line([(x1, y1), (x0, y0)], fill=al(CYAN, 0.75 * a), width=int(5 + 6 * level))
        px, py = -math.sin(an), math.cos(an); hs = 14 + 10 * level
        d.polygon([(x0, y0), (x0 + math.cos(an) * hs + px * hs * 0.7, y0 + math.sin(an) * hs + py * hs * 0.7),
                   (x0 + math.cos(an) * hs - px * hs * 0.7, y0 + math.sin(an) * hs - py * hs * 0.7)], fill=al(CYAN, 0.85 * a))


def sc_dive(img, d, v, p, t, ctx):
    mode = v.get("mode", "balloon"); arr = v.get("arrive", 0.4); u = ease(clamp(p / arr)) if arr > 0 else 1.0; there = p >= arr
    d0, d1 = v.get("from", 0), v.get("depth", 0)
    dep = d0 + (d1 - d0) * u
    d.rectangle([0, SEA_TOP, W, SEA_BOT], fill=water_col(dep) + (255,))
    if dep < 40:  # surface line with sky above
        sy = SEA_TOP + 0 + dep * 6
        if sy < SEA_BOT: d.rectangle([0, SEA_TOP, W, min(SEA_BOT, sy)], fill=(14, 20, 40, 255)); d.line([(0, sy), (W, sy)], fill=al(WHITE, 0.6), width=4)
    # marine snow drifts upward while we sink
    rs = np.random.RandomState(5); spd = (0 if there else 900) + 30
    for _ in range(70):
        x = rs.rand() * W; y0 = rs.rand() * (SEA_BOT - SEA_TOP); r = 2 + rs.rand() * 4
        y = SEA_TOP + (y0 - t * spd * (0.5 + rs.rand())) % (SEA_BOT - SEA_TOP)
        d.ellipse([x - r, y - r, x + r, y + r], fill=al(WHITE, 0.25 + 0.2 * rs.rand()))
    lvl = clamp(math.log10(1 + dep / 10.06) / math.log10(1100))
    if mode in ("balloon", "cup", "hook", "deep"):
        cx, cy = OBJ
        if mode == "balloon":
            v0, v1 = v.get("vol0", 1.0), v.get("vol", 1.0); vol = v0 + (v1 - v0) * u; r = max(8, 150 * vol ** (1 / 3))
            if v.get("ghost", True) and v1 < 1: d.ellipse([cx - 150, cy - 168, cx + 150, cy + 168], outline=al(WHITE, 0.35), width=4)
            draw_balloon(d, cx, cy, r); rad = r * 1.12
        elif mode == "cup":
            s = 1.0 + (v.get("cup", 0.6) - 1.0) * ease(clamp((p - arr) / 0.3)) if there else 1.0
            draw_cup(d, cx, cy, 1.0, ghost=True); draw_cup(d, cx, cy + 100 * (1 - s), s); rad = 150 * s
        else:
            rad = 100; person(d, cx, cy + 85, 62, 0.0, col=YEL)  # you, at the bottom
        if dep > 1: press_arrows(d, cx, cy, rad, lvl, a=clamp(0.3 + lvl), phase=t)
    if mode in ("everest", "life"):
        FL = 1300; top = SEA_TOP + 40
        d.rectangle([0, FL, W, SEA_BOT], fill=(20, 16, 26, 255))
        if mode == "everest":
            fr = v.get("peak_frac", 0.8); g = ease(clamp(p / 0.5)); py = FL - (FL - top) * fr * g
            d.polygon([(120, FL), (430, py + 160 * g), (560, py), (700, py + 120 * g), (960, FL)], fill=(70, 74, 96, 255))
            d.polygon([(560, py), (520, py + 60 * g), (545, py + 50 * g), (600, py + 70 * g), (628, py + 60 * g)], fill=(235, 240, 250, 255))
            if p > 0.55:  # bracket: water still above the summit
                a = clamp((p - 0.55) / 0.1); bx = 800
                d.line([(bx, top), (bx, py)], fill=al(YEL, a), width=6); d.line([(bx - 20, top), (bx + 20, top)], fill=al(YEL, a), width=6); d.line([(bx - 20, py), (bx + 20, py)], fill=al(YEL, a), width=6)
            d.line([(0, top), (W, top)], fill=al(WHITE, 0.5), width=3)
        else:
            for k in range(3):  # small glowing amphipods hopping over the floor
                x = 300 + k * 230 + 40 * math.sin(t * 1.3 + k); y = FL - 40 - 30 * abs(math.sin(t * 2.1 + k * 1.7))
                g = Image.new("RGBA", (160, 160), (0, 0, 0, 0)); gd = ImageDraw.Draw(g); gd.ellipse([30, 30, 130, 130], fill=al(CYAN, 0.35)); g = g.filter(ImageFilter.GaussianBlur(18)); img.alpha_composite(g, (int(x - 80), int(y - 80)))
                d = ImageDraw.Draw(img, "RGBA")
                for j in range(5): d.ellipse([x - 34 + j * 12, y - 12 + 3 * abs(j - 2), x - 14 + j * 12, y + 8 + 3 * abs(j - 2)], fill=al((210, 240, 255), 1))
                for j in range(4): d.line([(x - 20 + j * 12, y + 10), (x - 26 + j * 12, y + 26)], fill=al((210, 240, 255), 0.9), width=3)
    # depth gauge: a bar on the left, the marker slides, the depth readout appears only on arrival (no tweened numbers)
    gx, gy0, gy1 = 70, SEA_TOP + 40, SEA_BOT - 190
    d.line([(gx, gy0), (gx, gy1)], fill=al(WHITE, 0.35), width=6)
    my = gy0 + (gy1 - gy0) * clamp(math.log10(1 + dep) / math.log10(11001)); d.ellipse([gx - 16, my - 16, gx + 16, my + 16], fill=al(YEL, 1))
    fade_top(img, SEA_TOP - 20, SEA_TOP + 60); fade_bottom(img, 1330, 1480); d = ImageDraw.Draw(img, "RGBA")
    if v.get("speed") and (there or v.get("speed_now")): bigtext(img, (540, 500), v["speed"], 64, CYAN, a=clamp((p - arr) / 0.08) if not v.get("speed_now") else 1, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
    va = v.get("value_at", arr)
    if "value" in v and p >= va:
        g = clamp((p - va) / 0.1)
        bigtext(img, (540, 1250), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 100), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a > 0: bigtext(img, (540, 1370 - (len(v["labels"]) - 1 - i) * 72 + (0 if "value" in v else -40)), lb, 54, WHITE if i else YEL, a=a)


SCENES["dive"] = sc_dive


# ============================================================ crash = fall (Nov 22 2026, w10): a car dropped from the height that matches its crash speed
CGY = 1240  # ground line
CTOP = 470  # highest drawable point (below the head band)
CAR_RED = (230, 70, 70)


def draw_car(d, x, y, s=1.0, ang=0.0, crush=0.0, col=CAR_RED):
    """Side-view car, wheels' bottom at (x, y), length about 220*s px. crush 0..1 squashes it on impact."""
    sy = 1 - 0.45 * crush
    def P(u, v):
        v = v * sy; return (x + (u * math.cos(ang) - v * math.sin(ang)) * s, y + (u * math.sin(ang) + v * math.cos(ang)) * s)
    d.polygon([P(-110, -22), P(110, -22), P(114, -58), P(60, -62), P(30, -98), P(-50, -98), P(-80, -62), P(-112, -58)], fill=al(col, 1))
    d.polygon([P(22, -66), P(-2, -92), P(-44, -92), P(-66, -66)], fill=al((150, 200, 240), 0.9))
    d.line([P(-110, -40), P(110, -40)], fill=al((120, 30, 30), 1), width=max(2, int(4 * s)))
    for wx in (-66, 66):
        cx, cy = P(wx, -20); r = 22 * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(18, 18, 26, 255)); d.ellipse([cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45], fill=al(STEEL, 1))
    hx, hy = P(108, -50); d.ellipse([hx - 7 * s, hy - 7 * s, hx + 7 * s, hy + 7 * s], fill=al(YEL, 1))


def draw_tower(d, x, w, top, col=(40, 46, 74), win=True, pxpm=1.0):
    d.rectangle([x, top, x + w, CGY], fill=col + (255,))
    if win:
        fh = max(3.0, 3.5 * pxpm)  # one window row per 3.5 m, drawing only
        if fh >= 6:
            y = top + fh * 0.4
            while y < CGY - fh * 0.5:
                for k in range(4): d.rectangle([x + 10 + k * (w - 20) / 4, y, x + 10 + k * (w - 20) / 4 + (w - 20) / 4 - 8, y + fh * 0.45], fill=(255, 220, 120, 60))
                y += fh


def draw_mark(img, d, kind, x, pxpm, a):
    """Landmark silhouette at the same scale: pyramid (147 m as built) or burj (828 m). Returns its top y."""
    c = al((120, 128, 170), 0.85 * a)
    if kind == "pyramid":
        hh = 147 * pxpm; half = 115 * pxpm  # base side 230 m
        d.polygon([(x - half, CGY), (x + half, CGY), (x, CGY - hh)], fill=c); return CGY - hh
    if kind == "burj":
        hh = 828 * pxpm; steps = [(0.0, 1.0), (0.25, 0.8), (0.45, 0.6), (0.62, 0.42), (0.76, 0.28), (0.86, 0.16), (0.93, 0.07)]
        bw = 70 * pxpm * 1.0 + 30
        for f0, wf in steps:
            y0 = CGY - hh * f0; y1 = CGY - hh * min(1.0, f0 + 0.3)
            d.rectangle([x - bw * wf / 2, y1, x + bw * wf / 2, y0], fill=c)
        d.line([(x, CGY - hh * 0.93), (x, CGY - hh)], fill=c, width=4); return CGY - hh
    return CGY


def sc_crash(img, d, v, p, t, ctx):
    """A car falls from h metres (no air) and lands at `land`. view = metres shown from ground to CTOP (camera scale).
    mark = pyramid | burj (landmark at same scale), tower = draw the drop building, value/unit = readout after landing."""
    h = float(v["h"]); view = float(v.get("view", h * 1.15)); pxpm = (CGY - CTOP) / view
    drop = v.get("drop", 0.12); land = v.get("land", 0.72)
    top_y = CGY - h * pxpm
    s = min(0.8, max(0.2, 1.8 * 4.6 * pxpm / 220))  # car drawn about 1.8x its real 4.6 m length so it stays visible
    landed = p >= land
    # ground and town
    d.rectangle([0, CGY, W, H], fill=(24, 28, 44, 255)); d.line([(0, CGY), (W, CGY)], fill=al(STEEL, 0.5), width=4)
    mk = v.get("mark")
    if mk:
        a = clamp((p - v.get("mark_at", 0.3)) / 0.12)
        if a > 0:
            my = draw_mark(img, d, mk, 870, pxpm, a)
            if v.get("mark_label"): bigtext(img, (870, max(CTOP - 10, my - 50)), v["mark_label"], 44, (180, 188, 230), a=a, outline=(5, 8, 20)); d = ImageDraw.Draw(img, "RGBA")
    if v.get("tower", True): draw_tower(d, 160, 220, top_y, pxpm=pxpm)
    # person for scale when the drop is short
    if h < 60: person(d, 70, CGY, max(6.0, 1.75 * pxpm / 2.7), 0.0, col=(200, 205, 225))
    # ruler from the ground to the drop height
    ra = clamp((p - 0.05) / 0.1)
    if ra > 0:
        d.line([(640, CGY), (640, top_y)], fill=al(YEL, 0.85 * ra), width=5)
        for y in (CGY, top_y): d.line([(616, y), (664, y)], fill=al(YEL, 0.85 * ra), width=5)
        if v.get("ruler"): bigtext(img, (640 + 20 + 22 * len(v["ruler"]), (CGY + top_y) / 2 if top_y < CGY - 120 else top_y - 50), v["ruler"], 52, YEL, a=ra, outline=(40, 30, 0)); d = ImageDraw.Draw(img, "RGBA")
    # the car: waits on the roof edge, then free falls (y grows with time squared) and lands at `land`
    cx = 400 + 120 * s
    if p < drop: cy = top_y; ang = 0.0
    elif not landed:
        q = (p - drop) / max(land - drop, 1e-3); cy = top_y + (CGY - top_y) * q * q; ang = 0.5 * q
        for k in range(1, 9):  # motion streaks
            q2 = max(0.0, q - k * 0.03); y2 = top_y + (CGY - top_y) * q2 * q2
            d.line([(cx - 60 * s, y2 - 40 * s), (cx + 60 * s, y2 - 40 * s)], fill=al(WHITE, 0.25 * (1 - k / 9)), width=3)
    else: cy = CGY; ang = 0.0
    crush = ease(clamp((p - land) / 0.05)) if landed else 0.0
    draw_car(d, cx, cy, s, ang, crush)
    if landed:
        qq = clamp((p - land) / 0.3); r = 60 + 380 * ease(qq)
        dust = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dd = ImageDraw.Draw(dust)
        dd.ellipse([cx - r, CGY - r * 0.4, cx + r, CGY + r * 0.4], fill=al((170, 150, 130), 0.7 * (1 - qq))); img.alpha_composite(dust.filter(ImageFilter.GaussianBlur(22)))
        if p - land < 0.03: img.alpha_composite(Image.new("RGBA", (W, H), al(WHITE, 0.3)))
        d = ImageDraw.Draw(img, "RGBA")
        if p - land < 0.08: bigtext(img, (cx, CGY - 170 * max(s, 0.5)), "CRUNCH", 80, RED, angle=-6, outline=(60, 0, 0)); d = ImageDraw.Draw(img, "RGBA")
    fade_top(img); d = ImageDraw.Draw(img, "RGBA")
    if v.get("speed"): bigtext(img, (540, 470), v["speed"], 60, CYAN, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
    at = v.get("value_at", land)
    if "value" in v and p >= at:
        g = clamp((p - at) / 0.12)
        bigtext(img, (540, 1340), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 104), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    y0 = 1430 if "value" in v else 1350
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a > 0: bigtext(img, (540, y0 - (len(v["labels"]) - 1 - i) * 76), lb, 58, WHITE if i else YEL, a=a)


SCENES["crash"] = sc_crash


# ============================================================ hair (Nov 23 2026, w11): loads hanging from hair, and a hair-pulled bus
HAIR = (214, 160, 96); HAIR_D = (150, 104, 58); SKIN = (236, 190, 150); GREY_E = (150, 156, 172)
HHX, HHY, HHR, KNOT = 540, 590, 92, 880


def draw_head(d, x, y, r):
    d.ellipse([x - r, y - r, x + r, y + r], fill=al(SKIN, 1))
    d.chord([x - r - 4, y - r - 6, x + r + 4, y + r * 0.7], 180, 360, fill=al(HAIR, 1))  # hair cap
    for ex in (-0.36, 0.36): d.ellipse([x + ex * r - 9, y + 0.12 * r - 9, x + ex * r + 9, y + 0.12 * r + 9], fill=(30, 24, 20, 255))
    d.arc([x - 0.4 * r, y + 0.2 * r, x + 0.4 * r, y + 0.62 * r], 20, 160, fill=(120, 50, 40, 255), width=6)


def draw_elephant(d, x, y, s=1.0, col=GREY_E):
    """Side view, feet on y, centred on x, about 260*s px long, facing right."""
    c = al(col, 1); dk = al(tuple(int(k * 0.75) for k in col), 1)
    for lx in (-80, -40, 40, 72): d.rounded_rectangle([x + (lx - 18) * s, y - 95 * s, x + (lx + 18) * s, y], radius=int(8 * s), fill=c)
    d.ellipse([x - 120 * s, y - 210 * s, x + 100 * s, y - 70 * s], fill=c)
    d.ellipse([x + 60 * s, y - 225 * s, x + 150 * s, y - 135 * s], fill=c)
    d.ellipse([x + 40 * s, y - 215 * s, x + 105 * s, y - 120 * s], fill=dk)  # ear
    d.line([(x + 140 * s, y - 165 * s), (x + 158 * s, y - 100 * s), (x + 150 * s, y - 45 * s)], fill=c, width=max(4, int(22 * s)), joint="curve")
    d.ellipse([x + 112 * s, y - 196 * s, x + 124 * s, y - 184 * s], fill=(20, 20, 28, 255))
    d.line([(x + 118 * s, y - 145 * s), (x + 150 * s, y - 140 * s)], fill=(240, 236, 220, 255), width=max(3, int(7 * s)))  # tusk
    d.line([(x - 118 * s, y - 160 * s), (x - 138 * s, y - 110 * s)], fill=c, width=max(2, int(6 * s)))  # tail


def draw_bus(d, x, y, s=1.0):
    """London-style double decker, side view, wheels' bottom at y, left end at x, about 520*s px long."""
    L, Hh = 520 * s, 300 * s
    d.rounded_rectangle([x, y - Hh, x + L, y - 26 * s], radius=int(18 * s), fill=(206, 30, 40, 255))
    for row in (0, 1):
        yy = y - Hh + 30 * s + row * 125 * s
        for k in range(5): d.rectangle([x + 24 * s + k * 98 * s, yy, x + 24 * s + k * 98 * s + 80 * s, yy + 60 * s], fill=(160, 210, 240, 235))
    d.rectangle([x, y - 152 * s, x + L, y - 140 * s], fill=(240, 220, 120, 255))
    for wx in (90, 420):
        r = 34 * s; cx, cy = x + wx * s, y - r
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(18, 18, 26, 255)); d.ellipse([cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45], fill=al(STEEL, 1))


def load_box(kind, n):
    """(width, height) of the hanging load in px."""
    return dict(bar=(210, 64), bag=(150, 170), person=(110, 290), elephant=(int(300 * max(1, math.ceil(n)) * 0.9) + 60, 205))[kind]


def sc_hair(img, d, v, p, t, ctx):
    """mode lift: `n` hairs from one head (n drawn as up to 70 strands) gather in a knot and lift a load off the floor at `hit`.
    load = bar | bag | person | elephant (count = elephants, fractions drawn cut off). mode pull: a person pulls a bus with their hair."""
    mode = v.get("mode", "lift"); hit = v.get("hit", 0.3)
    d.rectangle([0, CGY, W, H], fill=(24, 28, 44, 255)); d.line([(0, CGY), (W, CGY)], fill=al(STEEL, 0.5), width=4)
    if mode == "pull":
        q = ease(clamp((p - hit) / (1 - hit))); bx = 470 - 110 * q
        px = 170; s = 58; hy = CGY - 2.3 * s
        person(d, px, CGY, s, -0.28, col=SKIN)
        hx = px - math.sin(0.28) * 2.3 * s
        d.line([(hx + 8, hy - 8), (bx, CGY - 110)], fill=al(HAIR, 1), width=8)
        draw_bus(d, bx, CGY, 1.0)
        for k in range(5):  # motion lines behind the bus
            yy = CGY - 60 - k * 50; a = 0.35 * q
            d.line([(bx + 540, yy), (bx + 540 + 60 + 20 * k, yy)], fill=al(WHITE, a), width=4)
    else:
        n = float(v.get("n", 1)); kind = v.get("load", "bar"); cnt = float(v.get("count", 1))
        lift = 80 * ease(clamp((p - hit) / 0.25)) if p >= hit else 0.0
        bw, bh = load_box(kind, cnt)
        bot = CGY - lift; top = bot - bh; hook = top - 60
        ns = int(min(70, max(1, n)))
        taut = 1.0 if p >= hit else 0.0
        for i in range(ns):
            ang = math.pi * (1.02 + 0.96 * (i + 0.5) / ns) if ns > 1 else math.pi * 1.5  # roots on the hair cap; strands fall behind the head
            sx, sy = HHX + math.cos(ang) * HHR * 1.0, HHY + math.sin(ang) * HHR * 1.0
            sag = (1 - taut) * 30 * math.sin(i * 1.7)
            d.line([(sx, sy), ((sx + HHX) / 2 + sag, (sy + KNOT) / 2), (HHX, KNOT)], fill=al(HAIR if i % 3 else HAIR_D, 0.95 if ns < 10 else 0.8), width=4 if ns < 10 else 3, joint="curve")
        draw_head(d, HHX, HHY, HHR)
        d.ellipse([HHX - 14, KNOT - 14, HHX + 14, KNOT + 14], fill=al(HAIR_D, 1))
        d.line([(HHX, KNOT + 10), (HHX, hook)], fill=al(HAIR, 1), width=8 if ns > 1 else 4)
        d.arc([HHX - 22, hook - 6, HHX + 22, hook + 38], 0, 180, fill=al(STEEL, 1), width=8)
        if kind == "bar":
            d.line([(HHX, hook + 30), (HHX, top)], fill=al(STEEL, 0.9), width=3)
            d.rectangle([HHX - bw / 2, top, HHX + bw / 2, bot], fill=(110, 62, 34, 255)); d.rectangle([HHX - bw / 2 + 30, top, HHX - bw / 2 + 80, bot], fill=(230, 60, 70, 255))
            bigtext(img, (HHX + 30, (top + bot) / 2), "100 G", 40, WHITE); d = ImageDraw.Draw(img, "RGBA")
        elif kind == "bag":
            d.line([(HHX, hook + 30), (HHX, top)], fill=al(STEEL, 0.9), width=3)
            d.rounded_rectangle([HHX - bw / 2, top, HHX + bw / 2, bot], radius=14, fill=(238, 238, 245, 255))
            bigtext(img, (HHX, (top + bot) / 2), "1 KG", 40, (40, 60, 120)); d = ImageDraw.Draw(img, "RGBA")
        elif kind == "person":
            d.line([(HHX, hook + 30), (HHX, top + 10)], fill=al(STEEL, 0.9), width=3)
            person(d, HHX, bot, 100, 0.0, col=(200, 205, 225))
        else:
            full = int(math.floor(cnt + 1e-9)); frac = cnt - full; k = full + (1 if frac > 1e-6 else 0)
            gap = 275; xs = [HHX + (j - (k - 1) / 2) * gap for j in range(k)]
            bar_y = hook + 50
            d.line([(xs[0] - 10, bar_y), (xs[-1] + 10, bar_y)], fill=al(STEEL, 1), width=10)
            d.line([(HHX, hook + 30), (HHX, bar_y)], fill=al(STEEL, 1), width=6)
            for j, x in enumerate(xs):
                d.line([(x, bar_y), (x, top + 30)], fill=al(STEEL, 0.9), width=4)
                if j < full: draw_elephant(d, x - 15, bot, 0.9)
                else:
                    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay, "RGBA"); draw_elephant(ld, x - 15, bot, 0.9)
                    x0 = x - 15 - 0.9 * 140; x1 = x0 + 0.9 * 300 * frac
                    lay = lay.crop((int(x0), 0, int(x1), H)); img.alpha_composite(lay, (int(x0), 0)); d = ImageDraw.Draw(img, "RGBA")
                    d.line([(x1, bot - 215), (x1, bot)], fill=al(RED, 0.9), width=4)
        if p >= hit and p - hit < 0.08: bigtext(img, (HHX + 260, KNOT - 40), "TAUT", 56, CYAN, angle=-8, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
    fade_top(img); d = ImageDraw.Draw(img, "RGBA")
    at = v.get("value_at", hit + 0.05)
    if "value" in v and p >= at:
        g = clamp((p - at) / 0.12)
        bigtext(img, (540, 1340), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 104), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    y0 = 1430 if "value" in v else 1350
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a > 0: bigtext(img, (540, y0 - (len(v["labels"]) - 1 - i) * 76), lb, 58, WHITE if i else YEL, a=a)


SCENES["hair"] = sc_hair


# ============================================================ facts + check
WORDNUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
           "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


# ============================================================ lift chain (Nov 23 2026, w12): food energy turned into height
BAN = (255, 222, 80); BAN_D = (196, 150, 30)


def draw_banana(d, cx, cy, s=1.0, a=1.0):
    """A curved banana built from overlapping discs along an arc; s = scale (1 = about 300 px long)."""
    pts = []
    for k in range(41):
        q = k / 40; ang = math.radians(200 + 140 * q); r = 150 * s
        x = cx + r * math.cos(ang); y = cy - 60 * s + r * math.sin(ang) * -0.55 + 40 * s
        w = (10 + 28 * math.sin(math.pi * q) ** 0.7) * s; pts.append((x, y, w))
    for x, y, w in pts: d.ellipse([x - w, y - w, x + w, y + w], fill=al(BAN_D, a))
    for x, y, w in pts: d.ellipse([x - w * 0.82, y - w * 0.95, x + w * 0.82, y + w * 0.6], fill=al(BAN, a))
    x, y, w = pts[0]; d.rectangle([x - 9 * s, y - 6 * s, x + 4 * s, y + 6 * s], fill=al((90, 70, 30), a))
    x, y, w = pts[-1]; d.ellipse([x - 7 * s, y - 7 * s, x + 7 * s, y + 7 * s], fill=al((70, 50, 20), a))


def draw_ref(d, kind, x, base, hpx, label, a):
    """A landmark to scale: tower (spire), mountain (two-tone peak), pyramid; hpx = height in px."""
    c = al((70, 84, 120), a); c2 = al((110, 126, 170), a)
    if kind == "tower":
        w = max(10.0, hpx * 0.09)
        d.polygon([(x - w, base), (x + w, base), (x + w * 0.35, base - hpx * 0.72), (x, base - hpx), (x - w * 0.35, base - hpx * 0.72)], fill=c)
        d.line([(x, base), (x, base - hpx)], fill=c2, width=3)
    elif kind == "mountain":
        w = hpx * 0.75
        d.polygon([(x - w, base), (x + w, base), (x, base - hpx)], fill=c)
        d.polygon([(x - w * 0.22, base - hpx * 0.78), (x, base - hpx), (x + w * 0.22, base - hpx * 0.78), (x + w * 0.05, base - hpx * 0.72)], fill=al(WHITE, 0.85 * a))
    else:
        w = hpx * 0.78
        d.polygon([(x - w, base), (x + w, base), (x, base - hpx)], fill=al((196, 160, 96), a))
        d.polygon([(x, base - hpx), (x + w, base), (x + w * 0.25, base)], fill=al((150, 118, 66), a))
    if label: d.text((x, base - hpx - 30), label, font=font(F_BOLD, 34), fill=al(WHITE, a), anchor="mm")


def sc_lift(img, d, v, p, t, ctx):
    """All the energy goes into lifting you: you rise from the ground to height h (m) and stay there.
    mode lift (default) | glow (a banana glowing with its mass energy) | crowd (many people rise together).
    refs: [{"kind": tower|mountain|pyramid, "h": metres, "label": NAME}] drawn to the same scale."""
    mode = v.get("mode", "lift"); draw_stars(d, a=0.45)
    if mode == "glow":
        q = clamp((p - 0.15) / 0.5); cx, cy = 540, 860
        for k in range(8, 0, -1):
            r = (120 + 26 * k) * (0.5 + 0.5 * ease(q)); d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=al((255, 236, 150), 0.035 * q))
        for k in range(16):  # rays
            ang = k * math.pi / 8 + t * 0.4; r0, r1 = 190, 190 + 110 * ease(q)
            d.line([(cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)), (cx + r1 * math.cos(ang), cy + r1 * math.sin(ang))], fill=al((255, 236, 150), 0.6 * q), width=6)
        draw_banana(d, cx, cy, 1.1)
    else:
        h = float(v.get("h", 100)); top_m = max([h] + [r["h"] for r in v.get("refs", [])])
        pxpm = 620.0 / top_m
        d.rectangle([0, JGY, W, H], fill=al((38, 78, 50), 1)); d.line([(0, JGY), (W, JGY)], fill=al((58, 112, 72), 1), width=6)
        for i, r in enumerate(v.get("refs", [])):
            a = clamp((p - 0.05 - 0.1 * i) / 0.12)
            if a > 0: draw_ref(d, r.get("kind", "tower"), r.get("x", 300), JGY, r["h"] * pxpm, r.get("label", ""), a)
        u = ease(clamp((p - 0.12) / 0.6)); fy = JGY - h * pxpm * u
        if u < 0.08: draw_banana(d, 470, JGY - 30, 0.45, 1 - u / 0.08)
        if mode == "crowd":
            rs = np.random.RandomState(3)
            for _ in range(90):
                x = 380 + rs.rand() * 360; y = fy - rs.rand() * 70
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=al(CYAN if rs.rand() < 0.5 else WHITE, 0.9))
        else:
            for k in range(1, 10):
                y2 = JGY - h * pxpm * ease(clamp((p - 0.12 - k * 0.02) / 0.6))
                if 0.02 < u < 0.99: d.ellipse([600 - 7, y2 - 7, 600 + 7, y2 + 7], fill=al(CYAN, 0.4 * (1 - k / 10)))
            s = 1.75 * pxpm / 2.7
            if s >= 8: person(d, 600, fy, s, 0.0)
            else:
                d.ellipse([600 - 22, fy - 22, 600 + 22, fy + 22], fill=al(CYAN, 0.25)); d.ellipse([600 - 8, fy - 8, 600 + 8, fy + 8], fill=al(WHITE, 1))
                bigtext(img, (600 + 92, fy - 10), "YOU", 40, WHITE, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
        a = clamp((p - 0.3) / 0.1)
        if a > 0:
            ty = JGY - h * pxpm; d.line([(860, JGY), (860, ty)], fill=al(YEL, 0.8 * a), width=5)
            for y in (JGY, ty): d.line([(836, y), (884, y)], fill=al(YEL, 0.8 * a), width=5)
        fade_bottom(img, 1230, 1440); fade_top(img); d = ImageDraw.Draw(img, "RGBA")
    at = v.get("value_at", 0.5)
    if "value" in v and p >= at:
        g = clamp((p - at) / 0.12)
        bigtext(img, (540, 1340), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 100), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    y0 = 1430 if "value" in v else 1350
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a > 0: bigtext(img, (540, y0 - (len(v["labels"]) - 1 - i) * 76), lb, 58, WHITE if i else YEL, a=a)


SCENES["lift"] = sc_lift


# ============================================================ coin chain (Nov 24 2026, w13): a penny dropped with no air, then faster and faster
COPPER = (196, 118, 62); COPPER_D = (132, 74, 36); COPPER_L = (240, 178, 120); FIRE = (255, 150, 50)


def draw_penny(d, x, y, r, spin=0.0, a=1.0, glow=0.0):
    """A penny seen edge-on as it tumbles: width follows |cos(spin)|. glow 0..1 heats it toward orange-white."""
    if glow > 0:
        for k in range(6, 0, -1):
            rr = r * (1 + 0.5 * k * glow); d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=al(FIRE, 0.07 * glow * a))
    w = max(0.18, abs(math.cos(spin))) * r
    col = tuple(int(COPPER[i] + (255 - COPPER[i]) * 0.6 * glow) for i in range(3))
    d.ellipse([x - w, y - r, x + w, y + r], fill=al(COPPER_D, a))
    d.ellipse([x - w * 0.84, y - r * 0.86, x + w * 0.84, y + r * 0.86], fill=al(col, a))
    if w > r * 0.45: d.ellipse([x - w * 0.5, y - r * 0.55, x - w * 0.1, y - r * 0.1], fill=al(COPPER_L, 0.6 * a))


def draw_limb(img, d, top=1150, a=1.0):
    """Earth's curved edge seen from space, with a thin blue atmosphere glow."""
    R = 1700; cx, cy = 540, top + R
    for k in range(10, 0, -1):
        rr = R + 6 * k; d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=al((90, 170, 255), 0.05 * a))
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=al((34, 82, 160), a))
    rs = np.random.RandomState(8)
    for _ in range(14):
        x = rs.rand() * W; y = top + 40 + rs.rand() * 300; rx = 60 + rs.rand() * 140
        d.ellipse([x - rx, y - rx * 0.3, x + rx, y + rx * 0.3], fill=al((58, 112, 72), 0.8 * a))


def impact(img, x, y, q, big=1.0, col=FIRE):
    """Flash and expanding ring at (x, y); q 0..1 after impact."""
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
    r0 = 60 + 300 * big * ease(clamp(q / 0.5))
    od.ellipse([x - r0, y - r0, x + r0, y + r0], fill=al((255, 244, 210), 0.8 * clamp(1 - q / 0.6)))
    r1 = 80 + 900 * big * ease(q)
    od.ellipse([x - r1, y - r1 * 0.45, x + r1, y + r1 * 0.45], outline=al(col, 0.9 * (1 - q)), width=int(10 + 10 * big))
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(4)))
    if q < 0.25 * big: img.alpha_composite(Image.new("RGBA", (W, H), al(WHITE, 0.5 * (1 - q / (0.25 * big)))))


def sc_coin(img, d, v, p, t, ctx):
    """mode drop: a penny falls from a tower of h metres (no air), towers = extra ghost heights; land = beat fraction of impact.
    mode space / meteor: a penny streaks in from space to Earth's edge (meteor = on fire, bigger flash).
    mode car: penny = car (the same energy); mode cars: n cars appear in a row; mode real: the penny flutters down and taps a person."""
    mode = v.get("mode", "drop"); land = v.get("land", 0.7)
    if mode == "drop":
        hs = [float(v["h"])] + [float(x) for x in v.get("towers", [])]; top_m = max(hs); pxpm = (CGY - CTOP) / (top_m * 1.08)
        d.rectangle([0, CGY, W, H], fill=(22, 24, 38, 255)); d.line([(0, CGY), (W, CGY)], fill=al(TH["muted"], 0.6), width=4)
        h = hs[0]; tt = CGY - h * pxpm
        draw_tower(d, 150, 180, tt, pxpm=pxpm); d.line([(240, tt), (240, tt - 40)], fill=al(STEEL, 1), width=4)
        for k, hh in enumerate(sorted(hs[1:], reverse=True)):  # the stacked buildings: one outline per building
            a = clamp((p - 0.05) / 0.15); d.rectangle([150, CGY - hh * pxpm, 330, CGY], outline=al(CYAN, 0.9 * a), width=5)
            d.line([(150, CGY - hh * pxpm), (330, CGY - hh * pxpm)], fill=al(CYAN, 0.9 * a), width=6)
        x = 410; u = clamp((p - 0.1) / (land - 0.1)); y = tt + (CGY - 12 - tt) * u * u
        for k in range(1, 12):
            u2 = max(0.0, u - k * 0.015); y2 = tt + (CGY - 12 - tt) * u2 * u2
            if 0 < u < 1: d.ellipse([x - 9, y2 - 9, x + 9, y2 + 9], fill=al(COPPER_L, 0.35 * (1 - k / 12)))
        if u < 1: draw_penny(d, x, y, 18, spin=t * 9)
        else:
            q = clamp((p - land) / 0.25); draw_penny(d, x, CGY - 6, 18, spin=1.45)
            for j in range(9):
                an = -math.pi * (0.08 + 0.84 * j / 8); rr = 20 + 90 * ease(q)
                d.line([(x, CGY), (x + math.cos(an) * rr, CGY + math.sin(an) * rr)], fill=al((255, 230, 160), 1 - q), width=4)
        person(d, 640, CGY, 26, 0.0)
        a = clamp((p - 0.2) / 0.1)
        if a > 0:
            d.line([(880, CGY), (880, tt)], fill=al(YEL, 0.8 * a), width=5)
            for yy in (CGY, tt): d.line([(856, yy), (904, yy)], fill=al(YEL, 0.8 * a), width=5)
            if v.get("axis"): bigtext(img, (950, (CGY + tt) / 2), v["axis"], 44, YEL, angle=90, a=a, outline=(40, 30, 0)); d = ImageDraw.Draw(img, "RGBA")
    elif mode in ("space", "meteor"):
        draw_stars(d, n=160, a=0.8); draw_limb(img, d); d = ImageDraw.Draw(img, "RGBA")
        hot = mode == "meteor"; x0, y0, x1, y1 = 980, 450, 470, 1180
        u = ease(clamp((p - 0.08) / (land - 0.08))) if not hot else clamp((p - 0.08) / (land - 0.08)) ** 1.6
        x = x0 + (x1 - x0) * u; y = y0 + (y1 - y0) * u
        if u < 1:
            tl = 0.25 if not hot else 0.4
            for k in range(30):
                f = k / 30; xx = x - (x1 - x0) * tl * f * u; yy = y - (y1 - y0) * tl * f * u
                rr = (10 if not hot else 26) * (1 - f) + 2
                d.ellipse([xx - rr, yy - rr, xx + rr, yy + rr], fill=al(FIRE if hot else CYAN, (0.55 if hot else 0.35) * (1 - f)))
            draw_penny(d, x, y, 16 if not hot else 20, spin=t * 14, glow=1.0 if hot else 0.3 * u)
        else:
            q = clamp((p - land) / 0.3); impact(img, x1, y1, q, big=1.3 if hot else 0.6, col=FIRE if hot else (255, 230, 160)); d = ImageDraw.Draw(img, "RGBA")
        fade_bottom(img, 1250, 1440); d = ImageDraw.Draw(img, "RGBA")
    elif mode == "car":
        d.rectangle([0, CGY, W, H], fill=(22, 24, 38, 255)); d.line([(0, CGY), (W, CGY)], fill=al(TH["muted"], 0.6), width=4)
        draw_penny(d, 240, 860, 70, spin=0.0, glow=0.25)
        bigtext(img, (240, 990), "1 PENNY", 44, WHITE); bigtext(img, (450, 860), "=", 120, YEL, outline=(40, 30, 0)); d = ImageDraw.Draw(img, "RGBA")
        a = clamp((p - 0.15) / 0.15); cx = 640 + 200 * ease(clamp((p - 0.15) / 0.6))
        for k in range(6):
            yy = CGY - 30 - k * 14; d.line([(cx - 260 - 40 * (k % 3), yy), (cx - 130, yy)], fill=al(WHITE, 0.4 * a), width=3)
        if a > 0: draw_car(d, cx, CGY, 1.15, col=CAR_RED)
        if v.get("speed") and a > 0: bigtext(img, (cx, CGY - 170), v["speed"], 52, CYAN, a=a, outline=(5, 12, 24)); d = ImageDraw.Draw(img, "RGBA")
    elif mode == "cars":
        d.rectangle([0, CGY, W, H], fill=(22, 24, 38, 255)); d.line([(0, CGY), (W, CGY)], fill=al(TH["muted"], 0.6), width=4)
        draw_penny(d, 540, 620, 60, spin=t * 2, glow=1.0)
        n = int(v.get("n", 10)); cols = 4
        for i in range(n):
            a = clamp((p - 0.1 - i * 0.04) / 0.06)
            if a <= 0: continue
            r_, c_ = divmod(i, cols); x = 160 + c_ * 250 + (60 if r_ % 2 else 0); y = 860 + r_ * 130
            draw_car(d, x, y, 0.75, col=CAR_RED if i % 2 == 0 else (220, 110, 70))
    else:  # real: flutter down, tap the head
        d.rectangle([0, CGY, W, H], fill=(22, 24, 38, 255)); d.line([(0, CGY), (W, CGY)], fill=al(TH["muted"], 0.6), width=4)
        person(d, 540, CGY, 74, 0.0)
        headY = CGY - 74 * 2.68; u = clamp((p - 0.05) / (land - 0.05))
        x = 540 + 120 * math.sin(u * 9) * (1 - u); y = CTOP + (headY - CTOP) * u
        if u < 1: draw_penny(d, x, y, 22, spin=t * 3 + math.sin(t * 5))
        else:
            q = clamp((p - land) / 0.3); bx = 540 + 220 * q; by = headY - 120 * math.sin(math.pi * min(q * 1.2, 1)) + 600 * q * q
            draw_penny(d, bx, min(by, CGY - 8), 22, spin=t * 6)
            d.arc([540 - 40, headY - 30, 540 + 40, headY + 10], 200, 340, fill=al(YEL, 1 - q), width=4)
    at = v.get("value_at", 0.5)
    if "value" in v and p >= at:
        g = clamp((p - at) / 0.12)
        bigtext(img, (540, 1340), fmt_num(v["value"], v.get("dec", 0)) + " " + v.get("unit", ""), v.get("size", 100), YEL, a=g, scale=1 + 0.4 * (1 - ease(g)), outline=(40, 30, 0))
    y0 = 1430 if "value" in v else 1350
    for i, lb in enumerate(v.get("labels", [])):
        a = clamp((p - v.get("label_at", 0.55) - i * 0.12) / 0.08)
        if a > 0: bigtext(img, (540, y0 - (len(v["labels"]) - 1 - i) * 76), lb, 58, WHITE if i else YEL, a=a)


SCENES["coin"] = sc_coin


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
    out += [r.get("label", "") for r in v.get("refs", [])]
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
        if b.get("card"): chk(f"beat {i} card", b["card"])
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
    if b.get("card"):  # end card: dim the scene so the series card reads cleanly
        ov = Image.new("RGBA", (W, H), TH["bg"] + (int(225 * clamp((t - seg["start"]) / 0.3)),)); img.alpha_composite(ov)
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
        g = clamp((t - seg["start"]) / 0.3)
        if b.get("card"):  # series card (section F): episode name over the comment question
            tc(d, (540, 700), b["card"], font(F_BOLD, 110), al(YEL, g))
            fq = font(F_BOLD, 64); words = b["say"].split(); lines = [""]
            for w_ in words:
                if d.textlength((lines[-1] + " " + w_).strip(), font=fq) > 940: lines.append(w_)
                else: lines[-1] = (lines[-1] + " " + w_).strip()
            for k, ln in enumerate(lines): tc(d, (540, 880 + k * 80), ln, fq, al(TH["ink"], g))
        tc(d, (540, CAP_Y), "Follow @shipwithpip", font(F_BOLD, 80), al(TH["ink"], g))
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
