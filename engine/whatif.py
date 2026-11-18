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
             "vis": {"scene": stop|equator|bars|lat|daynight|slow|float|bignum, ...}}],
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
    if vis.get("scene"): SCENES[vis["scene"]](img, d, vis, p, t, {})
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
