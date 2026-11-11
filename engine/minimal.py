"""Minimal-style reels (light/dark UI mockups, keyword selection captions, prompt at the end).

usage: python minimal.py script.json out.mp4 --model <kokoro dir>
"""
import argparse
import json
import math
import os
import re
import subprocess

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import voice
from make_reel import music_bed, sfx_pop, sfx_whoosh, _envelope

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
W, H, FPS, SR = 1080, 1920, 30, 24000
GAP = 0.12
VOICE, SPEED = "am_michael", 1.12
MAX_LEN = 58.0

F_REG = os.path.join(A, "InterDisplay-Regular.otf")
F_MED = os.path.join(A, "InterDisplay-Medium.otf")
F_SEMI = os.path.join(A, "InterDisplay-SemiBold.otf")
F_BOLD = os.path.join(A, "InterDisplay-Bold.otf")
F_MONO = os.path.join(A, "JetBrainsMono-400.ttf")
F_MONO_B = os.path.join(A, "JetBrainsMono-700.ttf")

BLUE = (37, 99, 235)
RED = (239, 68, 68)
GREEN = (34, 197, 94)
TEAL = (45, 212, 191)
ORANGE = (249, 115, 22)
AMBER = (245, 158, 11)
CX = 540
VY0, VY1 = 360, 1250  # visual zone
VCY = (VY0 + VY1) // 2

THEMES = {
    "light": {"bg": (247, 247, 249), "grid": (236, 236, 241), "ink": (17, 17, 20), "muted": (118, 118, 128),
              "dim": (178, 178, 186), "card": (255, 255, 255), "field": (242, 242, 246), "line": (226, 226, 232),
              "shadow": 46, "pill": (255, 255, 255), "pillfill": (255, 237, 228)},
    "volt": {"bg": (22, 10, 56), "grid": (34, 20, 78), "ink": (255, 250, 235), "muted": (178, 166, 222),
             "dim": (110, 96, 160), "card": (38, 22, 90), "field": (50, 32, 112), "line": (74, 52, 150),
             "shadow": 140, "pill": (40, 24, 94), "pillfill": (90, 60, 40)},
    "dark": {"bg": (11, 11, 13), "grid": (22, 22, 26), "ink": (246, 246, 248), "muted": (142, 142, 152),
             "dim": (88, 88, 98), "card": (24, 24, 28), "field": (36, 36, 42), "line": (48, 48, 56),
             "shadow": 120, "pill": (28, 28, 32), "pillfill": (64, 40, 34)},
}

_fonts = {}


def font(p, s):
    k = (p, int(s))
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(p, max(int(s), 6))
    return _fonts[k]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x)
    return 3 * x * x - 2 * x * x * x


def back(x):
    x = clamp(x)
    c = 1.6
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def st(p, a, b):
    return ease((p - a) / max(b - a, 1e-6))


def lerp(a, b, f):
    return a + (b - a) * f


def mixc(a, b, f):
    return tuple(int(lerp(x, y, clamp(f))) for x, y in zip(a, b))


def al(c, a):
    return tuple(c[:3]) + (int(255 * clamp(a)),)


# ------------------------------------------------------------------ primitives
_shadow_cache = {}


def shadow(cv, box, r, th, a=1.0, spread=34, dy=22):
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0 or a <= 0:
        return
    k = (w, h, r, spread)
    if k not in _shadow_cache:
        pad = spread * 2
        m = Image.new("L", (w + pad * 2, h + pad * 2), 0)
        ImageDraw.Draw(m).rounded_rectangle([pad, pad, pad + w, pad + h], radius=r, fill=255)
        _shadow_cache[k] = (m.filter(ImageFilter.GaussianBlur(spread * 0.6)), pad)
    m, pad = _shadow_cache[k]
    m2 = m.point(lambda v: int(v * th["shadow"] / 255 * a))
    cv.paste((0, 0, 0), (x0 - pad, y0 - pad + dy), m2)


def card(cv, d, box, th, a=1.0, r=40, fill=None, border=True, sh=True):
    if sh:
        shadow(cv, box, r, th, a)
    d.rounded_rectangle(box, radius=r, fill=al(fill or th["card"], a),
                        outline=al(th["line"], a) if border else None, width=2)


def tc(d, xy, s, f, fill, anchor="mm"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def wrap(d, s, f, maxw):
    out, cur = [], ""
    for w in s.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= maxw:
            cur = t
        else:
            if cur:
                out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def cursor(d, x, y, click=0.0, s=1.0):
    pts = [(0, 0), (0, 44), (11, 33), (19, 51), (27, 47), (19, 30), (34, 30)]
    if click > 0:
        r = 26 + 30 * click
        d.ellipse([x - r, y - r, x + r, y + r], outline=(37, 99, 235, int(160 * (1 - click))), width=4)
    P = [(x + px * s, y + py * s) for px, py in pts]
    d.polygon([(px + 2, py + 3) for px, py in P], fill=(0, 0, 0, 60))
    d.polygon(P, fill=(12, 12, 14, 255), outline=(255, 255, 255, 255), width=3)


def pill(d, x, y, text, col, a=1.0, f=None, solid=True, anchor="m"):
    f = f or font(F_SEMI, 30)
    tw = d.textlength(text, font=f)
    w, h = tw + 40, 52
    x0 = x - w / 2 if anchor == "m" else (x - w if anchor == "r" else x)
    if solid:
        d.rounded_rectangle([x0, y - h / 2, x0 + w, y + h / 2], radius=26, fill=al(col, 0.14 * a), outline=al(col, a), width=2)
    tc(d, (x0 + w / 2, y), text, f, al(col, a))
    return w


def dot_icon(d, x, y, r, col, a=1.0, letter=None):
    d.rounded_rectangle([x - r, y - r, x + r, y + r], radius=int(r * 0.45), fill=al(col, a))
    if letter:
        tc(d, (x, y), letter, font(F_BOLD, r * 1.05), al((255, 255, 255), a))


# ------------------------------------------------------------------ syntax colouring
def code_tokens(line, dark=True):
    """Very small highlighter: returns [(text, colour)]."""
    base = (220, 220, 228) if dark else (30, 30, 36)
    pal = {"str": (134, 239, 172), "tag": (244, 114, 182), "key": (125, 211, 252), "com": (120, 120, 132),
           "num": (253, 186, 116)}
    if line.strip().startswith(("#", "//", "$")):
        return [(line, pal["com"] if not line.strip().startswith("$") else (230, 230, 236))]
    out = []
    for m in re.finditer(r'("[^"]*"|\'[^\']*\'|</?[a-zA-Z0-9_-]+|/?>|\b[A-Z_][A-Z0-9_]{2,}\b|\b\d[\d,\.]*\b|\w+(?=\s*[:=])|.)', line):
        s = m.group(0)
        if s.startswith(('"', "'")):
            c = pal["str"]
        elif s.startswith("<") or s in (">", "/>"):
            c = pal["tag"]
        elif re.match(r"^[A-Z_][A-Z0-9_]{2,}$", s):
            c = pal["key"]
        elif re.match(r"^\d", s):
            c = pal["num"]
        elif re.match(r"^\w+$", s) and re.search(re.escape(s) + r"\s*[:=]", line):
            c = pal["key"]
        else:
            c = base
        if out and out[-1][1] == c:
            out[-1] = (out[-1][0] + s, c)
        else:
            out.append((s, c))
    return out


# ------------------------------------------------------------------ scenes
# fn(cv, d, prm, p, t, th, ctx)

def sc_browser(cv, d, prm, p, t, th, ctx):
    w, h = 900, 660
    x0, y0 = CX - w // 2, VCY - h // 2 + 10
    a = st(p, 0, 0.1)
    card(cv, d, [x0, y0, x0 + w, y0 + h], th, a, r=34)
    for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([x0 + 30 + k * 30, y0 + 28, x0 + 46 + k * 30, y0 + 44], fill=al(c, a))
    d.rounded_rectangle([x0 + 140, y0 + 18, x0 + w - 40, y0 + 54], radius=18, fill=al(th["field"], a))
    tc(d, (x0 + 162, y0 + 36), prm.get("url", "yourapp.com"), font(F_MED, 24), al(th["muted"], a), "lm")
    blank = st(p, prm["blank_at"], prm["blank_at"] + 0.12) if "blank_at" in prm else 0
    ca = a * (1 - blank)
    ny = y0 + 110
    d.rounded_rectangle([x0 + 40, ny - 16, x0 + 72, ny + 16], radius=8, fill=al(th["ink"], ca))
    tc(d, (x0 + 86, ny), "yourapp", font(F_BOLD, 30), al(th["ink"], ca), "lm")
    for k, lk in enumerate(["Product", "Pricing", "Docs"]):
        tc(d, (x0 + 280 + k * 120, ny), lk, font(F_MED, 26), al(th["muted"], ca), "lm")
    fh = font(F_BOLD, 66)
    for k, ln in enumerate(prm.get("hero", ["Ship your idea", "this weekend."])):
        tc(d, (x0 + 40, y0 + 230 + k * 76), ln, fh, al(th["ink"], ca), "lm")
    for k, ln in enumerate(["The AI-native way to build, launch", "and grow. No code. No waiting."]):
        tc(d, (x0 + 40, y0 + 390 + k * 36), ln, font(F_REG, 27), al(th["muted"], ca), "lm")
    d.rounded_rectangle([x0 + 40, y0 + 470, x0 + 300, y0 + 540], radius=16, fill=al(th["ink"], ca))
    tc(d, (x0 + 170, y0 + 505), "Join the waitlist", font(F_SEMI, 27), al(th["card"], ca))
    d.rounded_rectangle([x0 + 560, y0 + 190, x0 + w - 40, y0 + 560], radius=22, fill=al(th["field"], ca))
    d.polygon([(x0 + 600, y0 + 520), (x0 + 690, y0 + 380), (x0 + 780, y0 + 520)], fill=al(th["line"], ca))
    if blank > 0:
        for k in range(6):
            yy = y0 + 110 + k * 72
            ww = [260, 640, 520, 700, 300, 420][k]
            d.rounded_rectangle([x0 + 40, yy, x0 + 40 + ww, yy + 26], radius=13, fill=al(th["field"], blank * a))
        tc(d, (CX, y0 + h - 70), prm.get("blank_label", "<div id=\"root\"></div>"), font(F_MONO, 32), al(RED, blank * a))
        pill(d, x0 + w - 40, y0 - 40, prm.get("badge", "What AI crawlers see"), RED, blank * a, anchor="r")


def sc_code(cv, d, prm, p, t, th, ctx):
    lines = prm["lines"]
    fs = prm.get("size", 35)
    fm = font(F_MONO, fs)
    lh = int(fs * 1.62)
    w = prm.get("w", 980)
    h = 120 + lh * len(lines) + 40
    x0, y0 = CX - w // 2, int(VCY - h / 2 + prm.get("dy", 0))
    a = st(p, 0, 0.1)
    dark = {"bg": (18, 18, 22)}
    shadow(cv, [x0, y0, x0 + w, y0 + h], 30, th, a)
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=30, fill=al(dark["bg"], a), outline=al((48, 48, 56), a), width=2)
    for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([x0 + 28 + k * 28, y0 + 30, x0 + 44 + k * 28, y0 + 46], fill=al(c, a))
    tc(d, (CX, y0 + 38), prm.get("title", ""), font(F_MED, 26), al((150, 150, 160), a))
    d.line([(x0, y0 + 76), (x0 + w, y0 + 76)], fill=al((40, 40, 48), a), width=2)
    hl = {int(k): v for k, v in prm.get("hl", {}).items()}
    typing = prm.get("typing", True)
    total_chars = sum(len(l) for l in lines)
    shown = int(total_chars * st(p, 0.05, prm.get("type_end", 0.55))) if typing else total_chars
    used = 0
    for i, ln in enumerate(lines):
        y = y0 + 110 + i * lh
        if i in hl:
            col = {"red": RED, "green": GREEN, "amber": AMBER, "blue": BLUE}[hl[i]]
            g = st(p, prm.get("hl_at", 0.5), prm.get("hl_at", 0.5) + 0.1)
            d.rounded_rectangle([x0 + 18, y - lh / 2 + 4, x0 + w - 18, y + lh / 2 - 4], radius=10,
                                fill=al(col, 0.18 * g * a), outline=al(col, 0.8 * g * a), width=2)
        n = clamp(shown - used, 0, len(ln))
        used += len(ln)
        x = x0 + 40
        rem = int(n)
        for txt, col in code_tokens(ln):
            if rem <= 0:
                break
            part = txt[:rem]
            rem -= len(part)
            d.text((x, y), part, font=fm, fill=al(col, a), anchor="lm")
            x += d.textlength(part, font=fm)
        if typing and 0 < n < len(ln) and int(t * 3) % 2 == 0:
            d.rectangle([x + 2, y - fs * 0.6, x + 14, y + fs * 0.6], fill=al((220, 220, 228), a))


def sc_rows(cv, d, prm, p, t, th, ctx):
    rows = prm["rows"]
    rh = 128
    w = 900
    head = 110 if prm.get("title") else 30
    h = head + rh * len(rows) + 30
    x0, y0 = CX - w // 2, VCY - h // 2
    a = st(p, 0, 0.1)
    card(cv, d, [x0, y0, x0 + w, y0 + h], th, a, r=36)
    if prm.get("title"):
        tc(d, (x0 + 44, y0 + 64), prm["title"], font(F_SEMI, 40), al(th["ink"], a), "lm")
        if prm.get("subtitle"):
            tc(d, (x0 + w - 44, y0 + 64), prm["subtitle"], font(F_MED, 28), al(th["muted"], a), "rm")
    flips = prm.get("flip")  # list of times (0-1) at which each row flips to good
    cur_target = None
    for i, r in enumerate(rows):
        label, sub, state = r[0], r[1], r[2]
        y = y0 + head + i * rh + rh / 2
        g = st(p, 0.06 + i * 0.07, 0.18 + i * 0.07)
        if i:
            d.line([(x0 + 40, y - rh / 2), (x0 + w - 40, y - rh / 2)], fill=al(th["line"], a * g), width=2)
        icon_col = r[3] if len(r) > 3 else None
        if icon_col:
            dot_icon(d, x0 + 76, y, 30, {"green": GREEN, "blue": BLUE, "red": RED, "ink": (40, 40, 48), "amber": AMBER, "teal": TEAL, "orange": ORANGE}.get(icon_col, BLUE), g, label[0].upper())
            lx = x0 + 128
        else:
            lx = x0 + 44
        tc(d, (lx, y - (16 if sub else 0)), label, font(F_SEMI, 38), al(th["ink"], g), "lm")
        if sub:
            tc(d, (lx, y + 26), sub, font(F_REG, 27), al(th["muted"], g), "lm")
        fl = 0
        if flips and i < len(flips) and flips[i] is not None:
            fl = st(p, flips[i], flips[i] + 0.05)
            if p < flips[i] + 0.06 and (cur_target is None):
                cur_target = (i, y, flips[i])
        s = state
        if fl >= 1:
            s = prm.get("flip_to", "pass")
        col = {"missing": RED, "fail": RED, "off": RED, "no": RED, "pass": GREEN, "on": GREEN, "yes": GREEN}.get(s.split()[0].lower(), th["muted"])
        if s.lower() in ("off", "on"):
            on = s.lower() == "on"
            tx, ty = x0 + w - 110, y
            d.rounded_rectangle([tx - 44, ty - 26, tx + 44, ty + 26], radius=26, fill=al(GREEN if on else th["line"], g))
            kx = tx + (18 if on else -18)
            d.ellipse([kx - 21, ty - 21, kx + 21, ty + 21], fill=al((255, 255, 255), g))
        else:
            pill(d, x0 + w - 44, y, s, col, g, anchor="r")
    if flips:
        # cursor travels to the next row to flip
        nxt = [(i, tt) for i, tt in enumerate(flips) if tt is not None and p < tt + 0.08]
        if nxt:
            i, tt = nxt[0]
            y = y0 + head + i * rh + rh / 2
            tx, ty = x0 + w - 110, y
            prev = [(j, t2) for j, t2 in enumerate(flips) if t2 is not None and t2 < tt]
            sx, sy = (CX + 120, y0 + h + 120) if not prev else (x0 + w - 110, y0 + head + prev[-1][0] * rh + rh / 2)
            m = ease_io((p - (tt - 0.12)) / 0.12)
            cxp, cyp = lerp(sx, tx, m), lerp(sy, ty, m)
            click = clamp((p - tt) / 0.06) if p >= tt else 0
            cursor(d, cxp, cyp, click)


def sc_number(cv, d, prm, p, t, th, ctx):
    v = prm["value"]
    f = ease_io((p - 0.02) / prm.get("count_end", 0.5)) if prm.get("count", True) else 1
    num = v * f
    dec = prm.get("dec", 0)
    s = prm.get("prefix", "") + (f"{num:,.{dec}f}") + prm.get("suffix", "")
    col = {"red": RED, "green": GREEN, "ink": th["ink"], "teal": TEAL, "orange": ORANGE, "amber": AMBER}[prm.get("color", "ink")]
    if prm.get("grid"):
        g = st(p, 0, 0.3)
        cols, rows_ = 11, 13
        cw, chh = 80, 64
        gx0 = CX - cols * cw / 2
        gy0 = VCY - rows_ * chh / 2
        k = 0
        for r in range(rows_):
            for c in range(cols):
                ff = st(p, 0.02 + (k % 37) * 0.006, 0.2 + (k % 37) * 0.006)
                k += 1
                x, y = gx0 + c * cw, gy0 + r * chh
                d.rounded_rectangle([x + 6, y + 6, x + cw - 6, y + chh - 6], radius=8, fill=al(th["card"], ff * g), outline=al(th["line"], ff * g), width=2)
                d.rounded_rectangle([x + 16, y + 20, x + cw - 30, y + 28], radius=4, fill=al(mixc(th["line"], RED, 0.5), ff * g))
                d.rounded_rectangle([x + 16, y + 36, x + cw - 18, y + 42], radius=3, fill=al(th["line"], ff * g))
    sz = prm.get("size", 200)
    fnt = font(F_BOLD, sz)
    while d.textlength(s, font=fnt) > 980:
        sz -= 6
        fnt = font(F_BOLD, sz)
    a = st(p, 0, 0.12)
    y = VCY + prm.get("dy", 0)
    if prm.get("grid"):
        bw = d.textlength(s, font=fnt) + 90
        card(cv, d, [CX - bw / 2, y - sz * 0.62, CX + bw / 2, y + sz * 0.95], th, a, r=36)
    tc(d, (CX, y), s, fnt, al(col, a))
    if prm.get("sub"):
        tc(d, (CX, y + sz * 0.62), prm["sub"], font(F_MED, 34), al(th["muted"], st(p, 0.15, 0.3)))


def sc_ring(cv, d, prm, p, t, th, ctx):
    R = 290
    a = st(p, 0, 0.1)
    f = ease_io((p - 0.05) / prm.get("count_end", 0.6))
    frac = lerp(prm.get("from_frac", 0.0), prm.get("to_frac", 1.0), f)
    v = lerp(prm.get("from", 0), prm["to"], f)
    danger = prm.get("danger_at", 9e9)
    good = prm.get("good", False)
    col = GREEN if good else (mixc(TEAL, RED, clamp((frac - danger) * 6)) if danger < 9e8 else TEAL)
    if prm.get("color"):
        col = {"red": RED, "green": GREEN, "teal": TEAL}[prm["color"]]
    box = [CX - R, VCY - R, CX + R, VCY + R]
    d.ellipse(box, outline=al(th["line"], a), width=26)
    end = -90 + 360 * min(frac, 1.0)
    for k, wd in ((0.15, 60), (0.3, 40), (1, 26)):
        if end > -90:
            d.arc(box, -90, end, fill=al(col, k * a), width=wd)
    tc(d, (CX, VCY - 110), prm.get("label", ""), font(F_MED, 34), al(th["muted"], a))
    dec = prm.get("dec", 0)
    txt = prm.get("prefix", "") + f"{v:,.{dec}f}" + prm.get("suffix", "")
    tc(d, (CX, VCY), txt, font(F_BOLD, prm.get("size", 120)), al(col, a))
    if prm.get("sub"):
        tc(d, (CX, VCY + 105), prm["sub"], font(F_MONO, 30), al(th["muted"], a))


def sc_stack(cv, d, prm, p, t, th, ctx):
    n = prm.get("n", 20)
    cache = prm.get("cache", False)
    w, h = 920, 760
    x0, y0 = CX - w // 2, VCY - h // 2
    a = st(p, 0, 0.1)
    card(cv, d, [x0, y0, x0 + w, y0 + h], th, a, r=36)
    tc(d, (x0 + 44, y0 + 60), prm.get("title", "Tokens sent per message"), font(F_SEMI, 36), al(th["ink"], a), "lm")
    bx0, bx1, by1 = x0 + 50, x0 + w - 50, y0 + h - 90
    maxh = h - 240
    bw = (bx1 - bx0) / n
    shown = st(p, 0.05, 0.7) * n
    tot = 0
    totc = 0
    for i in range(n):
        g = clamp(shown - i)
        if g <= 0:
            break
        hh = maxh * (i + 1) / n * ease(g)
        x = bx0 + i * bw + 4
        new = maxh / n * ease(g)
        if cache:
            d.rounded_rectangle([x, by1 - hh, x + bw - 8, by1], radius=6, fill=al(th["line"], a))
            d.rounded_rectangle([x, by1 - new, x + bw - 8, by1], radius=6, fill=al(GREEN, a))
        else:
            d.rounded_rectangle([x, by1 - hh, x + bw - 8, by1], radius=6, fill=al(mixc(AMBER, RED, i / n), a))
        tot += (i + 1) * ease(g)
        totc += ((i + 1) * 0.1 + 0.9 * 1) * ease(g) if cache else 0
    unit = prm.get("unit", 1000)
    tc(d, (x0 + w - 44, y0 + 60), f"{int(tot * unit):,} tokens", font(F_MONO_B, 32), al(th["muted"], a), "rm")
    tc(d, (bx0, by1 + 44), "msg 1", font(F_MONO, 26), al(th["muted"], a), "lm")
    tc(d, (bx1, by1 + 44), f"msg {n}", font(F_MONO, 26), al(th["muted"], a), "rm")
    if cache:
        g = st(p, 0.55, 0.7)
        d.rounded_rectangle([x0 + 50, y0 + 110, x0 + 74, y0 + 134], radius=6, fill=al(th["line"], g))
        tc(d, (x0 + 86, y0 + 122), "cached · 10% price", font(F_MED, 28), al(th["muted"], g), "lm")
        d.rounded_rectangle([x0 + 400, y0 + 110, x0 + 424, y0 + 134], radius=6, fill=al(GREEN, g))
        tc(d, (x0 + 436, y0 + 122), "new · full price", font(F_MED, 28), al(th["muted"], g), "lm")


def sc_chat(cv, d, prm, p, t, th, ctx):
    w = 900
    msgs = prm["msgs"]
    f = font(F_REG, 32)
    lay = []
    hh = 120
    for who, txt in msgs:
        ls = wrap(d, txt, f, 620)
        bh = len(ls) * 44 + 40
        lay.append((who, ls, bh))
        hh += bh + 24
    h = hh + 20
    x0, y0 = CX - w // 2, VCY - h // 2
    a = st(p, 0, 0.1)
    card(cv, d, [x0, y0, x0 + w, y0 + h], th, a, r=36)
    tc(d, (x0 + 44, y0 + 58), prm.get("title", "AI assistant"), font(F_SEMI, 34), al(th["ink"], a), "lm")
    d.ellipse([x0 + w - 70, y0 + 46, x0 + w - 46, y0 + 70], fill=al(GREEN, a))
    y = y0 + 110
    for i, (who, ls, bh) in enumerate(lay):
        g = st(p, 0.08 + i * 0.16, 0.2 + i * 0.16)
        if g <= 0:
            break
        bw = max(d.textlength(l, font=f) for l in ls) + 52
        if who == "you":
            bx = x0 + w - 40 - bw
            fill, col = BLUE, (255, 255, 255)
        else:
            bx = x0 + 40
            fill, col = th["field"], th["ink"]
        d.rounded_rectangle([bx, y + (1 - g) * 20, bx + bw, y + bh + (1 - g) * 20], radius=26, fill=al(fill, g))
        for j, l in enumerate(ls):
            hl = prm.get("strike") and who != "you" and j == len(ls) - 1 and False
            tc(d, (bx + 26, y + 40 + j * 44 + (1 - g) * 20), l, f, al(col, g), "lm")
        y += bh + 24
    if prm.get("note"):
        g = st(p, 0.7, 0.8)
        pill(d, CX, y0 + h + 60, prm["note"], RED, g)


def sc_inbox(cv, d, prm, p, t, th, ctx):
    w, rows = 940, prm["rows"]
    rh = 112
    h = 150 + rh * len(rows) + 20
    x0, y0 = CX - w // 2, VCY - h // 2
    a = st(p, 0, 0.1)
    card(cv, d, [x0, y0, x0 + w, y0 + h], th, a, r=34)
    mv = st(p, prm.get("move_at", 0.45), prm.get("move_at", 0.45) + 0.18)
    tabs = ["Primary", "Promotions", "Spam"]
    for k, tb in enumerate(tabs):
        x = x0 + 44 + k * 230
        on = k == 0
        tc(d, (x, y0 + 64), tb, font(F_SEMI, 32), al(th["ink"] if on else th["muted"], a), "lm")
        if on:
            d.rounded_rectangle([x, y0 + 96, x + 120, y0 + 102], radius=3, fill=al(BLUE, a))
    if mv > 0:
        bx = x0 + 44 + 2 * 230 + 92
        d.ellipse([bx - 20, y0 + 44, bx + 20, y0 + 84], fill=al(RED, mv * a))
        tc(d, (bx, y0 + 64), "1", font(F_BOLD, 26), al((255, 255, 255), mv * a))
    me = prm.get("me", 0)
    y = y0 + 130
    for i, (sender, subj, tm) in enumerate(rows):
        g = st(p, 0.04 + i * 0.05, 0.16 + i * 0.05)
        dx = 0
        ra = g
        if i == me:
            dx = ease_io((p - prm.get("move_at", 0.45)) / 0.18) * 900
            ra = g * (1 - mv * 0.9)
        yy = y + i * rh
        if i == me and mv < 1:
            d.rounded_rectangle([x0 + 20 + dx, yy, x0 + w - 20 + dx, yy + rh - 10], radius=18, fill=al(mixc(th["card"], BLUE, 0.06), ra))
        tc(d, (x0 + 60 + dx, yy + 34), sender, font(F_SEMI, 32), al(th["ink"], ra), "lm")
        tc(d, (x0 + 60 + dx, yy + 72), subj, font(F_REG, 28), al(th["muted"], ra), "lm")
        tc(d, (x0 + w - 50 + dx, yy + 34), tm, font(F_MED, 26), al(th["muted"], ra), "rm")
    if mv > 0:
        pill(d, CX, y0 + h + 64, prm.get("toast", "Moved to Spam"), RED, mv)


def sc_phone(cv, d, prm, p, t, th, ctx):
    w, h = 520, 880
    x0, y0 = CX - w // 2, VCY - h // 2
    a = st(p, 0, 0.1)
    shadow(cv, [x0, y0, x0 + w, y0 + h], 80, th, a)
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=80, fill=al((10, 10, 12), a), outline=al((70, 70, 78), a), width=6)
    d.rounded_rectangle([CX - 80, y0 + 26, CX + 80, y0 + 66], radius=20, fill=al((0, 0, 0), a))
    tc(d, (CX, y0 + 140), prm.get("date", "Saturday, October 24"), font(F_MED, 28), al((230, 230, 236), a))
    tc(d, (CX, y0 + 240), prm.get("time", "9:41"), font(F_BOLD, 150), al((240, 240, 244), a))
    y = y0 + 370
    for i, n in enumerate(prm["notes"]):
        g = st(p, 0.12 + i * prm.get("stagger", 0.12), 0.24 + i * prm.get("stagger", 0.12))
        if g <= 0:
            continue
        nh = 140
        yy = y + i * (nh + 14) - (1 - g) * 30
        d.rounded_rectangle([x0 + 24, yy, x0 + w - 24, yy + nh], radius=30, fill=al((44, 44, 50), g * 0.95))
        col = {"red": RED, "green": GREEN, "blue": BLUE, "ink": (60, 60, 66), "orange": ORANGE, "amber": AMBER}[n.get("color", "blue")]
        dot_icon(d, x0 + 74, yy + 46, 24, col, g, n["app"][0])
        tc(d, (x0 + 112, yy + 46), n["app"].upper(), font(F_SEMI, 22), al((170, 170, 180), g), "lm")
        tc(d, (x0 + w - 48, yy + 46), n.get("when", "now"), font(F_MED, 22), al((150, 150, 160), g), "rm")
        tc(d, (x0 + 52, yy + 86), n["title"], font(F_SEMI, 28), al((245, 245, 248), g), "lm")
        if n.get("body"):
            tc(d, (x0 + 52, yy + 120), n["body"], font(F_REG, 25), al((185, 185, 194), g), "lm")


def sc_timeline(cv, d, prm, p, t, th, ctx):
    items = prm["items"]
    a = st(p, 0, 0.1)
    timer = prm.get("timer", 60)
    tv = timer * ease_io((p - 0.05) / 0.75)
    if timer:
        tc(d, (CX, VY0 + 110), f"{int(tv // 60)}:{int(tv % 60):02d}", font(F_MONO_B, 150), al(th["ink"], a))
    y0 = VY0 + (300 if timer else 200)
    step = 150
    for i, (tm, txt, colname) in enumerate(items):
        g = st(p, 0.15 + i * 0.2, 0.27 + i * 0.2)
        col = {"red": RED, "green": GREEN, "ink": th["ink"], "amber": AMBER}[colname]
        y = y0 + i * step
        if i < len(items) - 1:
            g2 = st(p, 0.27 + i * 0.2, 0.35 + i * 0.2)
            d.line([(220, y + 22), (220, y + 22 + (step - 44) * g2)], fill=al(th["line"], a), width=6)
        d.ellipse([220 - 20, y - 20, 220 + 20, y + 20], fill=al(col, g), outline=al(th["card"], g), width=4)
        tc(d, (270, y - 18), tm, font(F_MONO_B, 32), al(col, g), "lm")
        tc(d, (270, y + 26), txt, font(F_SEMI, 38), al(th["ink"], g), "lm")


def sc_dialog(cv, d, prm, p, t, th, ctx):
    w = 900
    reveal = st(p, prm.get("reveal_at", 0.55), prm.get("reveal_at", 0.55) + 0.15)
    args = prm["args"]
    hidden = prm.get("hidden", [])
    fm = font(F_MONO, 30)
    base_h = 300 + 52 * len(args)
    h = base_h + reveal * (52 * len(hidden) + 30)
    x0, y0 = CX - w // 2, VCY - base_h // 2 - 40
    a = st(p, 0, 0.1)
    bg = (22, 22, 26)
    shadow(cv, [x0, y0, x0 + w, y0 + h], 30, th, a)
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=30, fill=al(bg, a), outline=al((52, 52, 60), a), width=2)
    tc(d, (x0 + 44, y0 + 60), prm.get("title", "Run MCP tool?"), font(F_SEMI, 36), al((245, 245, 248), a), "lm")
    pill(d, x0 + w - 44, y0 + 60, prm.get("tool", "add"), BLUE, a, f=font(F_MONO_B, 28), anchor="r")
    y = y0 + 130
    for k, ln in enumerate(args):
        tc(d, (x0 + 44, y + k * 52), ln, fm, al((210, 210, 218), a), "lm")
    y += 52 * len(args)
    if reveal > 0:
        hy = y + 6
        d.rounded_rectangle([x0 + 28, hy - 24, x0 + w - 28, hy - 24 + 52 * len(hidden) + 18], radius=12, fill=al(RED, 0.16 * reveal), outline=al(RED, reveal), width=2)
        for k, ln in enumerate(hidden):
            tc(d, (x0 + 44, hy + 4 + k * 52), ln, fm, al((255, 160, 160), reveal), "lm")
        y += 52 * len(hidden) + 30
    by = y0 + h - 80
    d.rounded_rectangle([x0 + w - 330, by - 34, x0 + w - 190, by + 34], radius=16, fill=al((48, 48, 56), a))
    tc(d, (x0 + w - 260, by), "Skip", font(F_SEMI, 30), al((220, 220, 228), a))
    d.rounded_rectangle([x0 + w - 170, by - 34, x0 + w - 40, by + 34], radius=16, fill=al(BLUE, a))
    tc(d, (x0 + w - 105, by), "Run", font(F_SEMI, 30), al((255, 255, 255), a))
    ca = prm.get("click_at", 0.4)
    m = ease_io((p - (ca - 0.15)) / 0.15)
    if p < prm.get("reveal_at", 0.55) + 0.05:
        cursor(d, lerp(CX + 100, x0 + w - 105, m), lerp(y0 + h + 160, by, m), clamp((p - ca) / 0.06) if p >= ca else 0)


def sc_tool(cv, d, prm, p, t, th, ctx):
    """MCP tool definition card: name, short description, then hidden instructions slide open."""
    w = 940
    reveal = st(p, prm.get("reveal_at", 0.3), prm.get("reveal_at", 0.3) + 0.2)
    hid = prm.get("hidden", [])
    fm = font(F_MONO, 29)
    h = 280 + reveal * (54 * len(hid) + 40)
    x0, y0 = CX - w // 2, int(VCY - (280 + 54 * len(hid) + 40) / 2)
    a = st(p, 0, 0.1)
    card(cv, d, [x0, y0, x0 + w, y0 + h], th, a, r=34)
    pill(d, x0 + 44, y0 + 62, "MCP tool", BLUE, a, anchor="l")
    tc(d, (x0 + 220, y0 + 62), prm.get("name", "add(a, b)"), font(F_MONO_B, 36), al(th["ink"], a), "lm")
    tc(d, (x0 + 44, y0 + 140), "description", font(F_MED, 26), al(th["muted"], a), "lm")
    tc(d, (x0 + 44, y0 + 190), prm.get("desc", "Adds two numbers."), font(F_SEMI, 38), al(th["ink"], a), "lm")
    if reveal < 0.4:
        tc(d, (x0 + w - 44, y0 + 140), "what you see", font(F_MED, 26), al(GREEN, a * (1 - reveal / 0.4)), "rm")
    if reveal > 0:
        yy = y0 + 250
        d.rounded_rectangle([x0 + 28, yy, x0 + w - 28, yy + 54 * len(hid) + 24], radius=14, fill=al(RED, 0.08 * reveal), outline=al(RED, reveal), width=2)
        for k, ln in enumerate(hid):
            g = st(p, prm.get("reveal_at", 0.3) + 0.05 + k * 0.05, prm.get("reveal_at", 0.3) + 0.15 + k * 0.05)
            hl = k in prm.get("hl", [])
            tc(d, (x0 + 48, yy + 40 + k * 54), ln, font(F_MONO_B if hl else F_MONO, 29), al(RED if hl else mixc(th["ink"], RED, 0.35), g), "lm")
        pill(d, x0 + w - 44, y0 + 140, "what the AI sees", RED, reveal, anchor="r")


def sc_statement(cv, d, prm, p, t, th, ctx):
    lines = prm["lines"]
    a = st(p, 0, 0.15)
    sz = prm.get("size", 120)
    y = VCY - (len(lines) - 1) * sz * 0.58
    for i, ln in enumerate(lines):
        g = st(p, i * 0.1, 0.15 + i * 0.1)
        col = RED if i == prm.get("red", -1) else th["ink"]
        tc(d, (CX, y + (1 - g) * 30), ln, font(F_BOLD, sz), al(col, g))
        y += sz * 1.16
    if prm.get("sub"):
        tc(d, (CX, y + 10), prm["sub"], font(F_MED, 34), al(th["muted"], st(p, 0.25, 0.4)))


def sc_prompt(cv, d, prm, p, t, th, ctx):
    w = 960
    fm = font(F_MONO, 33)
    ls = []
    for para in prm["text"].split("\n"):
        ls += wrap(d, para, fm, w - 100) or [""]
    h = 170 + len(ls) * 50 + 60
    x0, y0 = CX - w // 2, max(VY0 - 120, VCY - h // 2 - 60)
    a = st(p, 0, 0.12)
    card(cv, d, [x0, y0, x0 + w, y0 + h], th, a, r=36)
    # asterisk mark
    for k in range(4):
        ang = k * math.pi / 4
        d.line([(x0 + 62 - 16 * math.cos(ang), y0 + 66 - 16 * math.sin(ang)), (x0 + 62 + 16 * math.cos(ang), y0 + 66 + 16 * math.sin(ang))], fill=al(ORANGE, a), width=6)
    tc(d, (x0 + 96, y0 + 66), "The prompt", font(F_SEMI, 38), al(th["ink"], a), "lm")
    pill(d, x0 + w - 44, y0 + 66, "screenshot this", BLUE, a, anchor="r")
    shown = int(sum(len(l) for l in ls) * st(p, 0.05, 0.45))
    used = 0
    for i, l in enumerate(ls):
        n = clamp(shown - used, 0, len(l))
        used += len(l)
        tc(d, (x0 + 50, y0 + 150 + i * 50), l[:int(n)], fm, al(th["ink"], a), "lm")


SCENES = {"browser": sc_browser, "code": sc_code, "rows": sc_rows, "number": sc_number, "ring": sc_ring,
          "stack": sc_stack, "chat": sc_chat, "inbox": sc_inbox, "phone": sc_phone, "timeline": sc_timeline,
          "dialog": sc_dialog, "tool": sc_tool, "statement": sc_statement, "prompt": sc_prompt}


# ------------------------------------------------------------------ furniture
def make_bg(th):
    img = Image.new("RGB", (W, H), th["bg"])
    d = ImageDraw.Draw(img)
    for x in range(0, W, 54):
        d.line([(x, 0), (x, H)], fill=th["grid"], width=1)
    for y in range(0, H, 54):
        d.line([(0, y), (W, y)], fill=th["grid"], width=1)
    # soft radial fade so the grid disappears at the edges
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W * 0.7)) ** 2 + ((yy - H / 2) / (H * 0.6)) ** 2)
    m = np.clip(1 - r, 0, 1) ** 0.8
    base = np.full((H, W, 3), th["bg"], np.float32)
    arr = np.asarray(img).astype(np.float32)
    out = base + (arr - base) * m[..., None]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGB")


_parts = np.random.default_rng(3).random((90, 4))


def particles(d, t, col, a=1.0):
    for k, (px, py, ps, ph) in enumerate(_parts):
        x = px * W + math.sin(t * 0.5 + ph * 6) * 20
        y = (py * H - t * (20 + 40 * ps)) % H
        r = 3 + 4 * ps
        tw = 0.5 + 0.5 * math.sin(t * 2 + ph * 9)
        d.ellipse([x - r * 2.2, y - r * 2.2, x + r * 2.2, y + r * 2.2], fill=al(col, 0.12 * a * tw))
        d.ellipse([x - r, y - r, x + r, y + r], fill=al(col, 0.75 * a * tw))


def caption(d, seg, t, th):
    words = seg["beat"]["say"].split()
    keys = [k.lower() for k in seg["beat"].get("key", [])]
    dur = seg["end"] - seg["start"]
    wts = np.array([len(w) + 2 for w in words], float)
    cum = np.cumsum(wts) / wts.sum() * dur
    k = min(int(np.searchsorted(cum, t - seg["start"])), len(words) - 1)
    pre = t < seg["start"]
    # chunk words into caption groups of <= 4 words
    groups, g = [], []
    for i, w in enumerate(words):
        g.append(i)
        if len(g) == 4 or w.endswith((".", ",", "?", ":")) and len(g) >= 2:
            groups.append(g)
            g = []
    if g:
        groups.append(g)
    grp = next(gr for gr in groups if k in gr)
    if pre:
        k = grp[-1]
    f = font(F_SEMI, 88)
    toks = [words[i] for i in grp]
    # layout in up to 2 lines
    lines, cur = [], []
    for i, w in zip(grp, toks):
        test = " ".join(words[j] for j in cur + [i])
        if cur and d.textlength(test, font=f) > 920:
            lines.append(cur)
            cur = [i]
        else:
            cur.append(i)
    lines.append(cur)
    y = seg["beat"].get("cap_y", 1400) - (len(lines) - 1) * 52
    for ln in lines:
        widths = [d.textlength(words[i], font=f) for i in ln]
        sp = d.textlength(" ", font=f)
        x = CX - (sum(widths) + sp * (len(ln) - 1)) / 2
        for i, wd in zip(ln, widths):
            if i <= k:
                clean = re.sub(r"[^\w$%.,'-]", "", words[i]).strip(".,").lower()
                is_key = any(clean == kk.lower().strip(".,") or clean.startswith(kk.lower()) and len(kk) > 3 for kk in keys)
                col = th["ink"] if i == k or is_key else th["dim"]
                if is_key:
                    g = 1.0 if pre else clamp((t - (seg["start"] + (cum[i - 1] if i else 0))) / 0.12)
                    bx0, by0, bx1, by1 = x - 12, y - 58, x + wd + 12, y + 50
                    d.rectangle([bx0, by0, bx1, by1], fill=al(BLUE, 0.13 * g), outline=al(BLUE, g), width=3)
                    for (hx, hy) in ((bx0, by0), (bx1, by1)):
                        d.ellipse([hx - 11, hy - 11, hx + 11, hy + 11], fill=al(BLUE, g), outline=al((255, 255, 255), g), width=3)
                d.text((x, y), words[i], font=f, fill=al(col, 1), anchor="lm")
            x += wd + sp
        y += 104


def progress_pill(cv, d, t, prompt_t, total_speech, th, a=1.0):
    w, h = 820, 82
    x0, y0 = CX - w // 2, 1700
    shadow(cv, [x0, y0, x0 + w, y0 + h], 41, th, 0.6 * a, spread=24, dy=12)
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=41, fill=al(th["pill"], a), outline=al(th["line"], a), width=2)
    frac = clamp(t / max(prompt_t, 0.1))
    if frac > 0.02:
        d.rounded_rectangle([x0 + 4, y0 + 4, x0 + 4 + (w - 8) * frac, y0 + h - 4], radius=37, fill=al(th["pillfill"], a))
    ax, ay = x0 + 48, y0 + h / 2
    for k in range(4):
        ang = k * math.pi / 4
        d.line([(ax - 14 * math.cos(ang), ay - 14 * math.sin(ang)), (ax + 14 * math.cos(ang), ay + 14 * math.sin(ang))], fill=al(ORANGE, a), width=5)
    tc(d, (x0 + 80, ay), "the prompt · at the end", font(F_SEMI, 32), al(th["ink"], a), "lm")
    left = max(0, math.ceil(prompt_t - t))
    tc(d, (x0 + w - 40, ay), f"{left}s", font(F_SEMI, 32), al(th["muted"], a), "rm")


# ------------------------------------------------------------------ audio
def build_audio(script, model_dir):
    scale = 1.0
    for _ in range(4):
        clips = []
        for b in script["beats"]:
            sp = min(script.get("speed", SPEED) * scale, 1.35)
            vc = b.get("voice", script.get("voice", VOICE))
            if script.get("natural"):
                import re
                parts, pieces = re.split(r"(?<=[.!?])\s+", b["say"].strip()), []
                for pt in parts:
                    px, sr = voice.say(model_dir, pt, vc, sp)
                    pieces += [px, np.zeros(int(SR * (0.34 if pt.endswith("?") else 0.24)), np.float32)]
                x = np.concatenate(pieces[:-1])
            else:
                x, sr = voice.say(model_dir, b["say"], vc, sp)
            clips.append(x)
        speech = 0.15 + sum(len(x) / SR + GAP for x in clips)
        if speech + 2.5 <= MAX_LEN or scale >= 1.2:
            break
        scale *= (speech + 2.5) / (MAX_LEN - 0.2)
    hold = script.get("hold", 2.6)
    total = speech + hold
    n = int(total * SR) + SR // 10
    vox = np.zeros(n, np.float32)
    t = 0.15
    tl = []
    for b, x in zip(script["beats"], clips):
        i = int(t * SR)
        vox[i:i + len(x)] += x
        tl.append({"start": t, "end": t + len(x) / SR, "beat": b, "env": _envelope(x)})
        t += len(x) / SR + GAP
    vox = vox / (np.max(np.abs(vox)) or 1) * 0.92
    return tl, vox, total


def mix(vox, total, tl, starts, seed):
    n = len(vox)
    bed = music_bed(total, seed)[:n]
    bed = np.pad(bed, (0, n - len(bed)))
    duck = np.ones(n, np.float32) * 0.32
    for s in tl:
        duck[int(s["start"] * SR): int(s["end"] * SR)] = 0.13
    duck = np.convolve(duck, np.ones(2400) / 2400, mode="same")
    out = vox + bed * duck
    wh = sfx_whoosh() * 0.35
    pop = sfx_pop() * 0.35
    for k, s in enumerate(starts):
        i = int(max(s - 0.12, 0) * SR)
        sig = wh if k else pop
        out[i:i + len(sig)] += sig[: n - i]
    return (out / (np.max(np.abs(out)) or 1) * 0.95).astype(np.float32)


# ------------------------------------------------------------------ render
def plan(script, tl, total):
    scenes = []
    for i, s in enumerate(tl):
        v = s["beat"].get("vis")
        if v:
            span = v.get("span", 1)
            e = min(i + span, len(tl)) - 1
            scenes.append({"vis": v, "start": s["start"] - 0.08, "anim_end": tl[e]["end"],
                           "theme": s["beat"].get("bg", script.get("bg", "light"))})
    for k, sc in enumerate(scenes):
        sc["end"] = scenes[k + 1]["start"] if k + 1 < len(scenes) else total + 1
    if scenes:
        scenes[0]["start"] = -0.4
    return scenes


def render_frame(t, script, tl, scenes, total, bgs, prompt_t):
    cur, prev = None, None
    for sc in scenes:
        if sc["start"] <= t:
            prev, cur = cur, sc
    cur = cur or scenes[0]
    ts = t - cur["start"]
    th = THEMES[cur["theme"]]
    if prev and prev["theme"] != cur["theme"] and ts < 0.25:
        base = Image.blend(bgs[prev["theme"]], bgs[cur["theme"]], ease(ts / 0.25))
    else:
        base = bgs[cur["theme"]]
    img = base.copy()
    d = ImageDraw.Draw(img, "RGBA")
    if cur["vis"].get("particles"):
        particles(d, t, {"red": RED, "teal": TEAL, "green": GREEN, "blue": BLUE}[cur["vis"]["particles"]], clamp(ts / 0.4))
    p = clamp(ts / max(cur["anim_end"] - cur["start"], 0.5))
    SCENES[cur["vis"]["type"]](img, d, cur["vis"], p, ts, th, {})
    # entry: blur + scale in; exit: quick fade
    ein = clamp(ts / 0.28)
    eout = clamp((cur["end"] - t) / 0.14)
    if ein < 1:
        z = 1 + 0.05 * (1 - ease(ein))
        lw, lh = int(W * z), int(H * z)
        oy = int((lh - H) * VCY / H)
        img = img.resize((lw, lh), Image.BILINEAR).crop(((lw - W) // 2, oy, (lw - W) // 2 + W, oy + H))
        img = img.filter(ImageFilter.GaussianBlur(10 * (1 - ease(ein))))
        img = Image.blend(base, img, ease(ein))
    if eout < 1:
        img = Image.blend(base, img.filter(ImageFilter.GaussianBlur(8 * (1 - eout))), eout)
    d = ImageDraw.Draw(img, "RGBA")
    seg = None
    for s in tl:
        if s["start"] <= t <= s["end"] + GAP:
            seg = s
    if seg is None and t < tl[0]["start"]:
        seg = tl[0]
    if seg and not seg["beat"].get("nocap"):
        caption(d, seg, min(t, seg["end"]), th)
    if t < prompt_t:
        progress_pill(img, d, t, prompt_t, total, th, clamp((prompt_t - t) / 0.3))
    return img


def render(script, out, model_dir, only=None):
    tl, vox, total = build_audio(script, model_dir)
    scenes = plan(script, tl, total)
    prompt_t = next((sc["start"] for sc in scenes if sc["vis"]["type"] == "prompt"), total)
    bgs = {k: make_bg(v) for k, v in THEMES.items()}
    if only is not None:
        return [render_frame(tt, script, tl, scenes, total, bgs, prompt_t) for tt in only], tl, total
    seed = sum(ord(c) for c in script["id"])
    audio = mix(vox, total, tl, [s["start"] for s in scenes], seed)
    wav = out[:-4] + ".wav"
    sf.write(wav, audio, SR)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-af", "loudnorm=I=-14:TP=-1.2:LRA=11", "-c:a", "aac", "-b:a", "192k", "-shortest",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for fr in range(int(total * FPS)):
        ff.stdin.write(render_frame(fr / FPS, script, tl, scenes, total, bgs, prompt_t).tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    return total


def write_post(s, path, total):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{s['title']}\nLength: {total:.1f}s\n\nCAPTION (paste this)\n{s['caption']}\n\n")
        if s.get("base44"): fh.write("Build your own AI app on Base44. Code AT100 = $100 in free credits (link in bio).\n\n")
        fh.write(" ".join("#" + h for h in s["hashtags"]) + "\n\nTHE PROMPT\n" + s["prompt"] + "\n")
        if s.get("sources"):
            fh.write("\nSOURCES (for you)\n" + "\n".join("- " + x for x in s["sources"]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills", action="store_true", help="contact sheet only (uses real voice timing)")
    a = ap.parse_args()
    s = json.load(open(a.script, encoding="utf-8"))
    for b in s["beats"]:
        if b.get("vis", {}).get("type") == "prompt":
            b["vis"]["text"] = s["prompt"]
    if a.stills:
        _, tl, total = render(s, a.out, a.model, only=[0.0])
        ts = [0.0] + [x["start"] + (x["end"] - x["start"]) * 0.8 for x in tl] + [total - 0.3]
        frames, _, _ = render(s, a.out, a.model, only=ts)
        sh = Image.new("RGB", (240 * len(frames), 427))
        for i, f in enumerate(frames):
            sh.paste(f.resize((240, 427)), (i * 240, 0))
        sh.save(a.out)
        print(a.out, f"{total:.1f}s")
    else:
        total = render(s, a.out, a.model)
        write_post(s, a.out[:-4] + "_post.txt", total)
        print(f"{a.out}: {total:.1f}s")
