"""Pip Explains: 1080x1920 animated explainer reels.

usage: python explain.py script.json out.mp4 --model <kokoro dir>

A script is a hook, a list of beats (Pip's lines) and, per beat, an optional
visual scene. A scene keeps playing until a later beat brings a new one; "span"
stretches one scene's animation across several beats.
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import characters as C
import voice
from make_reel import music_bed, sfx_pop, sfx_whoosh, _envelope

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
W, H, FPS, SR = 1080, 1920, 30, 24000
MAX_LEN, GAP, END = 33.0, 0.16, 1.7

F_BOLD = os.path.join(A, "Poppins-Bold.ttf")
F_MED = os.path.join(A, "Poppins-Medium.ttf")
F_MONO = os.path.join(A, "DejaVuSansMono.ttf")

TEXT = (244, 244, 252)
MUTED = (150, 156, 196)
CARD = (27, 31, 62)
LINE = (72, 82, 140)
BAD = (255, 84, 104)
GOOD = (80, 224, 150)
ACCENTS = {
    "orange": (255, 128, 92), "cyan": (58, 214, 197), "violet": (146, 128, 255),
    "yellow": (255, 210, 102), "pink": (255, 92, 150), "green": (92, 224, 138),
}
SC = (60, 320, 1020, 1250)  # scene box
CX, CY = 540, 780

_fonts = {}


def font(path, size):
    k = (path, int(size))
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(path, max(int(size), 6))
    return _fonts[k]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def back(x):
    x = clamp(x)
    c = 1.9
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def st(p, a, b):
    return ease((p - a) / max(b - a, 1e-6))


def lerp(a, b, f):
    return a + (b - a) * f


def mixc(c1, c2, f):
    return tuple(int(lerp(a, b, f)) for a, b in zip(c1, c2))


def al(c, a):
    return tuple(c[:3]) + (int(255 * clamp(a)),)


# ------------------------------------------------------------------ drawing helpers
def text_c(d, xy, s, f, fill, anchor="mm", stroke=0, sfill=None):
    d.text(xy, s, font=f, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=sfill)


def wrap(d, s, f, maxw):
    out, cur = [], ""
    for w in s.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= maxw:
            cur = t
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def card(d, box, a=1.0, fill=CARD, line=LINE, r=34, w=3):
    d.rounded_rectangle(box, radius=r, fill=al(fill, a), outline=al(line, a), width=w)


def glow_line(d, p0, p1, col, a=1.0, w=6):
    d.line([p0, p1], fill=al(col, 0.16 * a), width=w + 18)
    d.line([p0, p1], fill=al(col, 0.32 * a), width=w + 8)
    d.line([p0, p1], fill=al(col, a), width=w)


def glow_dot(d, x, y, r, col, a=1.0):
    d.ellipse([x - r * 2.4, y - r * 2.4, x + r * 2.4, y + r * 2.4], fill=al(col, 0.18 * a))
    d.ellipse([x - r * 1.5, y - r * 1.5, x + r * 1.5, y + r * 1.5], fill=al(col, 0.35 * a))
    d.ellipse([x - r, y - r, x + r, y + r], fill=al((255, 255, 255), a))


def particles(d, p0, p1, t, col, a=1.0, n=3, speed=0.9, r=6):
    for k in range(n):
        f = (t * speed + k / n) % 1.0
        glow_dot(d, lerp(p0[0], p1[0], f), lerp(p0[1], p1[1], f), r, col, a * math.sin(math.pi * f))


def arrow_head(d, p0, p1, col, a=1.0, s=22):
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    pts = [p1, (p1[0] - s * math.cos(ang - 0.45), p1[1] - s * math.sin(ang - 0.45)),
           (p1[0] - s * math.cos(ang + 0.45), p1[1] - s * math.sin(ang + 0.45))]
    d.polygon(pts, fill=al(col, a))


def icon(d, name, x, y, s, col, a=1.0):
    """Simple line icons, s = half size."""
    c = al(col, a)
    w = max(3, int(s * 0.14))
    if name == "doc":
        d.polygon([(x - s * .65, y - s), (x + s * .3, y - s), (x + s * .65, y - s * .6), (x + s * .65, y + s), (x - s * .65, y + s)], outline=c, width=w)
        for k in range(3):
            d.line([(x - s * .38, y - s * .2 + k * s * .4), (x + s * .38, y - s * .2 + k * s * .4)], fill=c, width=w)
    elif name == "db":
        d.ellipse([x - s * .8, y - s, x + s * .8, y - s * .5], outline=c, width=w)
        d.line([(x - s * .8, y - s * .75), (x - s * .8, y + s * .75)], fill=c, width=w)
        d.line([(x + s * .8, y - s * .75), (x + s * .8, y + s * .75)], fill=c, width=w)
        d.arc([x - s * .8, y + s * .5, x + s * .8, y + s], 0, 180, fill=c, width=w)
        d.arc([x - s * .8, y - s * .1, x + s * .8, y + s * .4], 0, 180, fill=c, width=w)
    elif name == "globe":
        d.ellipse([x - s, y - s, x + s, y + s], outline=c, width=w)
        d.ellipse([x - s * .45, y - s, x + s * .45, y + s], outline=c, width=w)
        d.line([(x - s, y), (x + s, y)], fill=c, width=w)
    elif name == "mail":
        d.rectangle([x - s, y - s * .65, x + s, y + s * .65], outline=c, width=w)
        d.line([(x - s, y - s * .65), (x, y + s * .1), (x + s, y - s * .65)], fill=c, width=w)
    elif name == "code":
        text_c(d, (x, y), "</>", font(F_BOLD, s * 1.2), c)
    elif name == "calendar":
        d.rounded_rectangle([x - s, y - s * .8, x + s, y + s * .9], radius=int(s * .15), outline=c, width=w)
        d.line([(x - s, y - s * .35), (x + s, y - s * .35)], fill=c, width=w)
        for i in range(3):
            for j in range(2):
                d.ellipse([x - s * .6 + i * s * .6 - 5, y + j * s * .5 - 5, x - s * .6 + i * s * .6 + 5, y + j * s * .5 + 5], fill=c)
    elif name == "chat":
        d.rounded_rectangle([x - s, y - s * .75, x + s, y + s * .45], radius=int(s * .35), outline=c, width=w)
        d.polygon([(x - s * .4, y + s * .4), (x - s * .6, y + s * .95), (x, y + s * .4)], fill=c)
    elif name == "brain":
        for (ox, oy, r) in [(-.42, -.25, .5), (.42, -.25, .5), (-.45, .3, .45), (.45, .3, .45)]:
            d.ellipse([x + ox * s - r * s, y + oy * s - r * s, x + ox * s + r * s, y + oy * s + r * s], outline=c, width=w)
        d.line([(x, y - s * .7), (x, y + s * .7)], fill=c, width=w)
    elif name == "gear":
        pts = []
        for k in range(16):
            ang = k * math.pi / 8
            r = s if k % 2 == 0 else s * .72
            pts.append((x + r * math.cos(ang), y + r * math.sin(ang)))
        d.polygon(pts, outline=c, width=w)
        d.ellipse([x - s * .3, y - s * .3, x + s * .3, y + s * .3], outline=c, width=w)
    elif name == "user":
        d.ellipse([x - s * .4, y - s, x + s * .4, y - s * .2], outline=c, width=w)
        d.arc([x - s * .85, y - s * .05, x + s * .85, y + s * 1.6], 180, 360, fill=c, width=w)
    elif name == "lock":
        d.rounded_rectangle([x - s * .75, y - s * .1, x + s * .75, y + s], radius=int(s * .15), outline=c, width=w)
        d.arc([x - s * .5, y - s, x + s * .5, y + s * .1], 180, 360, fill=c, width=w)
    elif name == "search":
        d.ellipse([x - s * .85, y - s * .85, x + s * .35, y + s * .35], outline=c, width=w)
        d.line([(x + s * .2, y + s * .2), (x + s, y + s)], fill=c, width=w + 2)
    elif name == "image":
        d.rounded_rectangle([x - s, y - s * .8, x + s, y + s * .8], radius=int(s * .15), outline=c, width=w)
        d.polygon([(x - s * .8, y + s * .6), (x - s * .2, y - s * .1), (x + s * .3, y + s * .6)], fill=c)
        d.ellipse([x + s * .3, y - s * .55, x + s * .7, y - s * .15], fill=c)
    elif name == "chip":
        d.rounded_rectangle([x - s * .6, y - s * .6, x + s * .6, y + s * .6], radius=int(s * .15), outline=c, width=w)
        for k in range(3):
            o = -s * .35 + k * s * .35
            for (a0, a1, b0, b1) in [(x + o, y - s * .6, x + o, y - s), (x + o, y + s * .6, x + o, y + s),
                                     (x - s * .6, y + o, x - s, y + o), (x + s * .6, y + o, x + s, y + o)]:
                d.line([(a0, a1), (b0, b1)], fill=c, width=w)
        text_c(d, (x, y), "AI", font(F_BOLD, s * .55), c)
    elif name == "warn":
        d.polygon([(x, y - s), (x + s, y + s * .8), (x - s, y + s * .8)], outline=c, width=w)
        text_c(d, (x, y + s * .2), "!", font(F_BOLD, s), c)
    elif name == "folder":
        d.polygon([(x - s, y - s * .7), (x - s * .2, y - s * .7), (x, y - s * .45), (x + s, y - s * .45), (x + s, y + s * .75), (x - s, y + s * .75)], outline=c, width=w)
    elif name == "spark":
        pts = []
        for k in range(8):
            ang = -math.pi / 2 + k * math.pi / 4
            r = s if k % 2 == 0 else s * .3
            pts.append((x + r * math.cos(ang), y + r * math.sin(ang)))
        d.polygon(pts, fill=c)
    elif name == "eye":
        d.chord([x - s, y - s * .6, x + s, y + s * .6], 180, 360, outline=c, width=w)
        d.chord([x - s, y - s * .6, x + s, y + s * .6], 0, 180, outline=c, width=w)
        d.ellipse([x - s * .3, y - s * .3, x + s * .3, y + s * .3], fill=c)
    elif name == "laptop":
        d.rounded_rectangle([x - s * .8, y - s * .7, x + s * .8, y + s * .35], radius=int(s * .1), outline=c, width=w)
        d.polygon([(x - s, y + s * .5), (x + s, y + s * .5), (x + s * 1.1, y + s * .75), (x - s * 1.1, y + s * .75)], fill=c)
    elif name == "cloud":
        for (ox, oy, r) in [(-.45, .15, .45), (.05, -.15, .6), (.5, .15, .45)]:
            d.ellipse([x + ox * s - r * s, y + oy * s - r * s, x + ox * s + r * s, y + oy * s + r * s], outline=c, width=w)
    elif name == "check":
        d.line([(x - s * .7, y), (x - s * .15, y + s * .55), (x + s * .8, y - s * .6)], fill=c, width=w + 3)
    elif name == "x":
        d.line([(x - s * .6, y - s * .6), (x + s * .6, y + s * .6)], fill=c, width=w + 3)
        d.line([(x + s * .6, y - s * .6), (x - s * .6, y + s * .6)], fill=c, width=w + 3)


def node(d, x, y, label, ic, col, a=1.0, r=78, lit=1.0, sub=None):
    """Round glowing node with an icon and a label underneath."""
    if a <= 0:
        return
    rr = r * (0.6 + 0.4 * back(a))
    ring = mixc((80, 88, 140), col, lit)
    d.ellipse([x - rr - 16, y - rr - 16, x + rr + 16, y + rr + 16], fill=al(col, 0.10 * a * lit))
    d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=al(CARD, a), outline=al(ring, a), width=5)
    if ic:
        icon(d, ic, x, y, rr * 0.42, mixc((120, 126, 170), (255, 255, 255), lit), a)
    if label:
        text_c(d, (x, y + rr + 34), label, font(F_BOLD, 32), al(mixc(MUTED, TEXT, lit), a))
    if sub:
        text_c(d, (x, y + rr + 72), sub, font(F_MED, 26), al(MUTED, a))


# ------------------------------------------------------------------ scenes
# every scene: fn(d, img, prm, p, t, ac, ctx) ; p = 0..1 across its span, t = seconds since scene start

def sc_slam(d, img, prm, p, t, ac, ctx):
    lines = prm["lines"]
    hi = prm.get("hi", len(lines) - 1)
    size = prm.get("size", 104)
    total = len(lines) * 112 + (90 if prm.get("sub") else 0)
    y = CY - total / 2 + 56
    for i, ln in enumerate(lines):
        f = clamp((t - i * 0.22) / 0.32)
        if f <= 0:
            y += 112
            continue
        sz = size * (0.4 + 0.6 * back(f))
        fnt = font(F_BOLD, sz)
        while d.textlength(ln, font=fnt) > 920 and sz > 40:
            sz -= 4
            fnt = font(F_BOLD, sz)
        col = ac if i == hi else TEXT
        text_c(d, (CX, y), ln, fnt, al(col, f), stroke=0)
        y += 112
    if prm.get("sub"):
        f = clamp((t - len(lines) * 0.22 - 0.1) / 0.3)
        sf_ = font(F_MED, 38)
        tw = d.textlength(prm["sub"], font=sf_) + 60
        d.rounded_rectangle([CX - tw / 2, y - 2, CX + tw / 2, y + 66], radius=33, fill=al(ac, 0.18 * f), outline=al(ac, f), width=3)
        text_c(d, (CX, y + 32), prm["sub"], sf_, al(TEXT, f))


def sc_tokens(d, img, prm, p, t, ac, ctx):
    chunks, ids = prm["chunks"], prm.get("ids")
    cols = [ac, ACCENTS["cyan"], ACCENTS["violet"], ACCENTS["yellow"], ACCENTS["pink"], ACCENTS["green"]]
    fm = font(F_MONO, 92)
    word = "".join(chunks)
    full_w = d.textlength(word, font=fm)
    widths = [d.textlength(c, font=fm) for c in chunks]
    gap = 46
    split_w = sum(widths) + gap * (len(chunks) - 1) + 60 * len(chunks)
    sp = st(p, 0.18, 0.42)
    x_join = CX - full_w / 2
    x_split = CX - split_w / 2
    y = CY - 90
    for i, ch in enumerate(chunks):
        jx = x_join + sum(widths[:i])
        sx = x_split + sum(widths[:i]) + i * (gap + 60) + 30
        x = lerp(jx, sx, sp)
        if sp > 0:
            col = cols[i % len(cols)]
            pad = 30 * sp
            d.rounded_rectangle([x - pad, y - 74, x + widths[i] + pad, y + 74], radius=24,
                                fill=al(col, 0.20 * sp), outline=al(col, sp), width=4)
        tf = clamp(t / 0.6)
        n = len(ch)
        typed = int(clamp(tf * len(word) - sum(len(c) for c in chunks[:i]), 0, n))
        d.text((x, y), ch[:typed], font=fm, fill=al(TEXT, 1), anchor="lm")
        if ids:
            f = st(p, 0.48 + i * 0.06, 0.62 + i * 0.06)
            if f > 0:
                text_c(d, (x + widths[i] / 2, y + 140 + (1 - f) * 30), str(ids[i]), font(F_MONO, 50), al(cols[i % len(cols)], f))
                d.line([(x + widths[i] / 2, y + 84), (x + widths[i] / 2, y + 108)], fill=al(MUTED, f), width=3)
    if prm.get("note"):
        f = st(p, 0.72, 0.85)
        text_c(d, (CX, CY + 230), prm["note"], font(F_MED, 40), al(MUTED, f))


def sc_probs(d, img, prm, p, t, ac, ctx):
    opts, pick = prm["options"], prm.get("pick", 0)
    cf = st(p, 0, 0.12)
    card(d, [SC[0], SC[1] + 10, SC[2], SC[1] + 200], cf)
    chosen = opts[pick][0]
    fly = st(p, 0.72, 0.86)
    prompt = prm["prompt"] + ((" " + chosen) if fly >= 1 else "")
    fp = font(F_BOLD, 46)
    text_c(d, (SC[0] + 50, SC[1] + 105), prompt, fp, al(TEXT, cf), anchor="lm")
    if fly < 1 and int(t * 2.5) % 2 == 0:
        cx_ = SC[0] + 54 + d.textlength(prompt, font=fp)
        d.rectangle([cx_, SC[1] + 78, cx_ + 22, SC[1] + 132], fill=al(ac, cf))
    y0 = SC[1] + 290
    mx = max(o[1] for o in opts)
    for i, (w_, pr) in enumerate(opts):
        f = st(p, 0.12 + i * 0.07, 0.32 + i * 0.07)
        y = y0 + i * 118
        is_pick = i == pick
        hl = st(p, 0.62, 0.7) if is_pick else 0
        dim = 1 - 0.55 * st(p, 0.62, 0.7) * (not is_pick)
        text_c(d, (SC[0] + 30, y), w_, font(F_BOLD, 44), al(mixc(TEXT, ac, hl), f * dim), anchor="lm")
        bx0, bx1 = SC[0] + 330, SC[2] - 140
        d.rounded_rectangle([bx0, y - 30, bx1, y + 30], radius=30, fill=al((40, 45, 84), f * dim))
        bw = (bx1 - bx0) * (pr / mx) * f
        if bw > 12:
            d.rounded_rectangle([bx0, y - 30, bx0 + bw, y + 30], radius=30, fill=al(ac if is_pick or hl else mixc(ac, (90, 96, 150), 0.55), f * dim))
        text_c(d, (SC[2] - 20, y), f"{int(round(pr * 100 * f))}%", font(F_BOLD, 42), al(TEXT, f * dim), anchor="rm")
    if 0 < fly < 1:
        sx, sy = SC[0] + 30, y0 + pick * 118
        ex = SC[0] + 60 + d.textlength(prm["prompt"] + " ", font=fp)
        ey = SC[1] + 105
        text_c(d, (lerp(sx, ex, fly), lerp(sy, ey, fly)), chosen, font(F_BOLD, lerp(44, 46, fly)), al(ac, 1), anchor="lm")


def sc_versus(d, img, prm, p, t, ac, ctx):
    L, R = prm["left"], prm["right"]
    for side, it in ((0, L), (1, R)):
        f = st(p, 0.02 + side * 0.12, 0.2 + side * 0.12)
        x0 = SC[0] + side * 490 + (1 - f) * (-60 if side == 0 else 60)
        x1 = x0 + 470
        col = it.get("color")
        col = ACCENTS.get(col, ac) if isinstance(col, str) else (ac if side == 1 else (140, 148, 200))
        card(d, [x0, SC[1] + 60, x1, SC[3] - 60], f, line=mixc(LINE, col, 0.8))
        d.rounded_rectangle([x0, SC[1] + 60, x1, SC[1] + 170], radius=34, fill=al(col, 0.22 * f))
        text_c(d, ((x0 + x1) / 2, SC[1] + 115), it["title"], font(F_BOLD, 44), al(TEXT, f))
        fm = font(F_MONO if it.get("mono") else F_BOLD, 44 if not it.get("mono") else 58)
        nl = sum(len(wrap(d, ln, fm, 410)) for ln in it["lines"])
        y = (SC[1] + 170 + SC[3] - (200 if it.get("mark") else 60)) / 2 - (nl * 62 + (len(it["lines"]) - 1) * 34) / 2 + 31
        for j, ln in enumerate(it["lines"]):
            g = st(p, 0.22 + side * 0.15 + j * 0.07, 0.36 + side * 0.15 + j * 0.07)
            for k, sub in enumerate(wrap(d, ln, fm, 410)):
                text_c(d, ((x0 + x1) / 2, y + (1 - g) * 20), sub, fm, al(TEXT, g))
                y += 62
            y += 34
        mark = it.get("mark")
        if mark:
            g = st(p, 0.7, 0.8)
            mc = GOOD if mark == "check" else BAD
            mx_, my_ = (x0 + x1) / 2, SC[3] - 160
            d.ellipse([mx_ - 56, my_ - 56, mx_ + 56, my_ + 56], fill=al(mc, 0.2 * g), outline=al(mc, g), width=5)
            icon(d, mark, mx_, my_, 40 * back(g) if g > 0 else 1, mc, g)


def sc_rows(d, img, prm, p, t, ac, ctx):
    rows = prm["rows"]
    y = SC[1] + 70
    if prm.get("title"):
        f = st(p, 0, 0.12)
        text_c(d, (CX, y + 20), prm["title"], font(F_BOLD, 56), al(TEXT, f))
        y += 130
    mono = prm.get("mono")
    for i, r in enumerate(rows):
        f = st(p, 0.1 + i * 0.14, 0.28 + i * 0.14)
        bad = len(r) > 2 and r[2] == "bad"
        col = BAD if bad else ac
        x = SC[0] + (1 - f) * 80
        card(d, [x, y, x + 960, y + 150], f, line=mixc(LINE, col, 0.6))
        tf = font(F_BOLD, 36)
        tw = max(d.textlength(r[0], font=tf) + 50, 110)
        d.rounded_rectangle([x + 28, y + 45, x + 28 + tw, y + 105], radius=30, fill=al(col, f))
        text_c(d, (x + 28 + tw / 2, y + 75), r[0], tf, al((20, 20, 36), f))
        fnt = font(F_MONO, 44) if mono else font(F_BOLD, 42)
        txt = r[1]
        while d.textlength(txt, font=fnt) > 960 - tw - 90 and fnt.size > 28:
            fnt = font(F_MONO if mono else F_BOLD, fnt.size - 2)
        text_c(d, (x + 60 + tw, y + 75), txt, fnt, al(TEXT, f), anchor="lm")
        y += 180


def sc_flow(d, img, prm, p, t, ac, ctx):
    nodes = prm["nodes"]
    n = len(nodes)
    bad = prm.get("bad", [])
    if n <= 3:
        pos = [(SC[0] + 150 + i * (660 / max(n - 1, 1)), CY - 40) for i in range(n)]
    else:
        # zig-zag 2 columns
        pos = []
        for i in range(n):
            pos.append((CX - 230 if i % 2 == 0 else CX + 230, SC[1] + 110 + i * (760 / (n - 1))))
    for i in range(n - 1):
        f = st(p, 0.12 + i * 0.16, 0.26 + i * 0.16)
        if f <= 0:
            continue
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        ang = math.atan2(y1 - y0, x1 - x0)
        sx, sy = x0 + 116 * math.cos(ang), y0 + 116 * math.sin(ang)
        ex, ey = x1 - 116 * math.cos(ang), y1 - 116 * math.sin(ang)
        col = BAD if (i + 1) in bad else ac
        mx, my = lerp(sx, ex, f), lerp(sy, ey, f)
        glow_line(d, (sx, sy), (mx, my), col, 0.9, 5)
        if f >= 1:
            arrow_head(d, (sx, sy), (ex, ey), col, 1, 26)
            particles(d, (sx, sy), (ex, ey), t, col, 1, 2, 0.8, 7)
    for i, (x, y) in enumerate(pos):
        f = st(p, 0.02 + i * 0.16, 0.16 + i * 0.16)
        it = nodes[i]
        col = BAD if i in bad else ac
        node(d, x, y, it.get("label"), it.get("icon"), col, f, r=(92 if n <= 3 else 72), sub=it.get("sub"))


def sc_hub(d, img, prm, p, t, ac, ctx):
    nodes = prm["nodes"]
    n = len(nodes)
    lit = prm.get("lit")
    R = prm.get("radius", 330)
    cf = st(p, 0, 0.12)
    dimf = st(p, 0.62, 0.72) if lit is not None else 0
    pts = []
    for i in range(n):
        ang = -math.pi / 2 + i * 2 * math.pi / n + math.sin(t * 0.6) * 0.02
        pts.append((CX + R * math.cos(ang), CY + R * math.sin(ang)))
    for i, (x, y) in enumerate(pts):
        f = st(p, 0.12 + i * 0.06, 0.24 + i * 0.06)
        if f <= 0:
            continue
        on = 1 - dimf * (0 if (lit and i in lit) else 0.85)
        ang = math.atan2(y - CY, x - CX)
        sx, sy = CX + 118 * math.cos(ang), CY + 118 * math.sin(ang)
        ex, ey = x - 86 * math.cos(ang), y - 86 * math.sin(ang)
        glow_line(d, (sx, sy), (lerp(sx, ex, f), lerp(sy, ey, f)), ac, 0.85 * on, 5)
        if f >= 1:
            particles(d, (sx, sy), (ex, ey), t + i * 0.13, ac, on, 2, 0.7, 6)
    pulse = 1 + 0.04 * math.sin(t * 4)
    c = prm["center"]
    rr = 112 * pulse
    d.ellipse([CX - rr - 30, CY - rr - 30, CX + rr + 30, CY + rr + 30], fill=al(ac, 0.14 * cf))
    d.ellipse([CX - rr, CY - rr, CX + rr, CY + rr], fill=al(ac, cf), outline=al((255, 255, 255), cf), width=4)
    text_c(d, (CX, CY), c["label"], font(F_BOLD, c.get("size", 54)), al((18, 16, 34), cf))
    for i, (x, y) in enumerate(pts):
        f = st(p, 0.12 + i * 0.06, 0.24 + i * 0.06)
        on = 1 - dimf * (0 if (lit and i in lit) else 0.8)
        node(d, x, y, nodes[i].get("label"), nodes[i].get("icon"), ac, f, r=70, lit=on)


def sc_loop(d, img, prm, p, t, ac, ctx):
    stages = prm["stages"]
    n = len(stages)
    R = 300
    cf = st(p, 0, 0.12)
    pts = [(CX + R * math.cos(-math.pi / 2 + i * 2 * math.pi / n), CY + R * math.sin(-math.pi / 2 + i * 2 * math.pi / n)) for i in range(n)]
    # ring
    d.ellipse([CX - R, CY - R, CX + R, CY + R], outline=al((70, 78, 130), cf), width=6)
    laps = prm.get("laps", 1.0)
    head = (st(p, 0.1, 0.95) * laps * n) if p > 0.1 else 0
    seg = int(head) % n
    # bright arc travelling
    a0 = -90 + (head % n) * 360 / n - 100
    a1 = -90 + (head % n) * 360 / n
    for k, wdt in ((0.18, 30), (0.4, 16), (1, 7)):
        d.arc([CX - R, CY - R, CX + R, CY + R], a0, a1, fill=al(ac, k * cf), width=wdt)
    hx = CX + R * math.cos(math.radians(a1))
    hy = CY + R * math.sin(math.radians(a1))
    glow_dot(d, hx, hy, 11, ac, cf)
    for i, (x, y) in enumerate(pts):
        f = st(p, 0.04 + i * 0.05, 0.16 + i * 0.05)
        lit = 1.0 if i == seg else 0.35
        node(d, x, y, None, stages[i].get("icon"), ac, f, r=82, lit=lit)
        lx, ly = x, y + 132 if y >= CY - 10 else y - 128
        if abs(x - CX) > 50:
            lx, ly = x + (150 if x > CX else -150), y + 110
        text_c(d, (lx, ly), stages[i]["label"], font(F_BOLD, 40), al(mixc(MUTED, TEXT, lit), f))
    if prm.get("center"):
        lines = prm["center"].split("\n")
        for j, ln in enumerate(lines):
            text_c(d, (CX, CY - (len(lines) - 1) * 28 + j * 56), ln, font(F_BOLD if j == 0 else F_MED, 44 if j == 0 else 36), al(TEXT if j == 0 else MUTED, cf))


def sc_mesh(d, img, prm, p, t, ac, ctx):
    L, R = prm["left"], prm["right"]
    phase = prm.get("phase", "before")
    ys_l = [SC[1] + 110 + i * (620 / max(len(L) - 1, 1)) for i in range(len(L))]
    ys_r = [SC[1] + 110 + i * (620 / max(len(R) - 1, 1)) for i in range(len(R))]
    xl, xr = SC[0] + 110, SC[2] - 110
    if phase == "before":
        k = 0
        tot = len(L) * len(R)
        for i, yl in enumerate(ys_l):
            for j, yr in enumerate(ys_r):
                f = st(p, 0.15 + k * 0.5 / tot, 0.3 + k * 0.5 / tot)
                k += 1
                if f > 0:
                    col = mixc(BAD, ac, 0.0)
                    d.line([(xl + 70, yl), (lerp(xl + 70, xr - 70, f), lerp(yl, yr, f))], fill=al(col, 0.75), width=4)
        cnt = int(round(tot * st(p, 0.15, 0.7)))
        lab = prm.get("before_label", "{n} custom connectors").replace("{n}", str(cnt))
        col = BAD
    else:
        hx, hy = CX, CY
        f0 = st(p, 0.0, 0.25)
        for i, yl in enumerate(ys_l + ys_r):
            x = xl + 70 if i < len(ys_l) else xr - 70
            f = st(p, 0.15 + i * 0.05, 0.3 + i * 0.05)
            glow_line(d, (x, yl), (lerp(x, hx, f), lerp(yl, hy, f)), ac, 0.9, 5)
            if f >= 1:
                particles(d, (x, yl), (hx, hy), t + i * 0.2, ac, 1, 2, 0.6, 6)
        rr = 100 * (0.5 + 0.5 * back(f0))
        d.ellipse([hx - rr - 24, hy - rr - 24, hx + rr + 24, hy + rr + 24], fill=al(ac, 0.15 * f0))
        d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], fill=al(ac, f0), outline=al((255, 255, 255), f0), width=4)
        text_c(d, (hx, hy), prm.get("hub", "MCP"), font(F_BOLD, 52), al((18, 16, 34), f0))
        lab = prm.get("after_label", "")
        col = GOOD
    for i, yl in enumerate(ys_l):
        node(d, xl, yl, None, L[i]["icon"], (150, 160, 220), 1, r=58)
        text_c(d, (xl, yl + 88), L[i]["label"], font(F_BOLD, 32), al(TEXT, 1))
    for i, yr in enumerate(ys_r):
        node(d, xr, yr, None, R[i]["icon"], ac, 1, r=58)
        text_c(d, (xr, yr + 88), R[i]["label"], font(F_BOLD, 32), al(TEXT, 1))
    f = st(p, 0.2, 0.35)
    fl = font(F_BOLD, 50)
    tw = d.textlength(lab, font=fl) + 70
    d.rounded_rectangle([CX - tw / 2, SC[3] - 80, CX + tw / 2, SC[3]], radius=40, fill=al(mixc(CARD, col, 0.3), f), outline=al(col, f), width=4)
    text_c(d, (CX, SC[3] - 40), lab, fl, al(TEXT, f))


def sc_scatter(d, img, prm, p, t, ac, ctx):
    box = [SC[0] + 20, SC[1] + 40, SC[2] - 20, SC[3] - 40]
    f0 = st(p, 0, 0.1)
    card(d, box, f0 * 0.9)
    for gx in range(1, 6):
        x = lerp(box[0], box[2], gx / 6)
        d.line([(x, box[1] + 20), (x, box[3] - 20)], fill=al((52, 60, 104), f0), width=1)
    for gy in range(1, 6):
        y = lerp(box[1], box[3], gy / 6)
        d.line([(box[0] + 20, y), (box[2] - 20, y)], fill=al((52, 60, 104), f0), width=1)

    def P(pt):
        return lerp(box[0] + 60, box[2] - 60, pt["x"]), lerp(box[3] - 70, box[1] + 70, pt["y"])

    pts = prm["points"]
    by = {pt["label"]: pt for pt in pts}
    for i, (a_, b_) in enumerate(prm.get("arrows", [])):
        f = st(p, 0.45 + i * 0.12, 0.6 + i * 0.12)
        if f <= 0:
            continue
        (x0, y0), (x1, y1) = P(by[a_]), P(by[b_])
        ang = math.atan2(y1 - y0, x1 - x0)
        sx, sy = x0 + 26 * math.cos(ang), y0 + 26 * math.sin(ang)
        ex, ey = x1 - 26 * math.cos(ang), y1 - 26 * math.sin(ang)
        glow_line(d, (sx, sy), (lerp(sx, ex, f), lerp(sy, ey, f)), ac, 0.95, 5)
        if f >= 1:
            arrow_head(d, (sx, sy), (ex, ey), ac, 1, 24)
    q = prm.get("query")
    if q:
        f = st(p, 0.5, 0.65)
        qx, qy = P(by[q])
        R = prm.get("radius", 190) * back(f) if f > 0 else 0
        pul = 1 + 0.05 * math.sin(t * 5)
        d.ellipse([qx - R * pul, qy - R * pul, qx + R * pul, qy + R * pul], fill=al(ac, 0.12 * f), outline=al(ac, 0.8 * f), width=4)
    for i, pt in enumerate(pts):
        f = st(p, 0.08 + i * 0.05, 0.2 + i * 0.05)
        if f <= 0:
            continue
        x, y = P(pt)
        col = ACCENTS.get(pt.get("color", ""), ac) if pt.get("color") else ac
        if pt.get("dim"):
            col = (120, 126, 170)
        glow_dot(d, x, y, 12 * back(f), col, f)
        anc = pt.get("anchor", "lb")
        ox = 26 if anc[0] == "l" else -26
        text_c(d, (x + ox, y - 22), pt["label"], font(F_BOLD, 40), al(TEXT if not pt.get("dim") else MUTED, f), anchor=anc[0] + "s")


_noise_cache = {}


def _noise_assets(prm, ac):
    key = json.dumps(prm, sort_keys=True)
    if key in _noise_cache:
        return _noise_cache[key]
    S = 660
    tgt = Image.new("RGB", (S, S), (0, 0, 0))
    g = np.linspace(0, 1, S)[:, None]
    top, bot = np.array([40, 60, 140]), np.array([255, 150, 110])
    grad = (top * (1 - g[..., None]) + bot * g[..., None])
    grad = np.repeat(grad, S, axis=1).astype(np.uint8)
    tgt = Image.fromarray(grad, "RGB").convert("RGBA")
    dd = ImageDraw.Draw(tgt, "RGBA")
    dd.ellipse([S * .62, S * .08, S * .84, S * .30], fill=(255, 236, 170, 255))
    dd.rectangle([0, S * .8, S, S], fill=(30, 40, 70, 255))
    spr = C.render("pip", {"mood": "happy", "wave": 1, "t": 0.6}, size=440)
    tgt.alpha_composite(spr, ((S - spr.width) // 2, int(S * .97 - spr.height)))
    tgt = np.asarray(tgt.convert("RGB")).astype(np.float32)
    rng = np.random.default_rng(7)
    noise = rng.normal(127, 70, (S, S, 3)).clip(0, 255).astype(np.float32)
    _noise_cache[key] = (tgt, noise, S)
    return _noise_cache[key]


def sc_noise(d, img, prm, p, t, ac, ctx):
    tgt, noise, S = _noise_assets({"k": 1}, ac)
    a0, a1 = prm.get("a0", 0.0), prm.get("a1", 1.0)
    rev = prm.get("reverse", False)
    q = lerp(a0, a1, ease(p))
    steps = prm.get("steps", 50)
    stepq = math.floor(q * steps) / steps
    clean = stepq if not rev else 1 - stepq
    k = clean ** 1.6
    frame = tgt * k + noise * (1 - k)
    im = Image.fromarray(frame.clip(0, 255).astype(np.uint8), "RGB")
    if k < 0.98:
        im = im.filter(ImageFilter.GaussianBlur(radius=(1 - k) * 2.5))
    x0, y0 = int(CX - S / 2), int(SC[1] + 150)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=36, fill=255)
    d.rounded_rectangle([x0 - 8, y0 - 8, x0 + S + 8, y0 + S + 8], radius=42, fill=al(ac, 0.5))
    img.paste(im, (x0, y0), mask)
    lab = prm.get("label")
    if lab:
        fl = font(F_BOLD, 40)
        tw = d.textlength(lab, font=fl) + 60
        d.rounded_rectangle([CX - tw / 2, SC[1] + 40, CX + tw / 2, SC[1] + 116], radius=38, fill=al(CARD, 1), outline=al(ac, 1), width=3)
        text_c(d, (CX, SC[1] + 78), lab, fl, al(TEXT, 1))
    if prm.get("counter", True):
        stp = int(round(clean * steps)) if not rev else int(round((1 - clean) * steps))
        txt = f"step {stp}/{steps}" if not rev else f"noise added: {int((1 - clean) * 100)}%"
        text_c(d, (CX, y0 + S + 60), txt, font(F_MONO, 40), al(ac, 1))


def sc_curve(d, img, prm, p, t, ac, ctx):
    ys = prm["values"]
    box = [SC[0] + 90, SC[1] + 120, SC[2] - 40, SC[3] - 180]
    f0 = st(p, 0, 0.12)
    d.line([(box[0], box[1]), (box[0], box[3]), (box[2], box[3])], fill=al(MUTED, f0), width=4)
    if prm.get("ylabel"):
        text_c(d, (box[0], box[1] - 50), prm["ylabel"], font(F_BOLD, 36), al(MUTED, f0), anchor="lm")
    lo, hi = min(ys) - 8, max(ys) + 6
    n = len(ys)
    pts = [(lerp(box[0] + 40, box[2] - 40, i / (n - 1)), lerp(box[3] - 20, box[1] + 20, (v - lo) / (hi - lo))) for i, v in enumerate(ys)]
    # dense smooth path
    dense = []
    for i in range(n - 1):
        for s_ in range(12):
            u = s_ / 12
            uu = (1 - math.cos(u * math.pi)) / 2
            dense.append((lerp(pts[i][0], pts[i + 1][0], u), lerp(pts[i][1], pts[i + 1][1], uu)))
    dense.append(pts[-1])
    f = st(p, 0.12, 0.6)
    m = max(2, int(len(dense) * f))
    seg = dense[:m]
    poly = seg + [(seg[-1][0], box[3]), (seg[0][0], box[3])]
    d.polygon(poly, fill=al(ac, 0.16))
    for k, wdt in ((0.18, 26), (0.4, 14), (1, 7)):
        d.line(seg, fill=al(ac, k), width=wdt, joint="curve")
    glow_dot(d, seg[-1][0], seg[-1][1], 10, ac, 1)
    for i, lab in enumerate(prm.get("xlabels", [])):
        x = lerp(box[0] + 40, box[2] - 40, i / max(len(prm["xlabels"]) - 1, 1))
        text_c(d, (x, box[3] + 50), lab, font(F_BOLD, 38), al(TEXT, f0))
    mk = prm.get("mark")
    if mk is not None:
        g = st(p, 0.65, 0.78)
        x, y = pts[mk]
        d.ellipse([x - 40, y - 40, x + 40, y + 40], outline=al(BAD, g), width=5)
        if prm.get("mark_label"):
            text_c(d, (x, y + 90), prm["mark_label"], font(F_BOLD, 40), al(BAD, g))
    if prm.get("note"):
        text_c(d, (CX, SC[3] - 50), prm["note"], font(F_MED, 32), al(MUTED, st(p, 0.7, 0.8)))


def sc_bars(d, img, prm, p, t, ac, ctx):
    rows = prm["rows"]
    unit = prm.get("unit", "")
    mx = max(max(r[1] for r in rows), prm.get("marker", {}).get("value", 0)) * 1.08
    x0, x1 = SC[0] + 230, SC[2] - 30
    rowh = min(170, 700 / len(rows))
    y0 = CY - rowh * len(rows) / 2 + rowh / 2
    if prm.get("title"):
        text_c(d, (CX, SC[1] + 60), prm["title"], font(F_BOLD, 50), al(TEXT, st(p, 0, 0.1)))
    for i, r in enumerate(rows):
        lab, val = r[0], r[1]
        f = st(p, 0.08 + i * 0.12, 0.38 + i * 0.12)
        y = y0 + i * rowh
        text_c(d, (x0 - 30, y), lab, font(F_BOLD, 40), al(TEXT, st(p, 0.05 + i * 0.12, 0.15 + i * 0.12)), anchor="rm")
        d.rounded_rectangle([x0, y - 44, x1, y + 44], radius=22, fill=al((36, 40, 78), 1))
        w_ = (x1 - x0) * val / mx * f
        col = ac if len(r) < 3 else ACCENTS.get(r[2], ac)
        if w_ > 10:
            d.rounded_rectangle([x0, y - 44, x0 + w_, y + 44], radius=22, fill=al(col, 1))
        num = val * f
        txt = (f"{num:.0f}" if val >= 10 else f"{num:.1f}") + unit
        tx = x0 + w_ + 20
        anchor = "lm"
        fill = TEXT
        if w_ > 260:
            tx, anchor, fill = x0 + w_ - 20, "rm", (18, 16, 34)
        text_c(d, (tx, y), txt, font(F_BOLD, 42), al(fill, f), anchor=anchor)
    mk = prm.get("marker")
    if mk:
        g = st(p, 0.6, 0.72)
        x = x0 + (x1 - x0) * mk["value"] / mx
        yy = y0 - rowh / 2 - 30
        while yy < y0 + rowh * (len(rows) - 0.5) + 20:
            d.line([(x, yy), (x, yy + 22)], fill=al(GOOD, g), width=5)
            yy += 38
        text_c(d, (x, y0 - rowh / 2 - 60), mk["label"], font(F_BOLD, 36), al(GOOD, g))


def sc_webpage(d, img, prm, p, t, ac, ctx):
    box = [SC[0] + 10, SC[1] + 20, SC[2] - 10, SC[3] - 20]
    f0 = st(p, 0, 0.12)
    d.rounded_rectangle(box, radius=30, fill=al((246, 246, 250), f0))
    d.rounded_rectangle([box[0], box[1], box[2], box[1] + 86], radius=30, fill=al((220, 222, 232), f0))
    d.rectangle([box[0], box[1] + 50, box[2], box[1] + 86], fill=al((220, 222, 232), f0))
    for k, c_ in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([box[0] + 30 + k * 40, box[1] + 30, box[0] + 54 + k * 40, box[1] + 54], fill=al(c_, f0))
    d.rounded_rectangle([box[0] + 170, box[1] + 22, box[2] - 40, box[1] + 64], radius=20, fill=al((255, 255, 255), f0))
    text_c(d, (box[0] + 196, box[1] + 43), prm.get("url", "bestgear-reviews.com"), font(F_MED, 26), al((110, 110, 130), f0), anchor="lm")
    text_c(d, (box[0] + 50, box[1] + 160), prm["title"], font(F_BOLD, 56), al((24, 22, 38), f0), anchor="lm")
    y = box[1] + 250
    for k in range(9):
        wl = [0.92, 0.85, 0.95, 0.7, 0.9, 0.8, 0.94, 0.6, 0.88][k]
        d.rounded_rectangle([box[0] + 50, y, box[0] + 50 + (box[2] - box[0] - 100) * wl, y + 22], radius=11, fill=al((205, 207, 220), f0))
        y += 50
        if k == 4:
            hy = y
            y += 150
    reveal = st(p, prm.get("at", 0.45), prm.get("at", 0.45) + 0.12)
    hidden = prm["hidden"]
    fh = font(F_BOLD, 34)
    lines = wrap(d, hidden, fh, box[2] - box[0] - 160)
    col = mixc((246, 246, 250), BAD, reveal)
    if reveal > 0:
        d.rounded_rectangle([box[0] + 40, hy - 6, box[2] - 40, hy + 26 + 46 * len(lines)], radius=16,
                            fill=al(BAD, 0.12 * reveal), outline=al(BAD, reveal), width=4)
    for j, ln in enumerate(lines):
        d.text((box[0] + 70, hy + 14 + j * 46), ln, font=fh, fill=al(col, f0))
    if reveal > 0:
        tag = "HIDDEN TEXT"
        tf = font(F_BOLD, 30)
        tw = d.textlength(tag, font=tf) + 40
        d.rounded_rectangle([box[2] - 60 - tw, hy - 30, box[2] - 60, hy + 18], radius=24, fill=al(BAD, reveal))
        text_c(d, (box[2] - 60 - tw / 2, hy - 6), tag, tf, al((255, 255, 255), reveal))


def sc_experts(d, img, prm, p, t, ac, ctx):
    toks = prm["tokens"]
    choices = prm["choices"]
    ne = prm.get("n", 8)
    cols, rows_ = 4, 2
    ex_pos = []
    for k in range(ne):
        cx_ = SC[0] + 130 + (k % cols) * 233
        cy_ = SC[3] - 330 + (k // cols) * 180
        ex_pos.append((cx_, cy_))
    rx, ry = CX, SC[1] + 330
    f0 = st(p, 0, 0.1)
    span = 0.85 / len(toks)
    cur = min(int(clamp((p - 0.1) / span, 0, len(toks) - 1e-6)), len(toks) - 1) if p > 0.1 else -1
    lp = clamp(((p - 0.1) - cur * span) / span) if cur >= 0 else 0
    # token row
    ft = font(F_MONO, 54)
    xs = []
    tot = sum(d.textlength(tk, font=ft) + 50 for tk in toks) + 20 * (len(toks) - 1)
    x = CX - tot / 2
    for i, tk in enumerate(toks):
        w_ = d.textlength(tk, font=ft) + 50
        xs.append((x, w_))
        x += w_ + 20
    for i, tk in enumerate(toks):
        x, w_ = xs[i]
        on = i == cur
        done = i < cur
        y = SC[1] + 90 + (ease(lp) * 0 if not on else 0)
        col = ac if on else ((120, 126, 170) if not done else (90, 96, 140))
        d.rounded_rectangle([x, y - 40, x + w_, y + 40], radius=20, fill=al(col, 0.25 * f0 if not on else 0.35), outline=al(col, f0), width=4)
        text_c(d, (x + w_ / 2, y), tk, ft, al(TEXT if not done else MUTED, f0))
    # router
    d.rounded_rectangle([rx - 170, ry - 64, rx + 170, ry + 64], radius=32, fill=al(CARD, f0), outline=al(ac, f0), width=5)
    text_c(d, (rx, ry), "Router", font(F_BOLD, 46), al(TEXT, f0))
    if cur >= 0:
        x, w_ = xs[cur]
        tx, ty = x + w_ / 2, SC[1] + 130
        k1 = st(lp, 0.0, 0.3)
        glow_line(d, (tx, ty), (lerp(tx, rx, k1), lerp(ty, ry - 64, k1)), ac, 0.9, 5)
        k2 = st(lp, 0.3, 0.6)
        for e in choices[cur]:
            ex, ey = ex_pos[e]
            if k2 > 0:
                glow_line(d, (rx, ry + 64), (lerp(rx, ex, k2), lerp(ry + 64, ey - 70, k2)), ac, 0.95, 6)
            if k2 >= 1:
                particles(d, (rx, ry + 64), (ex, ey - 70), t, ac, 1, 2, 1.1, 7)
    for k, (x, y) in enumerate(ex_pos):
        lit = 1.0 if (cur >= 0 and k in choices[cur] and st(lp, 0.45, 0.6) > 0) else 0.15
        f = st(p, 0.02 + k * 0.02, 0.1 + k * 0.02)
        d.rounded_rectangle([x - 95, y - 70, x + 95, y + 70], radius=26,
                            fill=al(mixc(CARD, ac, 0.35 * lit), f), outline=al(mixc((70, 78, 130), ac, lit), f), width=4)
        if lit > 0.5:
            d.rounded_rectangle([x - 110, y - 85, x + 110, y + 85], radius=34, outline=al(ac, 0.3), width=8)
        icon(d, "brain", x, y - 8, 34, mixc((110, 116, 160), (255, 255, 255), lit), f)
        text_c(d, (x, y + 46), f"Expert {k + 1}", font(F_MED, 24), al(mixc(MUTED, TEXT, lit), f))
    text_c(d, (CX, SC[3] - 50), prm.get("note", ""), font(F_BOLD, 36), al(MUTED, st(p, 0.2, 0.3)))


def sc_counter(d, img, prm, p, t, ac, ctx):
    v = prm["value"]
    f = st(p, 0.02, 0.45)
    num = v * f
    s = prm.get("prefix", "") + (f"{num:,.0f}" if v >= 10 else f"{num:.1f}") + prm.get("suffix", "")
    sz = 210
    fnt = font(F_BOLD, sz)
    while d.textlength(s, font=fnt) > 940:
        sz -= 8
        fnt = font(F_BOLD, sz)
    pulse = 1 + 0.03 * math.sin(t * 6) * (f >= 1)
    d.ellipse([CX - 380 * pulse, CY - 380 * pulse, CX + 380 * pulse, CY + 380 * pulse], fill=al(ac, 0.08))
    text_c(d, (CX, CY - 40), s, fnt, al(ac, 1))
    text_c(d, (CX, CY + 120), prm.get("label", ""), font(F_BOLD, 50), al(TEXT, st(p, 0.15, 0.3)))
    if prm.get("sub"):
        text_c(d, (CX, CY + 200), prm["sub"], font(F_MED, 38), al(MUTED, st(p, 0.3, 0.45)))


def sc_docs(d, img, prm, p, t, ac, ctx):
    labels = prm.get("labels", ["Start", "Middle", "End"])
    n = len(labels)
    for i, lab in enumerate(labels):
        f = st(p, 0.05 + i * 0.12, 0.2 + i * 0.12)
        x0 = SC[0] + 40 + i * 320
        x1 = x0 + 260
        y0, y1 = SC[1] + 80 + (1 - f) * 40, SC[3] - 160 + (1 - f) * 40
        card(d, [x0, y0, x1, y1], f, fill=(236, 238, 246), line=(200, 204, 222), r=20)
        nlines = 18
        key = [1, nlines // 2, nlines - 2][min(i, 2)] if n == 3 else int(i * (nlines - 1) / max(n - 1, 1))
        for k in range(nlines):
            yy = y0 + 40 + k * ((y1 - y0 - 70) / nlines)
            wl = 0.55 + 0.4 * ((k * 37 + i * 11) % 10) / 10
            if k == key:
                g = st(p, 0.3 + i * 0.1, 0.42 + i * 0.1)
                d.rounded_rectangle([x0 + 18, yy - 10, x1 - 18, yy + 22], radius=10, fill=al(mixc((190, 194, 214), ac, g), f))
            else:
                d.rounded_rectangle([x0 + 24, yy, x0 + 24 + (x1 - x0 - 48) * wl, yy + 12], radius=6, fill=al((190, 194, 214), f))
        text_c(d, ((x0 + x1) / 2, SC[3] - 100), lab, font(F_BOLD, 42), al(TEXT, f))
    if prm.get("note"):
        text_c(d, (CX, SC[3] - 30), prm["note"], font(F_MED, 34), al(ac, st(p, 0.6, 0.72)))


SCENES = {"slam": sc_slam, "tokens": sc_tokens, "probs": sc_probs, "versus": sc_versus, "rows": sc_rows,
          "flow": sc_flow, "hub": sc_hub, "loop": sc_loop, "mesh": sc_mesh, "scatter": sc_scatter,
          "noise": sc_noise, "curve": sc_curve, "bars": sc_bars, "webpage": sc_webpage,
          "experts": sc_experts, "counter": sc_counter, "docs": sc_docs}


# ------------------------------------------------------------------ frame furniture
def make_bg(ac):
    y = np.linspace(0, 1, H)[:, None, None]
    top, bot = np.array([9, 11, 26]), np.array([20, 16, 44])
    bg = np.repeat(top * (1 - y) + bot * y, W, axis=1)
    xx, yy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    vig = 1 - 0.35 * np.clip(xx ** 2 * 0.7 + yy ** 2 * 0.5, 0, 1)
    bg = (bg * vig[..., None]).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(bg, "RGB").convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    for yy_ in range(40, H, 60):
        for xx_ in range(30 + (yy_ // 60) % 2 * 30, W, 60):
            d.ellipse([xx_ - 2, yy_ - 2, xx_ + 2, yy_ + 2], fill=(255, 255, 255, 18))
    return img


def make_blob(col, r=430):
    s = r * 2
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([r * .35, r * .35, s - r * .35, s - r * .35], fill=col + (70,))
    return im.filter(ImageFilter.GaussianBlur(r * 0.3))


def draw_title(d, img, text, t, ac):
    f = 1.0
    big = 1 - st(t, 1.4, 1.9)  # big for the first ~1.5s
    sz = lerp(62, 86, big) * (0.6 + 0.4 * back(f))
    fnt = font(F_BOLD, sz)
    lines = wrap(d, text, fnt, 960)
    if len(lines) == 2:
        ws_ = text.split()
        best = min(range(1, len(ws_)), key=lambda k: max(d.textlength(" ".join(ws_[:k]), font=fnt), d.textlength(" ".join(ws_[k:]), font=fnt)))
        lines = [" ".join(ws_[:best]), " ".join(ws_[best:])]
    y = lerp(150, 200, big)
    lh = sz * 1.18
    for i, ln in enumerate(lines):
        w_ = d.textlength(ln, font=fnt)
        d.rounded_rectangle([CX - w_ / 2 - 26, y - lh / 2 - 4, CX + w_ / 2 + 26, y + lh / 2 + 4], radius=18,
                            fill=al((255, 255, 255), f) if i == 0 else al(ac, f))
        text_c(d, (CX, y), ln, fnt, al((16, 14, 30), f))
        y += lh + 8


def draw_progress(d, frac, ac):
    d.rounded_rectangle([40, 52, W - 40, 62], radius=5, fill=(255, 255, 255, 30))
    d.rounded_rectangle([40, 52, 40 + (W - 80) * clamp(frac), 62], radius=5, fill=al(ac, 1))


def draw_caption(d, seg, t, ac):
    words = seg["beat"]["say"].split()
    dur = seg["end"] - seg["start"]
    wts = np.array([len(w) + 2 for w in words], float)
    cum = np.cumsum(wts) / wts.sum() * dur
    k = min(int(np.searchsorted(cum, t - seg["start"])), len(words) - 1)
    g0 = (k // 3) * 3
    chunk = words[g0:g0 + 3]
    sz = 70
    while True:
        f = font(F_BOLD, sz)
        ws = [d.textlength(w, font=f) for w in chunk]
        if sum(ws) + 26 * (len(ws) - 1) <= W - 120 or sz <= 44:
            break
        sz -= 4
    x = CX - (sum(ws) + 26 * (len(ws) - 1)) / 2
    y = 1372
    for i, w in enumerate(chunk):
        on = g0 + i == k
        col = ac if on else (255, 255, 255)
        d.text((x, y), w, font=f, fill=col + (255,), stroke_width=10, stroke_fill=(10, 8, 22), anchor="ls")
        x += ws[i] + 26


def draw_pip(img, t, seg, end_card):
    talking = seg is not None and seg["start"] <= t <= seg["end"]
    mouth = 0.0
    if talking:
        i = int((t - seg["start"]) * FPS)
        mouth = float(seg["env"][min(i, len(seg["env"]) - 1)]) ** 0.8
    blink = 1.0 if (t * 0.9) % 3.3 < 0.12 else 0.0
    mood = (seg["beat"].get("mood", "happy") if seg else "happy")
    bob = -abs(math.sin(t * 8)) * 12 if talking else math.sin(t * 2) * 4
    wave = 1 if (end_card or (seg and seg["beat"].get("wave") and talking)) else 0
    spr = C.render("pip", {"t": t, "mouth": mouth, "blink": blink, "mood": mood, "look": -0.4,
                           "bob": bob, "wave": wave}, size=420)
    img.paste(spr, (int(CX - spr.width / 2), int(1915 - spr.height)), spr)


def draw_end(d, t, ac):
    f = ease(t / 0.35)
    card(d, [90, 640 - (1 - f) * 60, W - 90, 1120 - (1 - f) * 60], f, line=ac)
    text_c(d, (CX, 780 - (1 - f) * 60), "Follow for more", font(F_BOLD, 76), al(TEXT, f))
    text_c(d, (CX, 880 - (1 - f) * 60), "AI, explained by Pip", font(F_BOLD, 60), al(ac, f))
    text_c(d, (CX, 990 - (1 - f) * 60), "New explainer every day", font(F_MED, 40), al(MUTED, f))


# ------------------------------------------------------------------ audio
def build_audio(script, model_dir):
    pip = C.CAST["pip"]
    scale = 1.0
    for _ in range(4):
        clips = []
        for b in script["beats"]:
            c = C.CAST[b.get("who", "pip")]
            x, sr = voice.say(model_dir, b["say"], c["voice"], min(c["speed"] * scale, 1.4))
            clips.append(x)
        speech = 0.2 + sum(len(x) / SR + GAP for x in clips)
        if speech + END <= MAX_LEN or scale >= 1.25:
            break
        scale *= (speech + END) / (MAX_LEN - 0.2)
    total = speech + END
    n = int(total * SR) + SR // 10
    vox = np.zeros(n, np.float32)
    t = 0.2
    tl = []
    for b, x in zip(script["beats"], clips):
        i = int(t * SR)
        vox[i:i + len(x)] += x
        tl.append({"start": t, "end": t + len(x) / SR, "beat": b, "env": _envelope(x)})
        t += len(x) / SR + GAP
    vox = vox / (np.max(np.abs(vox)) or 1) * 0.9
    return tl, vox, total


def mix(vox, total, tl, scene_starts, seed):
    n = len(vox)
    bed = music_bed(total, seed)[:n]
    bed = np.pad(bed, (0, n - len(bed)))
    duck = np.ones(n, np.float32) * 0.6
    for s in tl:
        duck[int(s["start"] * SR): int(s["end"] * SR)] = 0.26
    duck = np.convolve(duck, np.ones(2400) / 2400, mode="same")
    out = vox + bed * duck
    wh, pop = sfx_whoosh() * 0.8, sfx_pop()

    def add(sig, at):
        i = int(at * SR)
        if 0 <= i < n:
            out[i:i + len(sig)] += sig[: n - i]

    for k, s in enumerate(scene_starts):
        add(wh if k else pop, s - (0.18 if k else 0))
        add(pop * 0.6, s + 0.12)
    add(wh, total - END - 0.2)
    return (out / (np.max(np.abs(out)) or 1) * 0.95).astype(np.float32)


# ------------------------------------------------------------------ render
def render(script, out, model_dir):
    ac = ACCENTS[script.get("accent", "orange")]
    tl, vox, total = build_audio(script, model_dir)
    # scene timeline: each scene starts at its beat and lasts until the next scene
    scenes = []
    for i, s in enumerate(tl):
        v = s["beat"].get("vis")
        if v:
            span = v.get("span", 1)
            end_i = min(i + span, len(tl)) - 1
            scenes.append({"vis": v, "start": s["start"] - 0.1, "anim_end": tl[end_i]["end"]})
    for k, sc in enumerate(scenes):
        sc["end"] = scenes[k + 1]["start"] if k + 1 < len(scenes) else total - END
    starts = [sc["start"] for sc in scenes]
    if scenes:
        scenes[0]["start"] = -0.45  # frame one already shows the hook visual (it's the thumbnail)
    seed = sum(ord(ch) for ch in script["id"])
    audio = mix(vox, total, tl, starts, seed)
    wav = out[:-4] + ".wav"
    sf.write(wav, audio, SR)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-af", "loudnorm=I=-14:TP=-1.2:LRA=11", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    bg = make_bg(ac)
    b1 = make_blob(ac)
    b2 = make_blob(ACCENTS["violet"] if script.get("accent") != "violet" else ACCENTS["cyan"])
    speech_end = total - END
    for fr in range(int(total * FPS)):
        t = fr / FPS
        base = bg.copy()
        base.alpha_composite(b1, (int(-250 + 120 * math.sin(t * 0.35)), int(300 + 140 * math.cos(t * 0.27))))
        base.alpha_composite(b2, (int(450 + 110 * math.cos(t * 0.3)), int(800 + 120 * math.sin(t * 0.4))))
        base = base.convert("RGB")
        img = base.copy()
        d = ImageDraw.Draw(img, "RGBA")
        end_card = t >= speech_end
        seg = None
        for s in tl:
            if s["start"] - GAP <= t <= s["end"] + GAP:
                seg = s
        if seg is None:
            seg = tl[0] if t < tl[0]["start"] else tl[-1]
        if not end_card:
            cur = None
            for sc in scenes:
                if sc["start"] <= t:
                    cur = sc
            if cur is None and scenes:
                cur = scenes[0]
            if cur:
                ts = max(t - cur["start"], 0)
                p = clamp(ts / max(cur["anim_end"] - cur["start"], 0.5))
                out_f = clamp((cur["end"] - t) / 0.15) if cur is not scenes[-1] else 1
                SCENES[cur["vis"]["type"]](d, img, cur["vis"], p, ts, ac, {})
                if ts < 0.2:
                    z = 1 + 0.06 * (1 - ease(ts / 0.2))
                    lw, lh = int(W * z), int(H * z)
                    oy = int((lh - H) * CY / H)
                    img = img.resize((lw, lh), Image.BILINEAR).crop(((lw - W) // 2, oy, (lw - W) // 2 + W, oy + H))
                a = min(out_f, clamp(ts / 0.2))
                if a < 1:
                    img = Image.blend(base, img, a)
                d = ImageDraw.Draw(img, "RGBA")
                if ts < 0.12:
                    d.rectangle([0, 0, W, H], fill=(255, 255, 255, int(40 * (1 - ts / 0.12))))
            draw_title(d, img, script["hook"], t, ac)
        else:
            draw_end(d, t - speech_end, ac)
        if not end_card and seg["start"] <= t <= seg["end"]:
            draw_caption(d, seg, t, ac)
        draw_pip(img, t, None if end_card else seg, end_card)
        draw_progress(d, t / total, ac)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    return total


def write_post(s, path, total):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{s['title']}\nLength: {total:.1f}s\n\nCAPTION\n{s['caption']}\n\n")
        if s.get("base44", True):
            fh.write("Build your own AI app on Base44. Code AT100 = $100 in free credits (link in bio).\n\n")
        fh.write(" ".join("#" + h.lstrip("#") for h in s["hashtags"]) + "\n")
        if s.get("sources"):
            fh.write("\nSOURCES (for you, not for posting)\n" + "\n".join("- " + x for x in s["sources"]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--preview", type=float, nargs="*", help="only dump frames at these seconds")
    a = ap.parse_args()
    s = json.load(open(a.script, encoding="utf-8"))
    total = render(s, a.out, a.model)
    write_post(s, a.out[:-4] + "_post.txt", total)
    print(f"{a.out}: {total:.1f}s")
