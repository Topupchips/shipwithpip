"""Hand-painted cartoon reels: watercolor backgrounds, boiling brush-painted characters, paper-note panels,
hand-lettered hooks, rehook banners and a comment-bait sticker.

usage: python paint.py script.json out.mp4 --model <kokoro-multi-lang-v1_0 dir> [--stills out.png]
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import characters as C  # noqa: E402
import voice  # noqa: E402
from make_reel import _envelope, music_bed, sfx_pop, sfx_whoosh  # noqa: E402

W, H, FPS, SR = 1080, 1920, 30, 24000
A = os.path.join(HERE, "assets")
F_BRUSH = os.path.join(A, "CaveatBrush.ttf")
F_HAND = os.path.join(A, "PatrickHand.ttf")
F_GOCHI = os.path.join(A, "GochiHand.ttf")
INK = (44, 30, 52)
COL = {"ink": (44, 30, 52), "red": (214, 48, 64), "green": (30, 140, 86), "blue": (44, 88, 200), "orange": (236, 110, 30), "pink": (222, 60, 140), "purple": (110, 70, 200)}
VOICES = {"pip": ("af_heart", 1.0), "dex": ("am_puck", 1.0), "flop": ("am_adam", 0.9)}
GAP = 0.2
_fc = {}


def font(p, s):
    k = (p, int(s))
    if k not in _fc:
        _fc[k] = ImageFont.truetype(p, int(s))
    return _fc[k]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    return 1 - (1 - clamp(x)) ** 3


def back_out(x):
    x = clamp(x)
    c1, c3 = 1.9, 2.9
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ noise + watercolor
def fbm(h, w, seed, scales=((96, 0.55), (24, 0.3), (6, 0.15))):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    for s, a in scales:
        n = rng.standard_normal((h // s + 3, w // s + 3)).astype(np.float32)
        out += zoom(n, s, order=1)[:h, :w] * a
    return out / (out.std() + 1e-6)


BW, BH = 1242, 2208


def wc_shape(cv, draw_fn, color, seed, alpha=0.95, wobble=7.0, edge=0.22, tex=0.09):
    """Paint one watercolor shape onto float canvas cv (BHxBWx3): wobbly edge, pigment variation, darker rim."""
    m = Image.new("L", (BW, BH), 0)
    draw_fn(ImageDraw.Draw(m))
    m = np.asarray(m, np.float32) / 255.0
    if m.max() <= 0:
        return
    dx = fbm(BH, BW, seed, ((70, 1.0), (18, 0.4))) * wobble
    dy = fbm(BH, BW, seed + 1, ((70, 1.0), (18, 0.4))) * wobble
    yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
    m = map_coordinates(m, [yy + dy, xx + dx], order=1, mode="nearest")
    m = gaussian_filter(m, 1.1)
    rim = np.clip(m - gaussian_filter(m, 9), 0, 1)
    tx = fbm(BH, BW, seed + 2, ((40, 0.7), (10, 0.3)))
    col = np.array(color, np.float32)[None, None, :] * (1 + tex * tx[..., None])
    col = col * (1 - edge * np.clip(rim * 3.2, 0, 1)[..., None])
    a = (m * alpha)[..., None]
    cv[:] = cv * (1 - a) + col * a


def grad(top, bot, y0, y1):
    ys = np.clip((np.arange(BH, dtype=np.float32) - y0) / max(1, y1 - y0), 0, 1)[:, None, None]
    return (np.array(top, np.float32) * (1 - ys) + np.array(bot, np.float32) * ys) * np.ones((1, BW, 1), np.float32)


def blotches(cv, seed, strength=0.10):
    n = fbm(BH, BW, seed, ((260, 0.7), (80, 0.3)))
    cv *= (1 + strength * n[..., None])


def paint_bg(kind):
    rng = random.Random(hash(kind) % 1000)
    R = lambda a, b: rng.uniform(a, b)  # noqa: E731
    if kind == "study":
        cv = grad((252, 224, 178), (244, 190, 150), 0, int(BH * 0.74))
        blotches(cv, 5, 0.09)
        floor = lambda d: d.rectangle([0, int(BH * 0.74), BW, BH], fill=255)  # noqa: E731
        wc_shape(cv, floor, (150, 98, 66), 11, 0.97)
        for i in range(9):
            y = int(BH * 0.74) + 70 + i * 120
            wc_shape(cv, lambda d, y=y: d.rectangle([0, y, BW, y + 5], fill=255), (110, 70, 48), 20 + i, 0.5, wobble=4)
        # window with night sky + moon
        win = lambda d: d.rounded_rectangle([int(BW * 0.12), int(BH * 0.08), int(BW * 0.62), int(BH * 0.3)], radius=26, fill=255)  # noqa: E731
        wc_shape(cv, win, (30, 40, 110), 31, 0.98)
        wc_shape(cv, lambda d: d.ellipse([int(BW * 0.44), int(BH * 0.11), int(BW * 0.54), int(BH * 0.145)], fill=255), (255, 240, 190), 32, 0.98)
        for i in range(14):
            x, y = R(0.15, 0.58) * BW, R(0.09, 0.28) * BH
            r = R(4, 9)
            wc_shape(cv, lambda d, x=x, y=y, r=r: d.ellipse([x - r, y - r, x + r, y + r], fill=255), (255, 246, 200), 40 + i, 0.9, wobble=2)
        for (x0, y0, x1, y1) in ((0.115, 0.075, 0.625, 0.088), (0.115, 0.292, 0.625, 0.305), (0.115, 0.075, 0.13, 0.305), (0.605, 0.075, 0.625, 0.305), (0.365, 0.075, 0.38, 0.305)):
            wc_shape(cv, lambda d, b=(x0, y0, x1, y1): d.rectangle([b[0] * BW, b[1] * BH, b[2] * BW, b[3] * BH], fill=255), (130, 84, 56), 60 + int(x0 * 100), 0.97, wobble=4)
        # bookshelf
        wc_shape(cv, lambda d: d.rectangle([int(BW * 0.68), int(BH * 0.05), int(BW * 0.97), int(BH * 0.72)], fill=255), (118, 74, 52), 70, 0.97)
        for row in range(5):
            y1 = int(BH * (0.14 + row * 0.115))
            x = BW * 0.70
            while x < BW * 0.95:
                bw, bh = R(26, 46), R(70, 130)
                col = rng.choice([(214, 60, 80), (60, 130, 200), (240, 190, 70), (90, 170, 120), (160, 90, 190), (240, 130, 60)])
                wc_shape(cv, lambda d, x=x, bw=bw, bh=bh, y1=y1: d.rectangle([x, y1 - bh, x + bw, y1], fill=255), col, 100 + int(x) + row, 0.95, wobble=3, edge=0.3)
                x += bw + 4
            wc_shape(cv, lambda d, y1=y1: d.rectangle([BW * 0.68, y1, BW * 0.97, y1 + 12], fill=255), (92, 56, 40), 130 + row, 0.97, wobble=3)
        # lamp
        wc_shape(cv, lambda d: d.rectangle([int(BW * 0.06) - 6, int(BH * 0.38), int(BW * 0.06) + 6, int(BH * 0.72)], fill=255), (90, 60, 50), 140, 0.97, wobble=3)
        wc_shape(cv, lambda d: d.polygon([(BW * 0.0, BH * 0.38), (BW * 0.12, BH * 0.38), (BW * 0.1, BH * 0.31), (BW * 0.02, BH * 0.31)], fill=255), (255, 214, 100), 141, 0.95)
        wc_shape(cv, lambda d: d.ellipse([-120, BH * 0.34, BW * 0.4, BH * 0.62], fill=255), (255, 236, 160), 142, 0.22, wobble=20, edge=0)
        wc_shape(cv, lambda d: d.ellipse([BW * 0.22, BH * 0.8, BW * 0.9, BH * 0.93], fill=255), (206, 84, 98), 150, 0.9, wobble=10)
    elif kind == "city":
        cv = grad((26, 28, 76), (84, 60, 130), 0, int(BH * 0.74))
        blotches(cv, 6, 0.1)
        wc_shape(cv, lambda d: d.ellipse([BW * 0.62, BH * 0.07, BW * 0.82, BH * 0.07 + BW * 0.2], fill=255), (255, 240, 200), 201, 0.97)
        for i in range(24):
            x, y = R(0.04, 0.96) * BW, R(0.03, 0.34) * BH
            r = R(3, 7)
            wc_shape(cv, lambda d, x=x, y=y, r=r: d.ellipse([x - r, y - r, x + r, y + r], fill=255), (255, 246, 210), 210 + i, 0.85, wobble=2)
        x = -30
        k = 0
        while x < BW:
            bw, bh = R(110, 210), R(0.16, 0.34) * BH
            top = BH * 0.74 - bh
            col = rng.choice([(36, 34, 92), (46, 44, 110), (58, 44, 104)])
            wc_shape(cv, lambda d, x=x, bw=bw, top=top: d.rectangle([x, top, x + bw, BH * 0.74], fill=255), col, 240 + k, 0.97, wobble=4)
            for wy in range(int(top) + 30, int(BH * 0.74) - 30, 56):
                for wx in range(int(x) + 16, int(x + bw) - 24, 40):
                    if rng.random() < 0.4:
                        wc_shape(cv, lambda d, wx=wx, wy=wy: d.rectangle([wx, wy, wx + 18, wy + 24], fill=255), (255, 214, 110), 300 + wx + wy, 0.9, wobble=1.5, edge=0.1)
            x += bw + R(-10, 14)
            k += 1
        wc_shape(cv, lambda d: d.rectangle([0, BH * 0.74, BW, BH], fill=255), (74, 46, 70), 400, 0.97)
        for i in range(8):
            y = BH * 0.74 + 60 + i * 130
            wc_shape(cv, lambda d, y=y: d.rectangle([0, y, BW, y + 5], fill=255), (46, 28, 50), 410 + i, 0.5, wobble=4)
    else:  # sunset
        cv = grad((255, 190, 170), (150, 120, 210), 0, int(BH * 0.7))
        blotches(cv, 7, 0.1)
        wc_shape(cv, lambda d: d.ellipse([BW * 0.2, BH * 0.14, BW * 0.8, BH * 0.14 + BW * 0.6], fill=255), (255, 214, 130), 501, 0.97)
        wc_shape(cv, lambda d: d.ellipse([BW * 0.28, BH * 0.17, BW * 0.72, BH * 0.17 + BW * 0.44], fill=255), (255, 236, 170), 502, 0.7)
        wc_shape(cv, lambda d: d.polygon([(0, BH * 0.62), (BW * 0.35, BH * 0.5), (BW * 0.7, BH * 0.6), (BW, BH * 0.52), (BW, BH * 0.8), (0, BH * 0.8)], fill=255), (120, 90, 170), 510, 0.95, wobble=12)
        wc_shape(cv, lambda d: d.polygon([(0, BH * 0.7), (BW * 0.4, BH * 0.62), (BW * 0.85, BH * 0.7), (BW, BH * 0.66), (BW, BH * 0.85), (0, BH * 0.85)], fill=255), (70, 150, 150), 511, 0.95, wobble=12)
        wc_shape(cv, lambda d: d.rectangle([0, BH * 0.74, BW, BH], fill=255), (226, 176, 130), 520, 0.97)
        for i in range(10):
            x = (i + 0.5) / 10 * BW
            y = BH * 0.12 + 100 * math.sin(i / 9 * math.pi)
            wc_shape(cv, lambda d, x=x, y=y: d.ellipse([x - 16, y - 16, x + 16, y + 16], fill=255), (255, 232, 140), 530 + i, 0.95, wobble=2)
        for i in range(16):
            x, y = R(0.02, 0.98) * BW, R(0.76, 0.95) * BH
            wc_shape(cv, lambda d, x=x, y=y: d.polygon([(x, y), (x + 10, y - 50), (x + 20, y)], fill=255), (110, 150, 80), 560 + i, 0.8, wobble=2)
    return Image.fromarray(np.clip(cv, 0, 255).astype(np.uint8), "RGB")


_PAPER = None


def paper_tex():
    global _PAPER
    if _PAPER is None:
        n = fbm(H, W, 99, ((2, 0.5), (7, 0.3), (60, 0.2)))
        _PAPER = (1.0 - 0.045 * np.abs(n))[..., None].astype(np.float32) * np.array([1.0, 0.985, 0.955], np.float32)[None, None, :]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2)
        _PAPER *= (1 - 0.32 * np.clip(d - 0.55, 0, 1) ** 1.5)[..., None]
    return _PAPER


# ------------------------------------------------------------------ painted characters
_WARP = {}


def warp_fields(h, w, boil):
    k = (h, w, boil)
    if k not in _WARP:
        dx = fbm(h, w, 1000 + boil * 7, ((22, 0.7), (8, 0.3))) * 2.6
        dy = fbm(h, w, 2000 + boil * 7, ((22, 0.7), (8, 0.3))) * 2.6
        tex = gaussian_filter(np.random.default_rng(300 + boil).standard_normal((h, w)).astype(np.float32), (1.2, 5))
        tex /= tex.std() + 1e-6
        _WARP[k] = (dx, dy, tex)
    return _WARP[k]


def paint_sprite(spr, boil):
    a = np.asarray(spr, np.float32)
    h, w = a.shape[:2]
    dx, dy, tex = warp_fields(h, w, boil)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.stack([map_coordinates(a[..., c], [yy + dy, xx + dx], order=1, mode="constant") for c in range(4)], -1)
    al = out[..., 3] / 255.0
    rim = np.clip(al - gaussian_filter(al, 4), 0, 1)
    out[..., :3] *= (1 + 0.075 * tex[..., None]) * (1 - 0.30 * np.clip(rim * 3, 0, 1))[..., None]
    out[..., 3] *= 0.955 + 0.045 * np.clip(tex, -1, 1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


# ------------------------------------------------------------------ hand lettering + paper
def hand_text(d, xy, text, f, fill, boil=0, jit=1.6, anchor="l", stroke=0, stroke_fill=None):
    """Draw text char by char with a per-boil wobble so it looks hand-lettered."""
    total = sum(d.textlength(c, font=f) for c in text)
    x, y = xy
    if anchor == "m":
        x -= total / 2
    elif anchor == "r":
        x -= total
    rng = random.Random(boil * 131 + len(text))
    for c in text:
        ox, oy = rng.uniform(-jit, jit), rng.uniform(-jit, jit)
        d.text((x + ox, y + oy), c, font=f, fill=fill, anchor="ls", stroke_width=stroke, stroke_fill=stroke_fill)
        x += d.textlength(c, font=f)


_PCACHE = {}


def paper(w, h, seed, lined=False, color=(252, 243, 222)):
    k = (w, h, seed, lined, color)
    if k in _PCACHE:
        return _PCACHE[k]
    pad = 44
    sz = (w + 2 * pad, h + 2 * pad)
    rng = random.Random(seed)
    pts = []
    n = 14
    for side in range(4):
        for i in range(n):
            u = i / n
            if side == 0:
                p = (pad + u * w, pad)
            elif side == 1:
                p = (pad + w, pad + u * h)
            elif side == 2:
                p = (pad + w - u * w, pad + h)
            else:
                p = (pad, pad + h - u * h)
            pts.append((p[0] + rng.uniform(-4, 4), p[1] + rng.uniform(-4, 4)))
    m = Image.new("L", sz, 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    ma = np.asarray(m, np.float32) / 255.0
    rim = np.clip(ma - gaussian_filter(ma, 5), 0, 1)
    tx = fbm(sz[1], sz[0], seed, ((40, 0.6), (6, 0.4)))
    col = np.array(color, np.float32)[None, None, :] * (1 + 0.025 * tx[..., None]) * (1 - 0.22 * np.clip(rim * 3, 0, 1))[..., None]
    img = Image.new("RGBA", sz, (0, 0, 0, 0))
    sh = Image.new("RGBA", sz, (0, 0, 0, 0))
    sh.putalpha(m.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.42)))
    img.alpha_composite(sh, (10, 16))
    body = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    body.putalpha(m)
    img.alpha_composite(body)
    d = ImageDraw.Draw(img, "RGBA")
    if lined:
        for yy in range(pad + 150, pad + h - 20, 56):
            d.line([(pad + 30, yy), (pad + w - 30, yy)], fill=(120, 150, 220, 120), width=3)
        d.line([(pad + 86, pad + 20), (pad + 86, pad + h - 20)], fill=(230, 100, 110, 140), width=3)
    # tape
    tp = Image.new("RGBA", (190, 62), (255, 224, 140, 200))
    tp = tp.rotate(rng.uniform(-6, 6), expand=True)
    img.alpha_composite(tp, (sz[0] // 2 - tp.width // 2, pad - 30))
    _PCACHE[k] = img
    return img


def squiggle(d, x0, x1, y, col, boil, amp=5, width=6):
    rng = random.Random(boil * 17 + int(x0))
    pts = []
    n = max(3, int((x1 - x0) / 20))
    for i in range(n + 1):
        pts.append((x0 + (x1 - x0) * i / n, y + (amp if i % 2 else -amp) * rng.uniform(0.5, 1.0)))
    d.line(pts, fill=col, width=width, joint="curve")


def draw_icon(d, kind, cx, cy, r, boil):
    rng = random.Random(boil + 7)
    j = lambda: rng.uniform(-2, 2)  # noqa: E731
    if kind == "check":
        d.line([(cx - r * 0.7 + j(), cy + j()), (cx - r * 0.15 + j(), cy + r * 0.6 + j()), (cx + r * 0.8 + j(), cy - r * 0.7 + j())], fill=COL["green"] + (255,), width=10, joint="curve")
    elif kind == "x":
        d.line([(cx - r * 0.6 + j(), cy - r * 0.6 + j()), (cx + r * 0.6 + j(), cy + r * 0.6 + j())], fill=COL["red"] + (255,), width=10)
        d.line([(cx + r * 0.6 + j(), cy - r * 0.6 + j()), (cx - r * 0.6 + j(), cy + r * 0.6 + j())], fill=COL["red"] + (255,), width=10)
    elif kind == "arrow":
        d.line([(cx - r * 0.7, cy + j()), (cx + r * 0.6, cy + j())], fill=COL["blue"] + (255,), width=9)
        d.line([(cx + r * 0.1, cy - r * 0.55), (cx + r * 0.7, cy), (cx + r * 0.1, cy + r * 0.55)], fill=COL["blue"] + (255,), width=9, joint="curve")
    else:  # number in a circle
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=COL["pink"] + (255,), width=7)
        hand_text(d, (cx, cy + r * 0.45), str(kind), font(F_BRUSH, r * 1.5), COL["pink"] + (255,), boil, 1, "m")


def panel_image(panel, prog, boil, accent):
    """Return RGBA hand-painted note for a panel at progress prog (0-1)."""
    typ = panel["type"]
    w = 900
    if typ == "note":
        lines = panel["lines"]
        h = 150 + len(lines) * 130 + (90 if panel.get("sub") else 0)
        img = paper(w, h, 3 + len(lines), False).copy()
        d = ImageDraw.Draw(img, "RGBA")
        y = 44 + 150
        for ln in lines:
            txt, ck = (ln, "ink") if isinstance(ln, str) else ln
            hand_text(d, (44 + w / 2, y), txt, font(F_BRUSH, 112), COL[ck] + (255,), boil, 1.8, "m")
            if ck == "red":
                squiggle(d, 44 + w / 2 - 280, 44 + w / 2 + 280, y + 24, COL["red"] + (200,), boil)
            y += 130
        if panel.get("sub"):
            hand_text(d, (44 + w / 2, y - 10), panel["sub"], font(F_HAND, 44), (110, 90, 100, 255), boil, 1, "m")
        return img
    if typ == "list":
        rows = panel["rows"]
        h = 190 + len(rows) * 128
        img = paper(w, h, 11 + len(rows), False).copy()
        d = ImageDraw.Draw(img, "RGBA")
        hand_text(d, (44 + 56, 44 + 100), panel.get("title", ""), font(F_BRUSH, 90), COL["pink"] + (255,), boil, 1.6)
        n = len(rows) if prog >= 1 else int(len(rows) * ease_out(prog / 0.8)) + 1
        y = 44 + 200
        for i, r in enumerate(rows[:n]):
            icon, txt = r[0], r[1]
            draw_icon(d, icon, 44 + 96, y - 24, 40, boil)
            hand_text(d, (44 + 160, y), txt, font(F_HAND, 62), INK + (255,), boil, 1.2)
            if len(r) > 2:
                hand_text(d, (44 + 160, y + 46), r[2], font(F_HAND, 40), (120, 100, 110, 255), boil, 0.8)
            y += 128
        return img
    if typ == "prompt":
        text = panel["text"]
        shown = text[: int(len(text) * ease_out(prog / 0.85))] if prog < 1 else text
        f = font(F_HAND, 44)
        dd = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
        ls = []
        for para in shown.split("\n"):
            cur = ""
            for wd in para.split():
                t = (cur + " " + wd).strip()
                if dd.textlength(t, font=f) <= w - 190:
                    cur = t
                else:
                    ls.append(cur)
                    cur = wd
            ls.append(cur)
        full = text
        nfull = max(4, int(len(full) / 38) + 2)
        h = 190 + nfull * 56
        img = paper(w, h, 21, True).copy()
        d = ImageDraw.Draw(img, "RGBA")
        hand_text(d, (44 + 110, 44 + 100), panel.get("label", "Your prompt"), font(F_BRUSH, 70), COL["blue"] + (255,), boil, 1.4)
        for i, ln in enumerate(ls[:nfull]):
            hand_text(d, (44 + 110, 44 + 196 + i * 56), ln, f, INK + (255,), boil, 0.9)
        return img
    if typ == "chat":
        bubs = panel["bubbles"]
        h = 130 + len(bubs) * 190
        img = paper(w, h, 31 + len(bubs), False).copy()
        d = ImageDraw.Draw(img, "RGBA")
        n = len(bubs) if prog >= 1 else int(len(bubs) * ease_out(prog / 0.85)) + 1
        y = 44 + 56
        f = font(F_HAND, 44)
        for who, txt in bubs[:n]:
            lines = []
            cur = ""
            for wd in txt.split():
                t = (cur + " " + wd).strip()
                if d.textlength(t, font=f) <= 560:
                    cur = t
                else:
                    lines.append(cur)
                    cur = wd
            lines.append(cur)
            bh = len(lines) * 54 + 44
            if who == "you":
                x0, x1, fill, tc = 44 + w - 60 - 620, 44 + w - 60, (255, 176, 120, 255), INK
            else:
                x0, x1, fill, tc = 44 + 60, 44 + 60 + 620, (255, 255, 255, 255), INK
            d.rounded_rectangle([x0, y, x1, y + bh], radius=34, fill=fill, outline=INK + (255,), width=5)
            for j, ln in enumerate(lines):
                hand_text(d, (x0 + 30, y + 62 + j * 54 - 8), ln, f, tc + (255,), boil, 0.9)
            y += bh + 26
        return img
    raise ValueError(typ)


def sticker_image(top, word, boil):
    w, h = 820, 560
    img = Image.new("RGBA", (w + 60, h + 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    rng = random.Random(boil + 3)
    pts = []
    for i in range(48):
        a = i / 48 * 2 * math.pi
        r = 1 + 0.025 * math.sin(a * 5 + boil) + rng.uniform(-0.012, 0.012)
        pts.append((30 + w / 2 + math.cos(a) * w / 2 * r, 30 + h / 2 - 60 + math.sin(a) * (h - 120) / 2 * r))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).polygon([(x + 12, y + 18) for x, y in pts], fill=(0, 0, 0, 90))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)))
    d.polygon(pts, fill=(255, 226, 96, 255), outline=INK + (255,), width=7)
    hand_text(d, (30 + w / 2, 30 + 150), top, font(F_BRUSH, 110), INK + (255,), boil, 2, "m")
    hand_text(d, (30 + w / 2, 30 + 285), word, font(F_BRUSH, 190), COL["red"] + (255,), boil, 3, "m")
    squiggle(d, 30 + w / 2 - 300, 30 + w / 2 + 300, 30 + 318, COL["red"] + (220,), boil, 6, 8)
    cx = 30 + w / 2
    wob = 6 * math.sin(boil * 1.3)
    d.line([(cx, 30 + h - 130), (cx, 30 + h - 20 + wob)], fill=INK + (255,), width=12)
    d.polygon([(cx - 38, 30 + h - 52 + wob), (cx + 38, 30 + h - 52 + wob), (cx, 30 + h + 12 + wob)], fill=INK + (255,))
    return img


def banner_image(text, color, boil, size=118, width=980):
    f = font(F_BRUSH, size)
    dd = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if dd.textlength(t, font=f) <= width - 120:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    lines.append(cur)
    lh = int(size * 1.02)
    h = len(lines) * lh + 90
    img = Image.new("RGBA", (width + 60, h + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    rng = random.Random(boil * 5 + len(text))
    for i, ln in enumerate(lines):
        tw_ = dd.textlength(ln, font=f)
        y0 = 24 + i * lh
        x0, x1 = 30 + (width - tw_) / 2 - 36, 30 + (width + tw_) / 2 + 36
        pts = [(x0 + rng.uniform(-6, 6), y0 + rng.uniform(-3, 3)), (x1 + rng.uniform(-6, 6), y0 + rng.uniform(-3, 3)), (x1 + rng.uniform(-8, 8), y0 + lh + rng.uniform(-3, 5)), (x0 + rng.uniform(-8, 8), y0 + lh + rng.uniform(-3, 5))]
        d.polygon([(x + 8, y + 10) for x, y in pts], fill=(0, 0, 0, 70))
        d.polygon(pts, fill=color + (255,))
        hand_text(d, (30 + width / 2, y0 + lh * 0.80), ln, f, (255, 255, 255, 255), boil, 2.2, "m", stroke=7, stroke_fill=INK + (255,))
    return img


# ------------------------------------------------------------------ audio
def build_audio(script, model_dir):
    clips = []
    for b in script["beats"]:
        vname, spd = VOICES[b["who"]]
        parts = []
        for s in re.split(r"(?<=[.!?])\s+", b["say"].strip()):
            x, sr = voice.say(model_dir, s, vname, spd)
            assert sr == SR
            parts += [x / (np.max(np.abs(x)) or 1) * 0.95, np.zeros(int(SR * (0.34 if s.endswith("?") else 0.24)), np.float32)]
        clips.append(np.concatenate(parts[:-1]))
    t = 0.2
    tl = []
    for b, x in zip(script["beats"], clips):
        tl.append({"start": t, "end": t + len(x) / SR, "beat": b, "env": _envelope(x), "x": x})
        t += len(x) / SR + GAP
    total = t + script.get("end_hold", 2.6)
    n = int(total * SR) + SR // 10
    vox = np.zeros(n, np.float32)
    for seg in tl:
        i = int(seg["start"] * SR)
        vox[i:i + len(seg["x"])] += seg["x"]
    vox = vox / (np.max(np.abs(vox)) or 1) * 0.9
    bed = music_bed(total, script.get("seed", 5))[:n]
    bed = np.pad(bed, (0, n - len(bed)))
    duck = np.ones(n, np.float32) * 0.5
    for seg in tl:
        duck[int(seg["start"] * SR): int(seg["end"] * SR)] = 0.24
    duck = np.convolve(duck, np.ones(2400) / 2400, mode="same")
    out = vox + bed * duck
    pop, whoosh = sfx_pop(), sfx_whoosh()

    def add(sig, at, g=1.0):
        i = int(at * SR)
        if 0 <= i < n:
            out[i:i + len(sig)] += sig[: n - i] * g

    for seg in tl:
        b = seg["beat"]
        if b.get("panel"):
            add(pop, seg["start"] - 0.05)
        if b.get("rehook"):
            add(whoosh, seg["start"] - 0.35, 1.6)
            t0 = int(seg["start"] * SR)
            tt = np.arange(int(0.5 * SR)) / SR
            out[t0:t0 + len(tt)] += (np.sin(2 * np.pi * (220 + 400 * tt) * tt) * np.exp(-tt * 7) * 0.2).astype(np.float32)[: n - t0]
        if b.get("sticker"):
            add(whoosh, seg["start"] - 0.3, 1.4)
            add(pop, seg["start"] + 0.1, 1.5)
    add(pop, 0.0, 1.6)
    for k, f in enumerate((523, 659, 784)):
        tt = np.arange(int(0.9 * SR)) / SR
        i = int((total - 2.5 + k * 0.08) * SR)
        out[i:i + len(tt)] += (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 4.5) * 0.1).astype(np.float32)[: n - i]
    out = out / (np.max(np.abs(out)) or 1) * 0.92
    return tl, out, total


# ------------------------------------------------------------------ render
class Painter:
    def __init__(self, script, tl, total):
        self.s, self.tl, self.total = script, tl, total
        self.bgs = {}
        kinds = []
        for b in script["beats"]:
            if b.get("scene") and b["scene"] not in kinds:
                kinds.append(b["scene"])
        self.default_scene = script.get("scene", "study")
        for k in set(kinds + [self.default_scene]):
            self.bgs[k] = paint_bg(k)
        # scene timeline: list of (start_time, kind)
        self.scenes = [(0.0, self.default_scene)]
        for seg in tl:
            if seg["beat"].get("scene") and seg["beat"]["scene"] != self.scenes[-1][1]:
                self.scenes.append((seg["start"] - 0.1, seg["beat"]["scene"]))
        self.paper = paper_tex()
        self.cast = script["cast"]
        self.pos = {c: x for c, x in zip(self.cast, ({2: [290, 790], 3: [200, 540, 880]}[len(self.cast)]))}
        self.size = 440 if len(self.cast) == 2 else 380
        self.sprites = {}

    def bg_frame(self, t):
        def crop(kind, t0):
            z = 1.0 + 0.15 * smooth((t - t0) / 9.0)
            cw, ch = BW / z, BH / z
            ox = (BW - cw) / 2 + 28 * math.sin(t * 0.3)
            oy = (BH - ch) * 0.55
            return self.bgs[kind].crop((int(ox), int(oy), int(ox + cw), int(oy + ch))).resize((W, H), Image.BILINEAR)
        cur = 0
        for i, (st, k) in enumerate(self.scenes):
            if t >= st:
                cur = i
        st, k = self.scenes[cur]
        img = crop(k, st)
        if cur > 0 and t - st < 0.45:
            pst, pk = self.scenes[cur - 1]
            img = Image.blend(crop(pk, pst), img, (t - st) / 0.45)
        return img

    def seg_at(self, t):
        idx = 0
        for i, seg in enumerate(self.tl):
            if seg["start"] - GAP <= t <= seg["end"] + GAP:
                idx = i
        return idx

    def frame(self, t):
        boil = int(t * 8) % 4
        img = self.bg_frame(t).convert("RGBA")
        end_card = t >= self.total - self.s.get("end_hold", 2.6)
        idx = self.seg_at(t)
        seg = self.tl[idx]
        speaker = seg["beat"]["who"] if not end_card else None
        # characters
        for k, who in enumerate(self.cast):
            cx = self.pos[who]
            talking = who == speaker and seg["start"] <= t <= seg["end"]
            mouth = 0.0
            if talking:
                i = int((t - seg["start"]) * FPS)
                mouth = float(seg["env"][min(i, len(seg["env"]) - 1)]) ** 0.8
            blink = 1.0 if (t * 0.9 + k * 0.37) % 3.1 < 0.12 else 0.0
            mood = "happy"
            if not end_card:
                moods = seg["beat"].get("moods", {})
                mood = seg["beat"].get("mood", "happy") if who == speaker else moods.get(who, "happy")
            look = 0.6 if speaker and self.pos.get(speaker, cx) > cx else -0.6
            if speaker == who or not speaker:
                look = 0
            bob = -abs(math.sin(t * 9)) * 14 if talking else math.sin(t * 2 + k) * 4
            if end_card:
                bob = -abs(math.sin(t * 6 + k)) * 24
            wave = 1 if (end_card or (talking and seg["beat"].get("wave"))) else 0
            spr = C.render(who, {"t": t + k, "mouth": round(mouth * 6) / 6, "blink": blink, "mood": mood, "look": look, "bob": bob, "flip": cx > W / 2, "wave": wave}, size=self.size)
            spr = paint_sprite(spr, boil + k)
            if talking:
                spr = spr.resize((int(spr.width * 1.06), int(spr.height * 1.06)), Image.BILINEAR)
            sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(sh).ellipse([cx - 150, 1570, cx + 150, 1625], fill=(40, 20, 40, 90))
            img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(8)))
            img.alpha_composite(spr, (int(cx - spr.width / 2), int(1608 - spr.height)))
        # panel
        panel_beat = None
        for i, sg in enumerate(self.tl):
            if sg["beat"].get("panel") and sg["start"] - 0.05 <= t:
                panel_beat = i
        if panel_beat is not None and not end_card:
            sg = self.tl[panel_beat]
            if not sg["beat"].get("panel_keep", True) and idx != panel_beat:
                pass
            prog = clamp((t - sg["start"]) / max(sg["end"] - sg["start"], 0.01))
            appear = t - sg["start"] + 0.05
            pi = panel_image(sg["beat"]["panel"], prog, boil, None)
            sc = back_out(appear / 0.4) if appear < 0.4 else 1.0
            rot = (-2.0 + math.sin(t * 1.2) * 0.5) + (1 - clamp(appear / 0.4)) * 6
            pw, ph = int(pi.width * sc * 0.92), int(pi.height * sc * 0.92)
            if pw > 10 and ph > 10:
                p2 = pi.resize((pw, ph), Image.BILINEAR).rotate(rot, expand=True, resample=Image.BICUBIC)
                img.alpha_composite(p2, (int(W / 2 - p2.width / 2), int(560 - p2.height / 2 + (1 - clamp(appear / 0.4)) * 80)))
        d = ImageDraw.Draw(img, "RGBA")
        # name tag of speaker
        # hook banner (first seconds)
        hk = self.s.get("hook")
        hd = self.s.get("hook_dur", 3.4)
        if hk and t < hd + 0.5:
            q = ease_out(t / 0.3) if t < hd else 1 - ease_out((t - hd) / 0.4)
            bi = banner_image(hk, COL[self.s.get("hook_color", "pink")], boil)
            bi = bi.rotate(-2 + math.sin(t * 5) * 0.6, expand=True, resample=Image.BICUBIC)
            img.alpha_composite(bi, (int(W / 2 - bi.width / 2), int(150 - (1 - q) * 330)))
        elif self.s.get("tag"):
            tg = banner_image(self.s["tag"], COL[self.s.get("hook_color", "pink")], boil, size=64, width=900)
            img.alpha_composite(tg, (int(W / 2 - tg.width / 2), 70))
        # rehook banner
        for sg in self.tl:
            rb = sg["beat"].get("rehook")
            if rb and sg["start"] - 0.3 <= t <= sg["start"] + 1.9:
                u = t - (sg["start"] - 0.3)
                q = back_out(u / 0.35) if u < 0.35 else 1.0
                if u > 1.9:
                    q = 1 - (u - 1.9) / 0.3
                bi = banner_image(rb, COL["red"], boil, size=132, width=1000)
                s2 = max(0.05, q)
                bi = bi.resize((int(bi.width * s2), int(bi.height * s2)), Image.BILINEAR).rotate(3 + math.sin(t * 9) * 1.2, expand=True, resample=Image.BICUBIC)
                shake = (math.sin(t * 40) * 6 if u < 0.6 else 0, math.cos(t * 37) * 5 if u < 0.6 else 0)
                img.alpha_composite(bi, (int(W / 2 - bi.width / 2 + shake[0]), int(760 - bi.height / 2 + shake[1])))
        # sticker (comment bait)
        for sg in self.tl:
            st_ = sg["beat"].get("sticker")
            if st_ and t >= sg["start"] - 0.2:
                u = t - (sg["start"] - 0.2)
                q = back_out(u / 0.45) if u < 0.45 else 1.0
                si = sticker_image(st_["top"], st_["word"], boil)
                s2 = max(0.05, q) * 0.95
                si = si.resize((int(si.width * s2), int(si.height * s2)), Image.BILINEAR).rotate(-4 + math.sin(t * 2.4) * 1.5, expand=True, resample=Image.BICUBIC)
                img.alpha_composite(si, (int(W / 2 - si.width / 2), int(660 - si.height / 2)))
        # captions
        if not end_card and seg["start"] <= t <= seg["end"] + 0.15:
            words = seg["beat"]["say"].split()
            dur = seg["end"] - seg["start"]
            wts = np.array([len(w_) + 2 for w_ in words], float)
            cum = np.cumsum(wts) / wts.sum() * dur
            k = min(int(np.searchsorted(cum, t - seg["start"])), len(words) - 1)
            g0 = (k // 4) * 4
            chunk = words[g0:g0 + 4]
            size = 92
            while True:
                f = font(F_BRUSH, size)
                ws = [d.textlength(w_, font=f) + 24 for w_ in chunk]
                if sum(ws) <= W - 90 or size <= 52:
                    break
                size -= 4
            x = (W - sum(ws)) / 2
            for i, w_ in enumerate(chunk):
                cur = g0 + i == k
                hand_text(d, (x, 1770), w_, f, ((255, 226, 96) if cur else (255, 255, 255)) + (255,), boil, 1.4, "l", stroke=10, stroke_fill=(52, 28, 48, 255))
                x += ws[i]
        if end_card:
            u = t - (self.total - self.s.get("end_hold", 2.6))
            q = ease_out(u / 0.4)
            hand_text(d, (W / 2, 1780), "Follow @shipwithpip", font(F_BRUSH, 96), (255, 255, 255, int(255 * q)), boil, 1.6, "m", stroke=10, stroke_fill=(52, 28, 48, int(255 * q)))
        # paper + progress
        arr = np.asarray(img.convert("RGB"), np.float32) * self.paper
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        ImageDraw.Draw(out).rectangle([0, 0, W * clamp(t / self.total), 10], fill=(255, 226, 96))
        return out


def write_post(s, path, total):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{s['title']}\nLength: {total:.1f}s\n\nCAPTION\n{s['caption']}\n\n")
        fh.write(" ".join("#" + h.lstrip("#") for h in s["hashtags"]) + "\n\n")
        if s.get("pinned_comment"):
            fh.write(f"PINNED COMMENT\n{s['pinned_comment']}\n\n")
        if s.get("sources"):
            fh.write("SOURCES\n" + "\n".join(s["sources"]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    s = json.load(open(a.script, encoding="utf-8"))
    tl, audio, total = build_audio(s, a.model)
    print("total", round(total, 1), "beat ends", [round(x["end"], 1) for x in tl])
    P = Painter(s, tl, total)
    if a.stills:
        ts = s.get("still_times") or [round(total * i / 9, 1) for i in range(1, 9)]
        fr = [P.frame(t).resize((300, 533)) for t in ts]
        sheet = Image.new("RGB", (300 * len(fr), 533))
        for i, f_ in enumerate(fr):
            sheet.paste(f_, (i * 300, 0))
        sheet.save(a.stills)
        return
    wav = os.path.splitext(a.out)[0] + "_a.wav"
    sf.write(wav, audio, SR)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-15:TP=-1.2:LRA=11",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for i in range(int(total * FPS)):
        ff.stdin.write(P.frame(i / FPS).tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    write_post(s, os.path.splitext(a.out)[0] + "_post.txt", total)
    print(a.out, round(total, 1))


if __name__ == "__main__":
    main()
