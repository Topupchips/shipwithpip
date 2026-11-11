"""Modern flat-design mascots: clean gradient shapes, soft shadows, no hand-drawn
outlines. Same pose-dict interface as characters.py so they drop into the same
timeline/animation logic (mouth 0-1, blink 0-1, look -1..1, mood, bob, flip, t).
"""
import math
from PIL import Image, ImageDraw, ImageFilter

SS = 2
INK = (36, 32, 58)

CAST = {
    "pip": {
        "name": "Pip",
        "top": (255, 140, 94), "bot": (255, 94, 122), "ring": (255, 214, 140),
        "face": (255, 255, 255),
    },
    "flop": {
        "name": "Flop",
        "top": (108, 124, 244), "bot": (74, 84, 196), "ring": (190, 220, 255),
        "face": (244, 246, 255),
    },
}


def _s(v):
    return int(round(v * SS))


def vgrad(w, h, top, bot):
    """Vertical two-tone gradient as an RGBA image (opaque)."""
    ys = [i / max(1, h - 1) for i in range(h)]
    row = [tuple(int(top[c] + (bot[c] - top[c]) * y) for c in range(3)) + (255,) for y in ys]
    img = Image.new("RGBA", (w, h))
    px = img.load()
    for y, col in enumerate(row):
        for x in range(w):
            px[x, y] = col
    return img


def grad_shape(mask_img, top, bot):
    """mask_img: L-mode or alpha mask same size. Returns RGBA filled with vertical gradient, clipped to mask."""
    g = vgrad(mask_img.width, mask_img.height, top, bot)
    g.putalpha(mask_img if mask_img.mode == "L" else mask_img.split()[-1])
    return g


def _eye(d, cx, cy, r, look, blink, mood, flip, pupil=INK):
    if blink > 0.85:
        d.line([_s(cx - r), _s(cy), _s(cx + r), _s(cy)], fill=pupil, width=_s(6))
        return
    h = r * (1 - 0.82 * blink)
    d.ellipse([_s(cx - r), _s(cy - h), _s(cx + r), _s(cy + h)], fill=(255, 255, 255, 255))
    pr = r * (0.44 if mood != "shocked" else 0.3)
    px = cx + look * r * 0.36 * (-1 if flip else 1)
    py = cy + r * 0.05
    d.ellipse([_s(px - pr), _s(py - pr), _s(px + pr), _s(py + pr)], fill=pupil)
    hl = pr * 0.32
    d.ellipse([_s(px - pr * 0.3 - hl), _s(py - pr * 0.75 - hl / 2), _s(px - pr * 0.3 + hl), _s(py - pr * 0.75 + hl / 2)], fill=(255, 255, 255, 230))


def _brow(d, cx, cy, w, mood, side, col=INK):
    tilt = {"happy": 4, "shocked": 12, "annoyed": -13, "smug": -5}.get(mood, 0) * side
    lift = -8 if mood == "shocked" else 0
    d.line([_s(cx - w), _s(cy + tilt + lift), _s(cx + w), _s(cy - tilt + lift)], fill=col, width=_s(7))


def _mouth(d, cx, cy, w, mouth, mood, col=INK):
    if mouth > 0.08:
        h = 6 + mouth * 40
        d.rounded_rectangle([_s(cx - w * 0.58), _s(cy - h * 0.3), _s(cx + w * 0.58), _s(cy + h * 0.68)],
                            radius=_s(min(h, w) * 0.5), fill=(150, 56, 72, 255))
        d.ellipse([_s(cx - w * 0.26), _s(cy + h * 0.18), _s(cx + w * 0.26), _s(cy + h * 0.6)], fill=(255, 140, 150, 255))
    elif mood == "shocked":
        d.ellipse([_s(cx - 11), _s(cy - 9), _s(cx + 11), _s(cy + 13)], fill=(150, 56, 72, 255))
    elif mood == "annoyed":
        d.line([_s(cx - w * 0.46), _s(cy + 5), _s(cx + w * 0.46), _s(cy - 2)], fill=col, width=_s(6))
    elif mood == "smug":
        d.arc([_s(cx - w * 0.55), _s(cy - 18), _s(cx + w * 0.38), _s(cy + 14)], 20, 140, fill=col, width=_s(6))
    else:
        d.arc([_s(cx - w * 0.55), _s(cy - 22), _s(cx + w * 0.55), _s(cy + 15)], 25, 155, fill=col, width=_s(6))


def _shadow(img, cx, by, w, h=26, a=70):
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([_s(cx - w), _s(by - h), _s(cx + w), _s(by + h)], fill=(20, 16, 40, a))
    sh = sh.filter(ImageFilter.GaussianBlur(_s(10)))
    img.alpha_composite(sh)


def _squircle_mask(size, w, h, rad):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w, h], radius=rad, fill=255)
    return m


# ---------------------------------------------------------------- Pip: floating gradient orb
def _draw_pip(img, p):
    c = CAST["pip"]
    t, bob = p["t"], p["bob"]
    cx, cy = 300, 420 + bob
    _shadow(img, cx, 660, 150, 22, 60)
    d = ImageDraw.Draw(img, "RGBA")
    # signal ring above head
    wob = math.sin(t * 3.4) * 6
    ry = cy - 280 + wob
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gr = 30 + 4 * math.sin(t * 5)
    gd.ellipse([_s(cx - gr), _s(ry - gr), _s(cx + gr), _s(ry + gr)], fill=c["ring"] + (160,))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(_s(14))))
    d.ellipse([_s(cx - 16), _s(ry - 16), _s(cx + 16), _s(ry + 16)], fill=c["ring"] + (255,))
    d.line([_s(cx), _s(ry + 16), _s(cx), _s(cy - 196)], fill=(220, 220, 236, 160), width=_s(6))
    # arms: simple flat capsules behind body
    wave = p.get("wave", 0)
    for sx in (-1, 1):
        lift = wave * 150 if (sx == 1) != p["flip"] else 0
        x0, y0 = cx + sx * 150, cy + 60
        x1, y1 = cx + sx * 215, cy + 140 - lift + math.sin(t * 6) * 8 * (wave > 0)
        d.line([_s(x0), _s(y0), _s(x1), _s(y1)], fill=c["bot"] + (255,), width=_s(26))
        d.ellipse([_s(x1 - 22), _s(y1 - 22), _s(x1 + 22), _s(y1 + 22)], fill=c["top"] + (255,))
    # body: tall squircle, gradient fill
    squash = 1 + 0.025 * math.sin(t * 4)
    bw, bh = 185 * squash * 2, 440 / squash
    bx0, by0 = cx - bw / 2, cy - bh / 2
    mask = _squircle_mask((_s(bw), _s(bh)), _s(bw), _s(bh), _s(min(bw, bh) * 0.42))
    body = grad_shape(mask, c["top"], c["bot"])
    img.alpha_composite(body, (_s(bx0), _s(by0)))
    d = ImageDraw.Draw(img, "RGBA")
    # face plate
    fw, fh = bw * 1.02, bh * 0.56
    fx0, fy0 = cx - fw * 0.42, cy - bh * 0.42
    _brow(d, cx - 62, fy0 + 70, 28, p["mood"], 1)
    _brow(d, cx + 62, fy0 + 70, 28, p["mood"], -1)
    _eye(d, cx - 62, fy0 + 128, 42, p["look"], p["blink"], p["mood"], p["flip"])
    _eye(d, cx + 62, fy0 + 128, 42, p["look"], p["blink"], p["mood"], p["flip"])
    _mouth(d, cx, fy0 + 210, 56, p["mouth"], p["mood"])
    if p["mood"] == "happy" or p["mood"] == "smug":
        for sx in (-1, 1):
            d.ellipse([_s(cx + sx * 108 - 20), _s(fy0 + 175), _s(cx + sx * 108 + 20), _s(fy0 + 205)], fill=(255, 255, 255, 60))


# ---------------------------------------------------------------- Flop: flat smart-tablet mascot
def _draw_flop(img, p):
    c = CAST["flop"]
    t, bob = p["t"], p["bob"]
    cx, cy = 300, 430 + bob
    _shadow(img, cx, 668, 160, 22, 60)
    d = ImageDraw.Draw(img, "RGBA")
    # feet
    for sx in (-1, 1):
        d.rounded_rectangle([_s(cx + sx * 78 - 34), _s(cy + 220), _s(cx + sx * 78 + 34), _s(cy + 256)], radius=_s(14), fill=c["bot"] + (255,))
    # body: rounded rect gradient, slight tablet tilt via skew-free flat rect
    bw, bh = 220, 400
    mask = _squircle_mask((_s(bw), _s(bh)), _s(bw), _s(bh), _s(54))
    body = grad_shape(mask, c["top"], c["bot"])
    img.alpha_composite(body, (_s(cx - bw / 2), _s(cy - bh / 2)))
    d = ImageDraw.Draw(img, "RGBA")
    # antenna dot (status light)
    lx, ly = cx, cy - bh / 2 - 24
    pulse = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(t * 5))
    d.line([_s(lx), _s(ly + 10), _s(lx), _s(cy - bh / 2)], fill=c["ring"] + (200,), width=_s(5))
    d.ellipse([_s(lx - 9), _s(ly - 9), _s(lx + 9), _s(ly + 9)], fill=tuple(int(c["ring"][i]) for i in range(3)) + (int(200 + 55 * pulse),))
    # screen / face plate
    fw, fh = bw * 0.78, bh * 0.56
    fx0, fy0 = cx - fw / 2, cy - bh * 0.30
    d.rounded_rectangle([_s(fx0), _s(fy0), _s(fx0 + fw), _s(fy0 + fh)], radius=_s(28), fill=c["face"] + (255,))
    ex = fw * 0.26
    _eye(d, cx - ex, fy0 + fh * 0.42, 34, p["look"], max(p["blink"], 0.2 if p["mood"] == "annoyed" else 0), p["mood"], p["flip"], pupil=c["bot"])
    _eye(d, cx + ex, fy0 + fh * 0.42, 34, p["look"], max(p["blink"], 0.2 if p["mood"] == "annoyed" else 0), p["mood"], p["flip"], pupil=c["bot"])
    _brow(d, cx - ex, fy0 + fh * 0.18, 26, p["mood"] if p["mood"] != "happy" else "smug", 1, col=c["bot"])
    _brow(d, cx + ex, fy0 + fh * 0.18, 26, p["mood"] if p["mood"] != "happy" else "smug", -1, col=c["bot"])
    _mouth(d, cx, fy0 + fh * 0.8, 48, p["mouth"], p["mood"], col=c["bot"])
    # arms
    if p["mood"] == "annoyed":
        d.line([_s(cx - bw / 2 + 6), _s(cy + 40), _s(cx + bw / 2 - 30), _s(cy + 92)], fill=c["bot"] + (255,), width=_s(20))
        d.line([_s(cx + bw / 2 - 6), _s(cy + 40), _s(cx - bw / 2 + 30), _s(cy + 92)], fill=c["bot"] + (255,), width=_s(20))
    else:
        for sx in (-1, 1):
            x0 = cx + sx * bw / 2
            x1, y1 = x0 + sx * 56, cy + 70 + math.sin(t * 3 + sx) * 8
            d.line([_s(x0), _s(cy + 20), _s(x1), _s(y1)], fill=c["bot"] + (255,), width=_s(18))
            d.ellipse([_s(x1 - 20), _s(y1 - 20), _s(x1 + 20), _s(y1 + 20)], fill=c["top"] + (255,))


DRAW = {"pip": _draw_pip, "flop": _draw_flop}


def render(who, pose=None, size=600):
    p = {"t": 0.0, "mouth": 0.0, "blink": 0.0, "look": 0.0, "mood": "happy", "bob": 0.0, "flip": False, "wave": 0}
    p.update(pose or {})
    img = Image.new("RGBA", (_s(600), _s(760)), (0, 0, 0, 0))
    DRAW[who](img, p)
    if p["flip"]:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img.resize((size, int(size * 760 / 600)), Image.LANCZOS)
