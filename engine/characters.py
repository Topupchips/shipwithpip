"""The reel cast: three original 2D mascots drawn in code so they look identical every day.

Each character is drawn at 2x and scaled down for smooth edges. Every draw call takes
a pose dict: mouth (0-1 open), blink (0-1 closed), look (-1..1 eye direction),
mood ('happy' | 'shocked' | 'annoyed' | 'smug'), bob (vertical bounce in px),
flip (face left), t (seconds, for idle wiggle).
"""
import math
from PIL import Image, ImageDraw

SS = 2  # supersample factor
INK = (24, 22, 38)
WHITE = (255, 255, 255)

CAST = {
    "pip": {
        "name": "Pip",
        "role": "The helper. Knows every Claude trick and can't wait to show you.",
        "voice": "af_bella", "speed": 1.08,
        "body": (255, 122, 89), "shade": (226, 92, 64), "accent": (255, 214, 102),
    },
    "dex": {
        "name": "Dex",
        "role": "Student founder. Always behind, always caffeinated, asks the questions viewers have.",
        "voice": "am_michael", "speed": 1.12,
        "body": (64, 196, 180), "shade": (44, 160, 146), "accent": (120, 92, 70),
    },
    "flop": {
        "name": "Flop",
        "role": "Grumpy retro floppy disk. Does everything the hard way and is slowly converted.",
        "voice": "bm_george", "speed": 1.05,
        "body": (92, 104, 214), "shade": (70, 80, 180), "accent": (200, 206, 222),
    },
}


def _s(v):
    return int(round(v * SS))


def _eye(d, cx, cy, r, look, blink, mood, flip):
    """Big cartoon eye with pupil, highlight and mood-shaped lid."""
    if blink > 0.85:
        d.arc([_s(cx - r), _s(cy - r * 0.4), _s(cx + r), _s(cy + r * 0.6)], 200, 340, fill=INK, width=_s(6))
        return
    h = r * (1 - 0.8 * blink)
    d.ellipse([_s(cx - r), _s(cy - h), _s(cx + r), _s(cy + h)], fill=WHITE, outline=INK, width=_s(5))
    pr = r * (0.42 if mood != "shocked" else 0.28)
    px = cx + look * r * 0.38 * (-1 if flip else 1)
    py = cy + r * 0.08
    d.ellipse([_s(px - pr), _s(py - pr), _s(px + pr), _s(py + pr)], fill=INK)
    d.ellipse([_s(px - pr * 0.25 - pr * 0.35), _s(py - pr * 0.7), _s(px + pr * 0.15 - pr * 0.35), _s(py - pr * 0.3)], fill=WHITE)


def _brow(d, cx, cy, w, mood, side):
    tilt = {"happy": 5, "shocked": 14, "annoyed": -14, "smug": -6}.get(mood, 0) * side
    lift = -10 if mood == "shocked" else 0
    d.line([_s(cx - w), _s(cy + tilt + lift), _s(cx + w), _s(cy - tilt + lift)], fill=INK, width=_s(9))


def _mouth(d, cx, cy, w, mouth, mood):
    if mouth > 0.08:
        h = 8 + mouth * 46
        d.rounded_rectangle([_s(cx - w * 0.6), _s(cy - h * 0.35), _s(cx + w * 0.6), _s(cy + h * 0.65)],
                            radius=_s(min(h, w) * 0.45), fill=(120, 30, 50), outline=INK, width=_s(5))
        d.ellipse([_s(cx - w * 0.3), _s(cy + h * 0.2), _s(cx + w * 0.3), _s(cy + h * 0.62)], fill=(240, 110, 120))
    elif mood == "shocked":
        d.ellipse([_s(cx - 12), _s(cy - 10), _s(cx + 12), _s(cy + 14)], fill=(120, 30, 50), outline=INK, width=_s(5))
    elif mood == "annoyed":
        d.line([_s(cx - w * 0.5), _s(cy + 6), _s(cx + w * 0.5), _s(cy - 2)], fill=INK, width=_s(7))
    elif mood == "smug":
        d.arc([_s(cx - w * 0.6), _s(cy - 20), _s(cx + w * 0.4), _s(cy + 16)], 20, 140, fill=INK, width=_s(7))
    else:
        d.arc([_s(cx - w * 0.6), _s(cy - 26), _s(cx + w * 0.6), _s(cy + 18)], 25, 155, fill=INK, width=_s(7))


def _cheeks(d, cx, cy, sep, col):
    for sx in (-1, 1):
        d.ellipse([_s(cx + sx * sep - 22), _s(cy - 10), _s(cx + sx * sep + 22), _s(cy + 12)], fill=col)


def _shadow(d, cx, by, w):
    d.ellipse([_s(cx - w), _s(by - 14), _s(cx + w), _s(by + 14)], fill=(0, 0, 0, 60))


# ---------------------------------------------------------------- Pip
def _draw_pip(d, p):
    c = CAST["pip"]
    t, bob = p["t"], p["bob"]
    cx, top = 300, 170 + bob
    _shadow(d, cx, 690, 170)
    # stubby feet
    for sx in (-1, 1):
        d.ellipse([_s(cx + sx * 70 - 46), _s(640 + bob * 0.3), _s(cx + sx * 70 + 46), _s(690 + bob * 0.3)], fill=c["shade"], outline=INK, width=_s(6))
    # antenna with star bulb, wobbling
    wob = math.sin(t * 5) * 14
    ax, ay = cx + wob, top - 90
    d.line([_s(cx), _s(top + 10), _s(ax), _s(ay)], fill=INK, width=_s(8))
    star = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5 + t * 1.5
        r = 34 if i % 2 == 0 else 15
        star.append((_s(ax + r * math.cos(a)), _s(ay + r * math.sin(a))))
    d.polygon(star, fill=c["accent"], outline=INK, width=_s(5))
    # bean body
    squash = 1 + 0.03 * math.sin(t * 4)
    d.rounded_rectangle([_s(cx - 175 * squash), _s(top), _s(cx + 175 * squash), _s(top + 490 / squash)],
                        radius=_s(170), fill=c["body"], outline=INK, width=_s(8))
    d.ellipse([_s(cx - 110), _s(top + 300), _s(cx + 110), _s(top + 470)], fill=(255, 168, 140))
    # arms
    wave = p.get("wave", 0)
    for sx in (-1, 1):
        lift = wave * 160 if (sx == 1) != p["flip"] else 0
        x0, y0 = cx + sx * 168, top + 300
        x1, y1 = cx + sx * 230, top + 380 - lift + math.sin(t * 6) * 10 * (wave > 0)
        d.line([_s(x0), _s(y0), _s(x1), _s(y1)], fill=INK, width=_s(22))
        d.line([_s(x0), _s(y0), _s(x1), _s(y1)], fill=c["body"], width=_s(10))
        d.ellipse([_s(x1 - 24), _s(y1 - 24), _s(x1 + 24), _s(y1 + 24)], fill=c["body"], outline=INK, width=_s(6))
    _brow(d, cx - 70, top + 115, 30, p["mood"], 1)
    _brow(d, cx + 70, top + 115, 30, p["mood"], -1)
    _eye(d, cx - 70, top + 180, 46, p["look"], p["blink"], p["mood"], p["flip"])
    _eye(d, cx + 70, top + 180, 46, p["look"], p["blink"], p["mood"], p["flip"])
    _cheeks(d, cx, top + 250, 125, (255, 150, 140))
    _mouth(d, cx, top + 265, 60, p["mouth"], p["mood"])


# ---------------------------------------------------------------- Dex
def _draw_dex(d, p):
    c = CAST["dex"]
    t, bob = p["t"], p["bob"]
    cx, top = 300, 150 + bob
    _shadow(d, cx, 690, 160)
    # legs
    for sx in (-1, 1):
        d.rounded_rectangle([_s(cx + sx * 60 - 26), _s(560), _s(cx + sx * 60 + 26), _s(680)], radius=_s(20), fill=(44, 52, 84), outline=INK, width=_s(6))
        d.ellipse([_s(cx + sx * 60 - 44), _s(650), _s(cx + sx * 60 + 44), _s(694)], fill=WHITE, outline=INK, width=_s(6))
    # messy hair tufts
    for i, (hx, hh) in enumerate([(-90, 60), (-40, 90), (10, 75), (60, 95), (100, 55)]):
        sway = math.sin(t * 3 + i) * 6
        d.polygon([(_s(cx + hx - 34), _s(top + 40)), (_s(cx + hx + sway), _s(top - hh)), (_s(cx + hx + 34), _s(top + 40))], fill=(60, 44, 34), outline=INK, width=_s(5))
    # body: rounded square head-body
    d.rounded_rectangle([_s(cx - 160), _s(top), _s(cx + 160), _s(top + 430)], radius=_s(90), fill=c["body"], outline=INK, width=_s(8))
    # hoodie strings
    for sx in (-1, 1):
        d.line([_s(cx + sx * 40), _s(top + 330), _s(cx + sx * 46), _s(top + 400)], fill=WHITE, width=_s(6))
    # coffee mug in the non-facing hand
    side = -1 if not p["flip"] else 1
    mx, my = cx + side * 190, top + 330 + math.sin(t * 2) * 6
    d.line([_s(cx + side * 150), _s(top + 280), _s(mx), _s(my)], fill=INK, width=_s(22))
    d.line([_s(cx + side * 150), _s(top + 280), _s(mx), _s(my)], fill=c["body"], width=_s(10))
    d.rounded_rectangle([_s(mx - 40), _s(my - 50), _s(mx + 40), _s(my + 45)], radius=_s(12), fill=WHITE, outline=INK, width=_s(6))
    d.arc([_s(mx + 28), _s(my - 30), _s(mx + 68), _s(my + 20)], 270, 90, fill=INK, width=_s(7))
    d.rectangle([_s(mx - 30), _s(my - 15), _s(mx + 30), _s(my + 5)], fill=c["accent"])
    for k in range(2):  # steam
        sx0 = mx - 14 + k * 26
        pts = [(_s(sx0 + math.sin(t * 4 + j + k) * 7), _s(my - 60 - j * 14)) for j in range(5)]
        d.line(pts, fill=(255, 255, 255, 170), width=_s(5))
    # other arm
    ox = cx - side * 190
    d.line([_s(cx - side * 150), _s(top + 280), _s(ox), _s(top + 380)], fill=INK, width=_s(22))
    d.line([_s(cx - side * 150), _s(top + 280), _s(ox), _s(top + 380)], fill=c["body"], width=_s(10))
    d.ellipse([_s(ox - 24), _s(top + 356), _s(ox + 24), _s(top + 404)], fill=c["body"], outline=INK, width=_s(6))
    # glasses + eyes
    _eye(d, cx - 62, top + 150, 40, p["look"], p["blink"], p["mood"], p["flip"])
    _eye(d, cx + 62, top + 150, 40, p["look"], p["blink"], p["mood"], p["flip"])
    for sx in (-1, 1):
        d.rounded_rectangle([_s(cx + sx * 62 - 58), _s(top + 100), _s(cx + sx * 62 + 58), _s(top + 200)], radius=_s(26), outline=INK, width=_s(7))
    d.line([_s(cx - 4), _s(top + 140), _s(cx + 4), _s(top + 140)], fill=INK, width=_s(7))
    # tired eye bags
    for sx in (-1, 1):
        d.arc([_s(cx + sx * 62 - 34), _s(top + 196), _s(cx + sx * 62 + 34), _s(top + 222)], 200, 340, fill=c["shade"], width=_s(5))
    _brow(d, cx - 62, top + 82, 30, p["mood"], 1)
    _brow(d, cx + 62, top + 82, 30, p["mood"], -1)
    _mouth(d, cx, top + 265, 54, p["mouth"], p["mood"])


# ---------------------------------------------------------------- Flop
def _draw_flop(d, p):
    c = CAST["flop"]
    t, bob = p["t"], p["bob"]
    cx, top = 300, 180 + bob
    _shadow(d, cx, 690, 170)
    for sx in (-1, 1):
        d.line([_s(cx + sx * 70), _s(top + 420), _s(cx + sx * 76), _s(670)], fill=INK, width=_s(14))
        d.rounded_rectangle([_s(cx + sx * 76 - 46), _s(650), _s(cx + sx * 76 + 46), _s(690)], radius=_s(16), fill=(40, 40, 50))
    # square disk body with clipped corner
    L, R, T, B = cx - 210, cx + 210, top, top + 440
    d.polygon([(_s(L), _s(T)), (_s(R - 50), _s(T)), (_s(R), _s(T + 50)), (_s(R), _s(B)), (_s(L), _s(B))], fill=c["body"], outline=INK, width=_s(8))
    # metal shutter on top
    d.rectangle([_s(cx - 110), _s(T), _s(cx + 90), _s(T + 120)], fill=c["accent"], outline=INK, width=_s(6))
    d.rectangle([_s(cx + 20), _s(T + 18), _s(cx + 62), _s(T + 102)], fill=c["shade"])
    # label area = face
    d.rounded_rectangle([_s(L + 50), _s(T + 170), _s(R - 50), _s(B - 20)], radius=_s(14), fill=(246, 244, 236), outline=INK, width=_s(6))
    for k in range(3):
        d.line([_s(L + 80), _s(B - 70 + k * 16), _s(R - 80), _s(B - 70 + k * 16)], fill=(200, 196, 230), width=_s(3))
    # bushy grumpy unibrow + mustache
    _eye(d, cx - 68, top + 250, 36, p["look"], max(p["blink"], 0.25 if p["mood"] == "annoyed" else 0), p["mood"], p["flip"])
    _eye(d, cx + 68, top + 250, 36, p["look"], max(p["blink"], 0.25 if p["mood"] == "annoyed" else 0), p["mood"], p["flip"])
    _brow(d, cx - 68, top + 200, 38, p["mood"] if p["mood"] != "happy" else "smug", 1)
    _brow(d, cx + 68, top + 200, 38, p["mood"] if p["mood"] != "happy" else "smug", -1)
    _mouth(d, cx, top + 340, 56, p["mouth"], p["mood"])
    stache_y = top + 318
    for sx in (-1, 1):
        d.chord([_s(cx + (sx - 1) * 60 + (0 if sx < 0 else 0)), _s(stache_y - 26), _s(cx + (sx + 1) * 60), _s(stache_y + 26)], 180, 360, fill=(230, 230, 236), outline=INK, width=_s(5))
    # arms: crossed when annoyed
    if p["mood"] == "annoyed":
        d.line([_s(L + 10), _s(top + 330), _s(R - 40), _s(top + 380)], fill=INK, width=_s(20))
        d.line([_s(R - 10), _s(top + 330), _s(L + 40), _s(top + 380)], fill=INK, width=_s(20))
    else:
        for sx in (-1, 1):
            x0 = L if sx < 0 else R
            d.line([_s(x0), _s(top + 300), _s(x0 + sx * 60), _s(top + 390 + math.sin(t * 3 + sx) * 8)], fill=INK, width=_s(16))
            d.ellipse([_s(x0 + sx * 60 - 22), _s(top + 368), _s(x0 + sx * 60 + 22), _s(top + 412)], fill=WHITE, outline=INK, width=_s(5))


DRAW = {"pip": _draw_pip, "dex": _draw_dex, "flop": _draw_flop}


def render(who, pose=None, size=600):
    """Return an RGBA sprite of a character, about size x size*1.2."""
    p = {"t": 0.0, "mouth": 0.0, "blink": 0.0, "look": 0.0, "mood": "happy", "bob": 0.0, "flip": False, "wave": 0}
    p.update(pose or {})
    img = Image.new("RGBA", (_s(600), _s(720)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    DRAW[who](d, p)
    if p["flip"]:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img.resize((size, int(size * 1.2)), Image.LANCZOS)
