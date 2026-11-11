"""Dark macro-explainer reels (magnified illustrations of everyday tech), word-by-word top captions.
usage: python macrofx.py micro|qi|gps|pixel out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa: E402
from scenes import chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, F_SEMI, W, H, clamp, ease_out, font, smooth, tone, tw  # noqa: E402

GOLD, AMBER, BLUE, ICE = (255, 200, 80), (255, 140, 50), (90, 150, 255), (170, 210, 255)
RED, GRN, WHT, DIM = (255, 70, 70), (70, 255, 130), (245, 245, 250), (120, 124, 138)
CX = W // 2


def lerp(a, b, u):
    return a + (b - a) * u


def col_a(c, a):
    return tuple(int(v) for v in c) + (int(255 * clamp(a)),)


def mix(c1, c2, u):
    return tuple(int(lerp(a, b, clamp(u))) for a, b in zip(c1, c2))


def heat_color(u):
    u = clamp(u)
    stops = [(0.0, (20, 14, 40)), (0.35, (150, 40, 20)), (0.7, (255, 150, 40)), (1.0, (255, 250, 200))]
    for (u0, c0), (u1, c1) in zip(stops, stops[1:]):
        if u <= u1:
            return mix(c0, c1, (u - u0) / (u1 - u0))
    return stops[-1][1]


def label(d, x, y, txt, size=22, col=(150, 154, 168), anchor="la"):
    d.text((x, y), txt, font=font(F_SEMI, size), fill=col + (255,), anchor=anchor)


def arrow(d, p0, p1, col, w=6, head=22):
    d.line([p0, p1], fill=col, width=w)
    a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    pts = [p1, (p1[0] - head * math.cos(a - 0.4), p1[1] - head * math.sin(a - 0.4)), (p1[0] - head * math.cos(a + 0.4), p1[1] - head * math.sin(a + 0.4))]
    d.polygon(pts, fill=col)


class Macro:
    """scenes: list of (t0, t1, fn(layer_draw, t_local, T_scene)); captions from vo"""

    def __init__(self, cfg, vo):
        self.cfg, self.vo, self.T = cfg, vo, cfg["T"]
        yy = np.linspace(0, 1, H)[:, None, None]
        img = np.broadcast_to(np.array([13, 14, 20]) * (1 - yy) + np.array([4, 4, 7]) * yy, (H, W, 3)).astype(float)
        Y, X = np.mgrid[0:H, 0:W]
        v = 1 - 0.55 * (((X - W / 2) / (W * 0.75)) ** 2 + ((Y - H / 2) / (H * 0.75)) ** 2)
        img = img * np.clip(v, 0.3, 1)[..., None]
        rs = np.random.default_rng(2)
        img += rs.normal(0, 1.6, (H, W, 1))
        self.bg = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        self.kw = set(cfg.get("kw", []))

    # ---- captions (top, word by word)
    def caption(self, d, t):
        for st, en, text in self.vo:
            if st - 0.05 <= t <= en + 0.35:
                words = text.split()
                wts = np.array([len(w) + 2 for w in words], float)
                cum = np.cumsum(wts) / wts.sum() * (en - st)
                k = min(int(np.searchsorted(cum, t - st)), len(words) - 1)
                n = 3
                g0 = (k // n) * n
                chunk = words[g0:g0 + n]
                size = 72
                while True:
                    fb = font(F_BOLD, size)
                    ws = [tw(d, w, fb) + 22 for w in chunk]
                    if sum(ws) <= W - 100 or size <= 44:
                        break
                    size -= 4
                x = (W - sum(ws)) / 2
                y = 215
                for i, w in enumerate(chunk):
                    cur = g0 + i == k
                    key = w.strip(".,!?;:").lower() in self.kw
                    if cur and key:
                        bx0, bx1, by0, by1 = x - 4, x + ws[i] - 14, y - size * 0.62, y + size * 0.62
                        d.rounded_rectangle([bx0, by0, bx1, by1], radius=8, fill=(60, 110, 255, 60), outline=(90, 150, 255, 255), width=3)
                        for cx_, cy_ in ((bx0, by0), (bx1, by0), (bx0, by1), (bx1, by1)):
                            d.ellipse([cx_ - 7, cy_ - 7, cx_ + 7, cy_ + 7], fill=(255, 255, 255, 255), outline=(90, 150, 255, 255), width=3)
                    d.text((x + 4, y), w, font=fb, fill=(255, 255, 255, 255) if cur else (150, 154, 168, 255), anchor="lm")
                    x += ws[i]
                return

    def cta(self, d, t):
        if t > self.T - 3.6:
            q = ease_out((t - (self.T - 3.6)) / 0.4)
            f = font(F_BOLD, 46)
            txt = "Follow @shipwithpip"
            w = tw(d, txt, f) + 90
            y = 1730 - (1 - q) * 24
            d.rounded_rectangle([(W - w) / 2, y, (W + w) / 2, y + 84], radius=42, fill=(255, 255, 255, int(235 * q)))
            d.text((W / 2, y + 42), txt, font=f, fill=(15, 15, 22, int(255 * q)), anchor="mm")

    def frame(self, t):
        img = self.bg.copy()
        for (a, b, fn) in self.cfg["scenes"]:
            if a - 0.01 <= t <= b + 0.5:
                fade = smooth(clamp((t - a) / 0.5)) * (1 - smooth(clamp((t - b) / 0.5)))
                if fade <= 0:
                    continue
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                ld = ImageDraw.Draw(layer, "RGBA")
                fn(self, layer, ld, t - a, b - a)
                if fade < 1:
                    layer.putalpha(layer.getchannel("A").point(lambda v: int(v * fade)))
                img = Image.alpha_composite(img, layer)
        rgb = img.convert("RGB")
        small = rgb.resize((W // 4, H // 4), Image.BILINEAR).point(lambda v: max(0, v - 70) * 2).filter(ImageFilter.GaussianBlur(7)).resize((W, H), Image.BICUBIC)
        rgb = ImageChops.add(rgb, small.point(lambda v: int(v * 0.9)))
        out = rgb.convert("RGBA")
        d = ImageDraw.Draw(out, "RGBA")
        self.caption(d, t)
        self.cta(d, t)
        return out.convert("RGB")


# ================================================================= REEL 1: microwave
def water_molecule(d, x, y, ang, s=1.0, heat=0.0, jit=0.0, rng=None):
    """O at (x,y). dipole axis points from H-side (+) toward O (-). ang = direction O->H midpoint reversed handled here."""
    # direction from O toward H midpoint
    a = ang + math.pi  # O->Hmid is opposite the dipole (dipole points toward O side)
    bl = 62 * s
    half = math.radians(52.25)
    base = mix((70, 130, 255), (255, 120, 40), heat)
    for sg in (-1, 1):
        hx, hy = x + bl * math.cos(a + sg * half), y + bl * math.sin(a + sg * half)
        d.line([(x, y), (hx, hy)], fill=(200, 205, 220, 200), width=max(2, int(5 * s)))
        d.ellipse([hx - 17 * s, hy - 17 * s, hx + 17 * s, hy + 17 * s], fill=(235, 238, 250, 255))
    d.ellipse([x - 32 * s, y - 32 * s, x + 32 * s, y + 32 * s], fill=base + (255,), outline=(255, 255, 255, 120), width=2)


def sc_oven(self, layer, d, t, T):
    x0, y0, x1, y1 = 70, 640, 1010, 1420
    d.rounded_rectangle([x0, y0, x1, y1], radius=36, fill=(14, 16, 24, 255), outline=(80, 86, 104, 255), width=7)
    p = 0.6 + 0.4 * math.sin(t * 7)
    d.rounded_rectangle([x0 + 24, 960, x0 + 150, 1100], radius=18, fill=col_a(AMBER, 0.35 + 0.4 * p), outline=col_a(AMBER, 1), width=5)
    label(d, x0 + 24, 1120, "MAGNETRON", 24)
    d.ellipse([740, 1240, 960, 1330], fill=(40, 90, 180, 255))
    d.pieslice([740, 1190, 960, 1390], 0, 180, fill=(50, 110, 210, 255))
    label(d, 850, 1405, "FOOD (MOSTLY WATER)", 22, anchor="ma")
    lam = 190
    k = 2 * math.pi / lam
    grow = ease_out(clamp(t / 1.2))
    for off, ph in ((-70, 0), (0, 1.1), (70, 2.1)):
        pts = []
        for x in range(x0 + 160, int(lerp(x0 + 160, 950, grow))):
            a = 34 * min(1, (x - x0 - 160) / 80)
            pts.append((x, 1030 + off + a * math.sin(k * x - 9 * t + ph)))
        if len(pts) > 2:
            d.line(pts, fill=col_a(GOLD, 0.9), width=5, joint="curve")
    if t > 2:
        a = ease_out(clamp((t - 2) / 1))
        yb = 810
        d.line([(x0 + 300, yb), (x0 + 300 + lam, yb)], fill=col_a(WHT, a), width=4)
        for xx in (x0 + 300, x0 + 300 + lam):
            d.line([(xx, yb - 14), (xx, yb + 14)], fill=col_a(WHT, a), width=4)
        d.text((x0 + 300 + lam / 2, yb - 44), "ONE WAVE ≈ 12 cm", font=font(F_SEMI, 32), fill=col_a(WHT, a), anchor="mm")
        d.text((CX, 590), "2,450,000,000 CYCLES PER SECOND", font=font(F_MONO, 30), fill=col_a(GOLD, a), anchor="mm")
    label(d, 70, 1560, "Simplified cutaway", 22)


MOLS = None


def mol_grid(rows=4, cols=3):
    rs = np.random.default_rng(7)
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((200 + c * 340 + (r % 2) * 120 + rs.uniform(-30, 30), 700 + r * 250 + rs.uniform(-25, 25), rs.uniform(0, 6.28), rs.uniform(0.7, 1.0)))
    return pts


def sc_molecules(self, layer, d, t, T):
    w = 2 * math.pi * 0.7
    c = math.cos(w * t)
    # field arrow
    mag = c * 150
    d.text((CX, 560), "ELECTRIC FIELD OF THE WAVE", font=font(F_MONO, 28), fill=col_a(GOLD, 1), anchor="mm")
    arrow(d, (CX - mag, 620), (CX + mag, 620), col_a(GOLD, 0.95), w=10, head=34) if abs(mag) > 25 else None
    heat = smooth(clamp((t - 7.0) / 7.0))
    th = math.pi * (0.5 + 0.5 * math.tanh(2.6 * c))  # dipole direction: 0 or pi
    rs = np.random.default_rng(3)
    for (x, y, ph, sc) in mol_grid():
        j = heat * 16
        jx, jy = j * math.sin(t * 23 + ph * 5), j * math.cos(t * 19 + ph * 3)
        lag = 0.25 * math.sin(ph)
        thm = math.pi * (0.5 + 0.5 * math.tanh(2.6 * math.cos(w * t - lag)))
        water_molecule(d, x + jx, y + jy, thm, sc * 1.05, heat)
    d.text((70, 1760 - 300), "", font=font(F_SEMI, 20))
    label(d, 70, 1740, "Magnified illustration. Slowed down billions of times.", 22)
    if heat > 0.05:
        a = clamp(heat * 3)
        d.text((CX, 1690), "MOTION BECOMES HEAT", font=font(F_BOLD, 40), fill=col_a(AMBER, a), anchor="mm")
        label(d, 70, 1775, "Colors illustrative", 22)


def cavity_field(n=200):
    L = 1.0
    x = np.linspace(0, L, n)[None, :]
    y = np.linspace(0, L, n)[:, None]
    f = np.sin(3 * np.pi * x) * np.sin(4 * np.pi * y) + 0.9 * np.sin(4 * np.pi * x) * np.sin(3 * np.pi * y + 0.6) + 0.35 * np.sin(5 * np.pi * x) * np.sin(2 * np.pi * y + 1.0)
    I = f ** 2
    return I / I.max()


_FIELD = None


def sc_turntable(self, layer, d, t, T):
    global _FIELD
    if _FIELD is None:
        _FIELD = cavity_field()
    I = _FIELD
    x0, y0, S = 120, 560, 840
    arr = np.zeros(I.shape + (3,), np.uint8)
    for ch in range(3):
        arr[..., ch] = np.vectorize(lambda u: heat_color(u)[ch])(I[::1, ::1]) if False else 0
    lut = np.array([heat_color(u / 255) for u in range(256)], np.uint8)
    arr = lut[(I * 255).astype(np.uint8)]
    im = Image.fromarray(arr, "RGB").resize((S, S), Image.BICUBIC)
    layer.paste(im.convert("RGBA"), (x0, y0))
    d.rectangle([x0, y0, x0 + S, y0 + S], outline=(80, 86, 104, 255), width=6)
    label(d, x0, y0 + S + 14, "Top view. Microwave energy pattern (illustrative).", 22)
    # plate with dots
    pc = (x0 + S / 2, y0 + S / 2)
    R = 250
    spin_t0 = 5.0
    ndots = 18
    rot = 0.0 if t < spin_t0 else 1.6 * (t - spin_t0)
    d.ellipse([pc[0] - R - 40, pc[1] - R - 40, pc[0] + R + 40, pc[1] + R + 40], outline=(230, 235, 250, 150), width=4)
    # accumulated heat per dot: integrate over time
    dt = 1 / 30
    steps = max(1, int(t / dt))
    ts = np.arange(steps) * dt
    angs = 0.0 + 1.6 * np.clip(ts - spin_t0, 0, None)
    vals = []
    for k in range(ndots):
        a0 = 2 * math.pi * k / ndots
        xs = pc[0] + R * np.cos(a0 + angs)
        ys = pc[1] + R * np.sin(a0 + angs)
        ix = np.clip(((xs - x0) / S * (I.shape[1] - 1)).astype(int), 0, I.shape[1] - 1)
        iy = np.clip(((ys - y0) / S * (I.shape[0] - 1)).astype(int), 0, I.shape[0] - 1)
        vals.append(float(I[iy, ix].mean()))
    vals = np.array(vals)
    for k in range(ndots):
        a = 2 * math.pi * k / ndots + rot
        x, y = pc[0] + R * math.cos(a), pc[1] + R * math.sin(a)
        v = vals[k] / max(vals.max(), 1e-6) if False else vals[k]
        d.ellipse([x - 26, y - 26, x + 26, y + 26], fill=heat_color(0.15 + 0.85 * clamp(v * 1.6)) + (255,), outline=(255, 255, 255, 220), width=3)
    spread = (vals.max() - vals.min())
    d.text((CX, 1500), "HEAT RECEIVED BY EACH SPOT OF FOOD", font=font(F_MONO, 26), fill=(200, 205, 220, 255), anchor="mm")
    bw = 760
    d.rounded_rectangle([CX - bw / 2, 1545, CX + bw / 2, 1575], radius=15, outline=(120, 124, 138, 255), width=3)
    d.rounded_rectangle([CX - bw / 2 + 4, 1549, CX - bw / 2 + 4 + (bw - 8) * clamp(1 - spread * 1.4), 1571], radius=12, fill=col_a(GRN if t > spin_t0 + 2 else AMBER, 1))
    d.text((CX, 1612), "UNEVEN" if t < spin_t0 + 2 else "EVENED OUT", font=font(F_BOLD, 40), fill=col_a(WHT, 1), anchor="mm")


def sc_mesh(self, layer, d, t, T):
    x0, x1, y0, y1 = 360, 720, 600, 1500
    d.rectangle([x0, y0, x1, y1], fill=(36, 40, 52, 255), outline=(110, 116, 136, 255), width=4)
    for r in range(int((y1 - y0 - 30) / 34)):
        for c in range(int((x1 - x0 - 30) / 34)):
            d.ellipse([x0 + 22 + c * 34 - 7, y0 + 28 + r * 34 - 7, x0 + 22 + c * 34 + 7, y0 + 28 + r * 34 + 7], fill=(3, 4, 7, 255))
    lam = 260
    k = 2 * math.pi / lam
    for off in (-260, 0, 260):
        pts = [(x, 1050 + off + 60 * math.sin(k * x - 8 * t)) for x in range(30, x0)]
        d.line(pts, fill=col_a(GOLD, 0.95), width=6, joint="curve")
        d.line([(x1 + 4, 1050 + off), (1050, 1050 + off)], fill=col_a(GOLD, 0.15), width=4)
        pts = [(x, 1050 + off + 22 * math.sin(k * x + 8 * t) * clamp((x - 180) / 180)) for x in range(180, x0)]
    d.line([(60, 560), (60 + lam, 560)], fill=col_a(WHT, 1), width=4)
    for xx in (60, 60 + lam):
        d.line([(xx, 546), (xx, 574)], fill=col_a(WHT, 1), width=4)
    d.text((60 + lam / 2, 520), "WAVE ≈ 12 cm", font=font(F_SEMI, 30), fill=col_a(WHT, 1), anchor="mm")
    label(d, x0, y1 + 20, "DOOR MESH: holes far smaller than the wave", 22)
    d.text((880, 1050), "BLOCKED", font=font(F_BOLD, 40), fill=col_a(GRN, smooth(clamp((t - 2) / 1))), anchor="mm")
    label(d, 60, 1620, "Not to scale", 22)


def build_micro(model):
    L = [
        (0.2, "Your microwave has no flame and no hot coil. So what is cooking your food?"),
        (5.0, "Inside is a magnetron. It makes invisible waves, 2.45 billion cycles every second, each about 12 centimeters long."),
        (14.5, "Water molecules are slightly positive on one end and slightly negative on the other."),
        (21.0, "So when the wave flips its electric field, every water molecule twists to follow it, about five billion times a second."),
        (31.0, "All that twisting and bumping turns into heat. Ice, where the molecules are locked in place, absorbs far less."),
        (40.0, "But the waves bounce around the box, making hot spots and cold spots. That is why there is a turntable."),
        (50.0, "And the holes in the door are much smaller than the waves, so they cannot get out."),
        (57.5, "Your microwave is just shaking water. Comment MICRO for more."),
    ]
    cfg = dict(T=90.0, seed=11, voice="af_heart", speed=0.98, kw=["microwave", "magnetron", "water", "heat", "twists", "turntable", "waves", "billion", "hot", "spots", "mesh", "door", "shaking"],
               scenes=[(0, 14.0, sc_oven), (14.0, 39.0, sc_molecules), (39.0, 49.0, sc_turntable), (49.0, 62.0, sc_mesh)],
               still_times=[3, 9, 17, 27, 36, 44, 53, 60])
    buf, vo = make_audio(cfg, L, model)
    cfg["T"] = vo[-1][1] + 2.4
    cfg["scenes"][-1] = (49.0, cfg["T"] + 0.5, sc_mesh)
    reel = Macro(cfg, vo)
    post = dict(title="MACRO-1  how a microwave really heats food",
                caption="A microwave has no flame and no heating coil. Its magnetron makes microwaves at 2.45 GHz, about 12 cm per wave. Water molecules are polar, so as the electric field flips billions of times per second they twist back and forth, and that motion becomes heat. The waves bounce around the box and make hot and cold spots, which is why there is a turntable. The metal mesh on the door has holes much smaller than the wave, so the waves stay in.\n\nComment MICRO for the next one.",
                tags=["science", "physics", "howthingswork", "microwave", "stem", "learnontiktok", "shipwithpip"],
                pinned="Comment MICRO and tell me which everyday gadget to break down next.",
                notes="Verified (SchoolPhysics microwave ovens page): microwave ovens use 2.45 GHz, wavelength about 12 cm; water molecules oscillate about 5 thousand million times a second; ice absorbs microwaves weakly while water absorbs strongly; reflections off oven walls create hot spots and a rotating table evens cooking; the door mesh holes are small enough to stop 12 cm waves. Visual notes: wave, molecule motion and the energy pattern are illustrations (the pattern is a computed standing-wave sum with modes (3,4),(4,3),(5,2), not a measurement of a real oven). Molecule geometry uses the real H-O-H angle of 104.5 degrees. Motion is slowed down billions of times.")
    return reel, buf, cfg, post


# ================================================================= REEL 2: wireless charging
def field_loops(d, cx, cy, t, amp, gap_top, n=5, col=ICE):
    """dipole-ish loops above/below a coil centre"""
    for i in range(1, n + 1):
        rx = 90 * i
        ry = 80 + 70 * i
        for sgn in (-1, 1):
            pts = []
            for u in np.linspace(0, math.pi, 40):
                pts.append((cx + sgn * rx * (1 - math.cos(u)) / 2 * 1.0 + 0, cy - ry * math.sin(u) * amp))
            a = 0.85 - 0.12 * i
            d.line(pts, fill=col_a(col, a * abs(amp)), width=4, joint="curve")


def coil_xs(cx, n=5, sp=64):
    return [cx + (i - (n - 1) / 2) * sp for i in range(n)]


def sc_qi_overview(self, layer, d, t, T, show_current=True):
    pad_y, ph_y = 1300, 960
    # pad
    d.rounded_rectangle([140, pad_y, 940, pad_y + 140], radius=26, fill=(24, 26, 36, 255), outline=(90, 96, 116, 255), width=5)
    # phone
    d.rounded_rectangle([200, ph_y, 880, ph_y + 120], radius=26, fill=(18, 20, 28, 255), outline=(150, 156, 176, 255), width=5)
    label(d, 220, ph_y - 38, "PHONE", 24)
    label(d, 160, pad_y + 160, "CHARGING PAD", 24)
    w = 2 * math.pi * 0.5
    c = math.sin(w * t)
    for x in coil_xs(CX, 5, 70):
        d.ellipse([x - 22, pad_y + 52, x + 22, pad_y + 96], fill=col_a((205, 125, 70), 1), outline=(255, 190, 120, 255), width=3)
        d.ellipse([x - 22, ph_y + 24, x + 22, ph_y + 68], fill=col_a((205, 125, 70), 1), outline=(255, 190, 120, 255), width=3)
    amp = c
    # field loops between and around
    for i in range(1, 5):
        rx = 130 + 95 * i
        ry = 120 + 80 * i
        pts = [(CX - rx * math.cos(u) * 1.0, (pad_y + 74) - ry * math.sin(u) * 0.55) for u in np.linspace(0, math.pi, 60)]
        d.line(pts, fill=col_a(ICE, abs(amp) * (0.9 - 0.15 * i)), width=4, joint="curve")
    d.text((CX, 780), "ALTERNATING CURRENT  →  CHANGING MAGNETIC FIELD", font=font(F_MONO, 26), fill=col_a(ICE, 1), anchor="mm")


def sc_qi_ac(self, layer, d, t, T):
    # sine panel
    x0, x1, y0 = 90, 990, 1000
    d.rounded_rectangle([x0 - 20, 700, x1 + 20, 1300], radius=28, fill=(14, 16, 24, 255), outline=(70, 76, 96, 255), width=4)
    d.line([(x0, y0), (x1, y0)], fill=(90, 96, 116, 255), width=3)
    pts = [(x, y0 - 170 * math.sin((x - x0) / 900 * 2 * math.pi * 3 - 2 * math.pi * 0.35 * t)) for x in range(x0, x1)]
    d.line(pts, fill=col_a(GOLD, 1), width=6, joint="curve")
    px = x1 - 60
    d.ellipse([px - 12, pts[px - x0][1] - 12, px + 12, pts[px - x0][1] + 12], fill=(255, 255, 255, 255))
    d.text((CX, 760), "CURRENT IN THE PAD COIL", font=font(F_MONO, 28), fill=(200, 205, 220, 255), anchor="mm")
    d.text((CX, 1380), "110 to 205 kHz", font=font(F_BOLD, 78), fill=col_a(GOLD, 1), anchor="mm")
    label(d, CX, 1450, "typical Qi charger up to 5 W: over 100,000 flips a second", 24, anchor="ma")
    label(d, 90, 1620, "Waveform slowed down. Values from the Qi standard range.", 22)


def sc_qi_induce(self, layer, d, t, T):
    sc_qi_overview(self, layer, d, t, T)
    # electrons around phone coil
    base = 960 + 46
    for i in range(14):
        u = (t * 0.7 + i / 14) % 1
        x = 230 + 620 * u
        d.ellipse([x - 8, base - 8 + 6 * math.sin(i + t * 6), x + 8, base + 8 + 6 * math.sin(i + t * 6)], fill=col_a(GOLD, 0.95))
    # battery
    q = clamp(t / (T * 0.9))
    d.rounded_rectangle([760, 880, 900, 940], radius=10, outline=(230, 235, 250, 255), width=4)
    d.rounded_rectangle([766, 886, 766 + 128 * q, 934], radius=6, fill=col_a(GRN, 1))
    label(d, 830, 846, "BATTERY", 22, anchor="ma")
    d.text((CX, 1580), "THE MOVING FIELD PUSHES ELECTRONS IN THE PHONE COIL", font=font(F_MONO, 24), fill=(200, 205, 220, 255), anchor="mm")


def sc_qi_align(self, layer, d, t, T):
    off = 190 * smooth(clamp((t - 1) / 5.0))
    pad_y, ph_y = 1250, 930
    d.rounded_rectangle([140, pad_y, 940, pad_y + 120], radius=26, fill=(24, 26, 36, 255), outline=(90, 96, 116, 255), width=5)
    d.rounded_rectangle([200 + off, ph_y, 880 + off - 0, ph_y + 110], radius=26, fill=(18, 20, 28, 255), outline=(150, 156, 176, 255), width=5)
    tx = CX - 120
    rx = CX - 120 + off
    d.ellipse([tx - 70, pad_y + 40, tx + 70, pad_y + 80], fill=col_a((205, 125, 70), 1))
    d.ellipse([rx - 70, ph_y + 36, rx + 70, ph_y + 74], fill=col_a((205, 125, 70), 1))
    coup = math.exp(-(off / 210) ** 2)
    for i in range(1, 5):
        rr = 60 + 60 * i
        pts = [(tx - rr * math.cos(u), pad_y + 60 - rr * 0.9 * math.sin(u)) for u in np.linspace(0, math.pi, 40)]
        d.line(pts, fill=col_a(ICE, 0.7 - 0.1 * i), width=4, joint="curve")
    d.text((CX, 640), "LINED UP", font=font(F_BOLD, 50), fill=col_a(GRN if coup > 0.7 else (RED if coup < 0.4 else AMBER), 1), anchor="mm") if False else None
    d.text((CX, 660), "SHARE OF THE FIELD THAT REACHES THE PHONE", font=font(F_MONO, 24), fill=(200, 205, 220, 255), anchor="mm")
    bw = 760
    d.rounded_rectangle([CX - bw / 2, 700, CX + bw / 2, 750], radius=22, outline=(120, 124, 138, 255), width=3)
    d.rounded_rectangle([CX - bw / 2 + 5, 705, CX - bw / 2 + 5 + (bw - 10) * coup, 745], radius=18, fill=col_a(mix(RED, GRN, coup), 1))
    label(d, 70, 1620, "Coupling curve illustrative: it falls as the coils drift apart.", 22)


def sc_qi_talk(self, layer, d, t, T):
    pad_y, ph_y = 1250, 900
    d.rounded_rectangle([140, pad_y, 940, pad_y + 120], radius=26, fill=(24, 26, 36, 255), outline=(90, 96, 116, 255), width=5)
    d.rounded_rectangle([200, ph_y, 880, ph_y + 110], radius=26, fill=(18, 20, 28, 255), outline=(150, 156, 176, 255), width=5)
    msgs = ["MORE POWER", "A BIT LESS", "FULL. STOP."]
    i = min(int(t / (T / 3)), 2)
    u = (t % (T / 3)) / (T / 3)
    # packet travels phone -> pad
    y = lerp(ph_y + 130, pad_y - 20, ease_out(u))
    d.rounded_rectangle([CX - 190, y - 34, CX + 190, y + 34], radius=18, fill=col_a(GOLD, 0.95 * (1 - 0.6 * u)))
    d.text((CX, y), msgs[i], font=font(F_BOLD, 34), fill=(20, 16, 6, 255), anchor="mm")
    d.text((CX, 760), "THE PHONE TALKS TO THE CHARGER", font=font(F_MONO, 28), fill=(200, 205, 220, 255), anchor="mm")
    label(d, 70, 1620, "Illustration of the Qi control messages.", 22)


def sc_qi_coin(self, layer, d, t, T):
    pad_y = 1250
    d.rounded_rectangle([140, pad_y, 940, pad_y + 120], radius=26, fill=(24, 26, 36, 255), outline=(90, 96, 116, 255), width=5)
    heat = clamp(t / 4.0)
    cut = t > 5.0
    h = 0.0 if cut and False else heat if not cut else max(0.0, 1 - (t - 5) / 2.5) * heat
    d.ellipse([CX - 110, pad_y - 40, CX + 110, pad_y + 10], fill=mix((150, 154, 170), (255, 120, 40), h) + (255,), outline=(255, 255, 255, 200), width=3)
    if not cut:
        for i in range(1, 4):
            rr = 90 + 80 * i
            pts = [(CX - rr * math.cos(u), pad_y - 10 - rr * 0.8 * math.sin(u)) for u in np.linspace(0, math.pi, 40)]
            d.line(pts, fill=col_a(ICE, 0.7 - 0.15 * i), width=4)
    d.text((CX, 760), "METAL ON THE PAD" if not cut else "CHARGER SHUTS OFF", font=font(F_BOLD, 54), fill=col_a(AMBER if not cut else GRN, 1), anchor="mm")
    d.text((CX, 840), "Currents are induced in the coin too.", font=font(F_SEMI, 30), fill=(200, 205, 220, 255), anchor="mm")
    label(d, 70, 1620, "Illustrative.", 22)


def build_qi(model):
    L = [
        (0.2, "Your phone is charging, and there is no wire. So how does the power get across?"),
        (5.8, "Under the pad is a flat coil of copper wire. Alternating current flows through it, cycling over a hundred thousand times every second."),
        (16.5, "A changing current makes a changing magnetic field around the coil."),
        (22.5, "Your phone has its own coil. That moving field pushes electrons through it. It is basically a transformer with an air gap."),
        (33.5, "That current charges the battery. But the coils must stay close and lined up, or less of the field reaches your phone."),
        (44.5, "The charger and phone even talk. The phone asks for more or less power, and says when it is full."),
        (54.0, "And metal matters. A coin on the pad gets currents induced in it and can heat up, so chargers detect it and shut off."),
        (64.0, "No wire. Just a magnetic field. Comment WIRELESS for more."),
    ]
    cfg = dict(T=90.0, seed=21, voice="af_heart", speed=0.98, kw=["wire", "coil", "magnetic", "field", "alternating", "transformer", "battery", "lined", "talk", "metal", "coin", "heat", "shut"],
               scenes=[(0, 5.8, sc_qi_overview), (5.8, 16.0, sc_qi_ac), (16.0, 33.0, sc_qi_induce), (33.0, 44.0, sc_qi_align), (44.0, 53.0, sc_qi_talk), (53.0, 90.0, sc_qi_coin)],
               still_times=[3, 9, 20, 28, 38, 49, 58, 66])
    buf, vo = make_audio(cfg, L, model)
    cfg["T"] = vo[-1][1] + 2.4
    cfg["scenes"][-1] = (53.0, cfg["T"] + 0.5, sc_qi_coin)
    cfg["scenes"][0] = (0, 5.8, lambda s, l, d, t, T: sc_qi_overview(s, l, d, t, T))
    reel = Macro(cfg, vo)
    post = dict(title="MACRO-2  how wireless charging works",
                caption="Wireless charging is a transformer with an air gap. A coil in the pad carries alternating current (typical low-power Qi chargers run roughly 110 to 205 kHz), which makes a changing magnetic field. A second coil in the phone has current induced in it by that field, and that charges the battery. The coils need to be close and lined up, the phone and charger exchange messages about how much power to send, and metal objects on the pad can heat up, so chargers detect them and shut off.\n\nComment WIRELESS for the next one.",
                tags=["science", "physics", "howthingswork", "wirelesscharging", "iphone", "tech", "learnontiktok", "shipwithpip"],
                pinned="Comment WIRELESS and tell me which everyday gadget to break down next.",
                notes="Verified (electronics-notes Qi page): Qi uses magnetic induction between a transmitting coil in the base and a receiver coil in the device; low-power chargers up to 5 W run about 110-205 kHz (medium power 80-300 kHz); the charger detects objects and validates devices before charging; the device sends control packets to adjust power and end charging when full; metal objects near the charger can have currents induced and become very hot, so foreign-object detection cuts charging. Verified (Mouser): Qi works like a transformer's primary and secondary windings; coupling is strongest with planar coils close together. Visual notes: field lines, the coupling-versus-offset bar and the coin heating are illustrations (the coupling bar uses a simple Gaussian falloff, not measured data).")
    return reel, buf, cfg, post


# ================================================================= REEL 3: GPS
def sc_gps_orbit(self, layer, d, t, T):
    cx, cy = CX, 1080
    R = 190
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(24, 60, 120, 255), outline=(120, 180, 255, 255), width=4)
    rng = np.random.default_rng(4)
    for (lx, ly, r) in ((-60, -40, 55), (50, 40, 70), (-20, 80, 35)):
        d.ellipse([cx + lx - r, cy + ly - r, cx + lx + r, cy + ly + r], fill=(40, 120, 90, 255))
    Ro = 520
    d.ellipse([cx - Ro, cy - Ro * 0.38, cx + Ro, cy + Ro * 0.38], outline=(90, 96, 116, 255), width=3)
    n = 6
    for i in range(n):
        a = 2 * math.pi * i / n + t * 0.35
        x, y = cx + Ro * math.cos(a), cy + Ro * 0.38 * math.sin(a)
        back = math.sin(a) < 0
        if back:
            continue
        d.rectangle([x - 22, y - 8, x + 22, y + 8], fill=(240, 220, 120, 255))
        d.rectangle([x - 56, y - 5, x - 26, y + 5], fill=(80, 130, 230, 255))
        d.rectangle([x + 26, y - 5, x + 56, y + 5], fill=(80, 130, 230, 255))
    d.text((CX, 600), "ABOUT 20,200 km UP", font=font(F_BOLD, 62), fill=col_a(WHT, 1), anchor="mm")
    d.text((CX, 680), "31 operational satellites (2023)", font=font(F_MONO, 28), fill=(200, 205, 220, 255), anchor="mm")
    label(d, 70, 1640, "Not to scale. Orbit drawn simplified.", 22)


def sc_gps_signal(self, layer, d, t, T):
    sx, sy = 200, 760
    d.rectangle([sx - 30, sy - 12, sx + 30, sy + 12], fill=(240, 220, 120, 255))
    label(d, sx, sy - 60, "SATELLITE", 24, anchor="ma")
    px, py = 880, 1380
    d.rounded_rectangle([px - 36, py - 60, px + 36, py + 60], radius=10, fill=(18, 20, 28, 255), outline=(150, 156, 176, 255), width=4)
    label(d, px, py + 80, "YOUR PHONE", 24, anchor="ma")
    dist = math.hypot(px - sx, py - sy)
    for k in range(4):
        r = ((t * 0.3 + k / 4) % 1) * dist * 1.05
        d.ellipse([sx - r, sy - r, sx + r, sy + r], outline=col_a(GOLD, 0.8 * (1 - r / (dist * 1.05))), width=4)
    d.text((CX, 560), "\"IT IS EXACTLY 12:00:00.000000\"", font=font(F_MONO, 30), fill=col_a(GOLD, 1), anchor="mm")
    a = smooth(clamp((t - 4) / 1.2))
    d.text((CX, 1560), "1 microsecond of delay  ≈  300 meters", font=font(F_BOLD, 44), fill=col_a(WHT, a), anchor="mm")
    label(d, 70, 1640, "Radio signals travel at the speed of light.", 22)


def sc_gps_tri(self, layer, d, t, T):
    me = (560, 1150)
    sats = [(200, 760), (880, 740), (540, 1560)]
    cols = [GOLD, ICE, GRN]
    d.ellipse([me[0] - 14, me[1] - 14, me[0] + 14, me[1] + 14], fill=(255, 255, 255, 255))
    stage = int(clamp(t / 4.0, 0, 3))
    names = ["ONE SATELLITE: somewhere on a circle", "TWO: two possible spots", "THREE: your spot", "A FOURTH fixes your own clock error"]
    for i, s in enumerate(sats):
        if t < i * 4.0:
            continue
        r = math.hypot(me[0] - s[0], me[1] - s[1]) * smooth(clamp((t - i * 4.0) / 1.2))
        d.ellipse([s[0] - r, s[1] - r, s[0] + r, s[1] + r], outline=col_a(cols[i], 0.9), width=4)
        d.rectangle([s[0] - 20, s[1] - 8, s[0] + 20, s[1] + 8], fill=(240, 220, 120, 255))
    d.text((CX, 600), names[min(int(t / 4.0), 3) if t < 12 else 3], font=font(F_SEMI, 34), fill=(230, 235, 250, 255), anchor="mm")
    if t > 12:
        a = smooth(clamp((t - 12) / 1))
        d.ellipse([me[0] - 44, me[1] - 44, me[0] + 44, me[1] + 44], outline=col_a(RED, a), width=5)
        d.text((CX, 1640), "4 unknowns: x, y, z and the clock", font=font(F_MONO, 28), fill=col_a(WHT, a), anchor="mm")
    label(d, 70, 1700, "2D sketch of a 3D problem (spheres in real life).", 22)


def sc_gps_rel(self, layer, d, t, T):
    items = [("MOVING FAST (4 km/s)", -7, RED), ("WEAKER GRAVITY", 45, GRN), ("NET PER DAY", 38, GOLD)]
    y0 = 760
    zero = 430
    scale = 8
    d.line([(zero, y0 - 60), (zero, y0 + 3 * 170)], fill=(120, 124, 138, 255), width=3)
    for i, (nm, v, c) in enumerate(items):
        q = smooth(clamp((t - i * 3.0) / 1.5))
        y = y0 + i * 170
        x2 = zero + v * scale * q
        d.rounded_rectangle([min(zero, x2), y, max(zero, x2), y + 70], radius=10, fill=col_a(c, 1))
        d.text((CX, y - 24), nm, font=font(F_MONO, 26), fill=(200, 205, 220, 255), anchor="mm")
        if q > 0.6:
            d.text((x2 + (30 if v > 0 else -30), y + 35), f"{v:+d} µs", font=font(F_BOLD, 40), fill=col_a(WHT, 1), anchor="lm" if v > 0 else "rm")
    label(d, 70, 1400, "Satellite clock vs a clock on the ground, per day.", 22)
    d.text((CX, 600), "EINSTEIN FIXES YOUR MAP", font=font(F_BOLD, 52), fill=col_a(WHT, 1), anchor="mm")


def sc_gps_drift(self, layer, d, t, T):
    q = smooth(clamp(t / 6.0))
    d.text((CX, 600), "IF NOBODY CORRECTED IT", font=font(F_BOLD, 52), fill=col_a(WHT, 1), anchor="mm")
    x0, y0 = 160, 1450
    d.line([(x0, y0), (940, y0)], fill=(90, 96, 116, 255), width=3)
    km = 11 * q
    xe = x0 + (940 - x0) * q
    d.ellipse([x0 - 14, y0 - 14, x0 + 14, y0 + 14], fill=(255, 255, 255, 255))
    d.line([(x0, y0), (xe, y0)], fill=col_a(RED, 1), width=8)
    d.ellipse([xe - 14, y0 - 14, xe + 14, y0 + 14], fill=col_a(RED, 1))
    d.text((CX, 1000), f"{km:0.1f} km", font=font(F_BOLD, 150), fill=col_a(RED, 1), anchor="mm")
    d.text((CX, 1160), "of ranging error after one day", font=font(F_SEMI, 36), fill=(220, 225, 240, 255), anchor="mm")
    label(d, 70, 1640, "Per GPS World: up to about 11 km between daily clock updates.", 22)


def build_gps(model):
    L = [
        (0.2, "Your phone knows where you are, but GPS never sends anything to find out. Here is the trick."),
        (6.0, "More than thirty satellites circle Earth, about twenty thousand kilometers up. Each one just keeps shouting the exact time from an atomic clock."),
        (17.0, "Radio travels at the speed of light, so your phone compares when the time was sent with when it arrived. One microsecond is about three hundred meters."),
        (28.0, "One satellite puts you on a huge sphere. Two make a circle. Three give you a spot. And a fourth fixes your own clock error."),
        (42.5, "And those clocks need Einstein. Moving fast slows them by about seven microseconds a day. Weaker gravity speeds them up by about forty five. Net, thirty eight microseconds a day."),
        (56.5, "Skip that correction, and the error grows to around eleven kilometers in a day."),
        (63.5, "Your map is relativity in your pocket. Comment GPS for more."),
    ]
    cfg = dict(T=90.0, seed=31, voice="af_heart", speed=0.98, kw=["gps", "satellites", "atomic", "clock", "light", "microsecond", "meters", "fourth", "einstein", "relativity", "eleven", "kilometers", "correction"],
               scenes=[(0, 6.0, sc_gps_orbit), (6.0, 17.0, sc_gps_orbit), (17.0, 28.0, sc_gps_signal), (28.0, 42.5, sc_gps_tri), (42.5, 56.0, sc_gps_rel), (56.0, 90.0, sc_gps_drift)],
               still_times=[3, 10, 22, 33, 41, 50, 58, 65])
    cfg["scenes"] = [(0, 17.0, sc_gps_orbit), (17.0, 28.0, sc_gps_signal), (28.0, 42.5, sc_gps_tri), (42.5, 56.0, sc_gps_rel), (56.0, 90.0, sc_gps_drift)]
    buf, vo = make_audio(cfg, L, model)
    cfg["T"] = vo[-1][1] + 2.4
    cfg["scenes"][-1] = (56.0, cfg["T"] + 0.5, sc_gps_drift)
    reel = Macro(cfg, vo)
    post = dict(title="MACRO-3  how GPS knows where you are",
                caption="GPS is receive-only. Each satellite, about 20,200 km up, broadcasts the exact time from an atomic clock. Your phone compares send time with arrival time (one microsecond of delay is about 300 meters at light speed), and with four satellites it solves for x, y, z and its own clock error. The satellite clocks also need relativity corrections: about 7 microseconds slower per day from speed, about 45 faster from weaker gravity, net 38 microseconds a day. Without correction the ranging error would grow to roughly 11 km.\n\nComment GPS for the next one.",
                tags=["science", "physics", "howthingswork", "gps", "einstein", "relativity", "tech", "learnontiktok", "shipwithpip"],
                pinned="Comment GPS and tell me which everyday gadget to break down next.",
                notes="Verified (GPS.gov Space Segment): 31 operational satellites as of July 2023, minimum commitment 24; altitude about 20,200 km; two orbits a day. Verified (GPS World, Inside the Box: GPS and Relativity): about 4 km/s orbital speed; special relativity about -7 microseconds/day, general relativity about +45, net about +38 microseconds/day; four satellites needed for a fix (3 position unknowns plus clock bias); uncorrected range error up to about 11 km between daily clock updates. Math check: light travels about 299,792 km/s, so 1 microsecond is about 300 m. Visual notes: orbit, circles and bars are illustrations; the bars use the real microsecond values.")
    return reel, buf, cfg, post


# ================================================================= REEL 4: screen pixels
def tiny_art(n=40):
    im = Image.new("RGB", (n, n), (10, 20, 60))
    d = ImageDraw.Draw(im)
    for y in range(n):
        c = mix((20, 30, 90), (255, 130, 60), (y / n) ** 1.6)
        d.line([(0, y), (n, y)], fill=c)
    d.ellipse([n * 0.28, n * 0.3, n * 0.72, n * 0.74], fill=(255, 215, 40))
    d.polygon([(0, n), (0, n * 0.78), (n * 0.3, n * 0.62), (n * 0.55, n * 0.82), (n * 0.8, n * 0.66), (n, n * 0.8), (n, n)], fill=(25, 20, 40))
    return im


_ART = None


def zoom_frame(z, cx_img, cy_img, wpx=960, hpx=1160, n=40, base_px=22):
    """vectorised subpixel render. z = zoom factor; each source pixel is base_px*z screen px"""
    global _ART
    if _ART is None:
        _ART = np.asarray(tiny_art(n), np.float32)
    ps = base_px * z
    ys, xs = np.mgrid[0:hpx, 0:wpx]
    u = (xs - wpx / 2) / ps + cx_img
    v = (ys - hpx / 2) / ps + cy_img
    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
    inside = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
    iu_c, iv_c = np.clip(iu, 0, n - 1), np.clip(iv, 0, n - 1)
    pix = _ART[iv_c, iu_c]
    fu, fv = u - iu, v - iv
    sub = np.clip((fu * 3).astype(int), 0, 2)
    stripe = np.zeros_like(pix)
    for c in range(3):
        stripe[..., c] = np.where(sub == c, pix[..., c], 0)
    gap = 0.07 + 0.0 * fu
    inbar = (np.abs(((fu * 3) % 1) - 0.5) < 0.40) & (np.abs(fv - 0.5) < 0.46)
    sub_mode = np.clip((ps - 10) / 24, 0, 1)
    flat = pix
    sub_img = stripe * inbar[..., None] * 1.25
    out = flat * (1 - sub_mode) + sub_img * sub_mode
    out = np.where(inside[..., None], out, 6)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def sc_pix_zoom(self, layer, d, t, T):
    u = smooth(clamp((t - 1.5) / (T - 3.0)))
    z = math.exp(lerp(math.log(1.0), math.log(18.0), u))
    im = zoom_frame(z, lerp(20, 24.5, u), lerp(20, 17.5, u))
    layer.paste(im.convert("RGBA"), (60, 420))
    d.rectangle([60, 420, 1020, 1580], outline=(80, 86, 104, 255), width=5)
    msg = "A PHOTO" if u < 0.25 else ("A GRID OF PIXELS" if u < 0.7 else "EACH PIXEL: 3 TINY LIGHTS")
    d.text((CX, 1650), msg, font=font(F_BOLD, 46), fill=col_a(WHT, 1), anchor="mm")
    label(d, 60, 1590, "Magnified illustration. Layout shown is a typical RGB stripe.", 22)


def sc_pix_mix(self, layer, d, t, T):
    r = 210
    cs = [((CX - 120, 960), RED), ((CX + 120, 960), GRN), ((CX, 1150), BLUE)]
    a = smooth(clamp(t / 1.5))
    base = Image.new("RGB", (W, H), (0, 0, 0))
    arr = np.zeros((H, W, 3), np.float32)
    Y, X = np.mgrid[0:H, 0:W]
    for (x, y), c in cs:
        m = ((X - x) ** 2 + (Y - y) ** 2 < (r * a) ** 2)
        arr[m] += np.array(c, np.float32)
    arr = np.clip(arr, 0, 255)
    m = arr.sum(axis=2) > 0
    lay = np.zeros((H, W, 4), np.uint8)
    lay[..., :3] = arr.astype(np.uint8)
    lay[..., 3] = np.where(m, 255, 0)
    layer.paste(Image.fromarray(lay, "RGBA"), (0, 0))
    d.text((CX, 680), "LIGHT ADDS UP", font=font(F_BOLD, 56), fill=col_a(WHT, 1), anchor="mm")
    names = [("RED + GREEN = YELLOW", 4.0), ("ALL THREE = WHITE", 8.0)]
    for txt, ts in names:
        if t >= ts:
            pass
    k = 0 if t < 8 else 1
    if t >= 4:
        d.text((CX, 1480), names[k][0], font=font(F_BOLD, 44), fill=col_a(WHT if k else GOLD, 1), anchor="mm")
    label(d, 60, 1640, "No yellow light source needed on most screens.", 22)


def sc_pix_swatches(self, layer, d, t, T):
    cols = [("WHITE", (255, 255, 255)), ("BLACK", (0, 0, 0)), ("YELLOW", (255, 255, 0)), ("ORANGE", (255, 140, 0)), ("PURPLE", (150, 60, 220))]
    i = min(int(t / (T / len(cols))), len(cols) - 1)
    nm, c = cols[i]
    d.rounded_rectangle([CX - 280, 640, CX + 280, 1040], radius=34, fill=c + (255,), outline=(230, 235, 250, 255), width=5)
    d.text((CX, 1090), nm, font=font(F_BOLD, 54), fill=col_a(WHT, 1), anchor="mm")
    for k, (ch, cc) in enumerate((("R", RED), ("G", GRN), ("B", BLUE))):
        x = CX - 300 + k * 300
        v = c[k] / 255
        d.rounded_rectangle([x - 60, 1180, x + 60, 1500], radius=14, outline=(120, 124, 138, 255), width=3)
        d.rounded_rectangle([x - 54, 1494 - 308 * v, x + 54, 1494], radius=10, fill=col_a(cc, 1))
        d.text((x, 1540), f"{ch} {c[k]}", font=font(F_MONO, 32), fill=(220, 225, 240, 255), anchor="mm")
    label(d, 60, 1640, "Brightness of each subpixel, 0 to 255.", 22)


def sc_pix_pentile(self, layer, d, t, T):
    x0, y0 = 160, 700
    sp = 90
    cols = [RED, GRN, BLUE]
    a = smooth(clamp(t / 1.2))
    d.text((CX, 600), "NOT EVERY SCREEN IS A PERFECT TRIO", font=font(F_BOLD, 40), fill=col_a(WHT, 1), anchor="mm")
    for r in range(6):
        for c in range(9):
            if (r + c) % 2 == 0:
                col = GRN
            elif r % 2 == 0:
                col = RED if c % 4 == 1 else BLUE
            else:
                col = BLUE if c % 4 == 1 else RED
            x, y = x0 + c * sp, y0 + r * sp + 40
            if col == GRN:
                d.ellipse([x + 10, y + 14, x + 40, y + 44], fill=col_a(col, a))
            else:
                d.rounded_rectangle([x + 4, y + 4, x + 56, y + 56], radius=10, fill=col_a(col, a))
    d.text((CX, 1380), "Some OLED screens share subpixels", font=font(F_SEMI, 36), fill=(220, 225, 240, 255), anchor="mm")
    d.text((CX, 1440), "between neighboring pixels (PenTile)", font=font(F_SEMI, 36), fill=(220, 225, 240, 255), anchor="mm")
    label(d, 60, 1640, "Illustrative layout.", 22)


def build_pixel(model):
    L = [
        (0.2, "Your phone screen can show millions of colors. But every single one is made from just three."),
        (6.5, "Zoom in far enough and the picture breaks into tiny lights of red, green and blue."),
        (14.5, "Those are subpixels. Make all three bright, and your eyes see white. Turn them all off, and you get black."),
        (24.5, "Yellow? There is no yellow light on most screens. Just red and green glowing together, and your brain fills in the rest."),
        (34.0, "Change how bright each one is, and the screen mixes millions of colors."),
        (41.5, "Some OLED screens even share subpixels between neighbors, so a pixel is not always a perfect red green blue trio."),
        (51.0, "Comment PIXEL for the next one."),
    ]
    cfg = dict(T=90.0, seed=41, voice="af_heart", speed=0.98, kw=["three", "red", "green", "blue", "subpixels", "white", "black", "yellow", "millions", "oled", "share"],
               scenes=[(0, 14.0, sc_pix_zoom), (14.0, 34.0, sc_pix_mix), (34.0, 41.0, sc_pix_swatches), (41.0, 90.0, sc_pix_pentile)],
               still_times=[1, 6, 11, 17, 27, 37, 45, 52])
    buf, vo = make_audio(cfg, L, model)
    cfg["T"] = vo[-1][1] + 2.4
    cfg["scenes"][-1] = (41.0, cfg["T"] + 0.5, sc_pix_pentile)
    reel = Macro(cfg, vo)
    post = dict(title="MACRO-4  your screen only has 3 colors",
                caption="Every color on your screen comes from three kinds of tiny lights: red, green and blue subpixels. All three bright looks white, all off looks black, red plus green looks yellow, and varying each one's brightness gives millions of colors. Some OLED screens use layouts like PenTile where neighboring pixels share subpixels.\n\nComment PIXEL for the next one.",
                tags=["science", "howthingswork", "screens", "oled", "iphone", "tech", "learnontiktok", "shipwithpip"],
                pinned="Comment PIXEL and tell me which everyday gadget to break down next.",
                notes="Verified (TechTerms Subpixel): a subpixel is one of the red, green, blue elements of a pixel; all three at full intensity looks white, all off looks black; varying brightness gives millions of colors; some OLEDs add a white subpixel or use PenTile with shared subpixels. Visual notes: the zoom is a computed subpixel render of a small generated image using a typical RGB stripe layout (real phone layouts vary); the PenTile-style layout is illustrative. Red plus green looking yellow is standard additive color mixing.")
    return reel, buf, cfg, post


BUILD = {"micro": build_micro, "qi": build_qi, "gps": build_gps, "pixel": build_pixel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = BUILD[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
