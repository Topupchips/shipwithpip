"""Slow-motion glass shatter reels (stylized 2D physics sim, clearly labeled as a simulation).
usage: python shatterfx.py cup|bottles out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
"""
import argparse
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from scipy.signal import butter, lfilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa: E402
from scenes import CYAN, GREEN, PINK, VIOLET, YEL, ORANGE, Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, INK, SR, VS, clamp, ease_out, font, tone  # noqa: E402

FPS = 30
G = 1500.0
SUB = 4
SLOW = 0.3
BS = 0.72


# ------------------------------------------------------------ geometry
def clip_half(poly, a, b, c):
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if fp <= 0:
            out.append(p)
        if (fp < 0 < fq) or (fq < 0 < fp):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def inside(poly, x, y):
    c = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            c = not c
    return c


def area(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))


def centroid(poly):
    a = area(poly)
    if abs(a) < 1e-6:
        return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
    cx = cy = 0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cr = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    return (cx / (6 * a), cy / (6 * a))


def fracture(poly, impact, rng, n_max=26, dmin0=14, dmin_k=0.16, bias=1.0):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    seeds = []
    tries = 0
    while len(seeds) < n_max and tries < 4000:
        tries += 1
        x, y = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
        if not inside(poly, x, y):
            continue
        r = math.hypot(x - impact[0], y - impact[1])
        dm = dmin0 + dmin_k * r * bias
        if all(math.hypot(x - sx, y - sy) >= dm for sx, sy in seeds):
            seeds.append((x, y))
    cells = []
    for i, (xi, yi) in enumerate(seeds):
        cell = poly
        for j, (xj, yj) in enumerate(seeds):
            if i == j:
                continue
            cell = clip_half(cell, xj - xi, yj - yi, (xj * xj + yj * yj - xi * xi - yi * yi) / 2)
            if len(cell) < 3:
                break
        if len(cell) >= 3 and abs(area(cell)) > 20:
            cells.append(cell)
    return cells


# ------------------------------------------------------------ sim objects
class Shard:
    def __init__(self, poly, release, vel, angvel, floor):
        c = centroid(poly)
        self.local = [(x - c[0], y - c[1]) for x, y in poly]
        self.pos = np.array(c, float)
        self.vel = np.array(vel, float)
        self.ang = 0.0
        self.angvel = angvel
        self.release = release
        self.floor = floor
        self.asleep = False
        self.bounces = 0
        self.ev = []   # (time, speed) bounce events
        self.rad = max(math.hypot(x, y) for x, y in self.local)

    def verts(self):
        ca, sa = math.cos(self.ang), math.sin(self.ang)
        return [(self.pos[0] + x * ca - y * sa, self.pos[1] + x * sa + y * ca) for x, y in self.local]

    def step(self, dt, s):
        if s < self.release or self.asleep:
            return
        self.vel[1] += G * dt
        self.vel *= (1 - 0.15 * dt)
        self.pos += self.vel * dt
        self.ang += self.angvel * dt
        vs = self.verts()
        low = max(v[1] for v in vs)
        if low > self.floor:
            self.pos[1] -= low - self.floor
            sp = abs(self.vel[1])
            if sp > 60:
                self.ev.append((s, sp))
            self.vel[1] = -self.vel[1] * 0.28
            self.vel[0] *= 0.72
            self.angvel *= 0.55
            self.bounces += 1
            # align an edge to the floor
            n = len(vs)
            best = None
            for i in range(n):
                dx, dy = vs[(i + 1) % n][0] - vs[i][0], vs[(i + 1) % n][1] - vs[i][1]
                th = math.atan2(dy, dx)
                d = (-th + math.pi / 2) % math.pi - math.pi / 2
                if best is None or abs(d) < abs(best):
                    best = d
            self.ang += best * 0.35
            if sp < 90 and abs(self.vel[0]) < 40:
                self.asleep = True
                self.vel[:] = 0


class Drop:
    def __init__(self, x, y, r, release, vel):
        self.p = np.array([x, y], float)
        self.v = np.array(vel, float)
        self.r = r
        self.release = release
        self.flat = False
        self.ev = None

    def step(self, dt, s, floor):
        if s < self.release or self.flat:
            return
        self.v[1] += G * dt
        self.v *= (1 - 0.35 * dt)
        self.p += self.v * dt
        if self.p[1] > floor - self.r:
            self.p[1] = floor - self.r
            sp = abs(self.v[1])
            if self.ev is None and sp > 80:
                self.ev = (s, sp)
            if sp < 140 or self.r < 5:
                self.flat = True
            else:
                self.v[1] = -self.v[1] * 0.22
                self.v[0] *= 0.6


def glass_poly(d, poly, alpha=1.0, spark=0.0, edge=(235, 250, 255)):
    d.polygon(poly, fill=(150, 215, 238, int(62 * alpha)))
    d.line(poly + [poly[0]], fill=edge + (int(215 * alpha),), width=2, joint="curve")
    # highlight along the longest edge, nudged inward
    n = len(poly)
    best, bi = 0, 0
    for i in range(n):
        L = math.hypot(poly[(i + 1) % n][0] - poly[i][0], poly[(i + 1) % n][1] - poly[i][1])
        if L > best:
            best, bi = L, i
    a, b = poly[bi], poly[(bi + 1) % n]
    cx, cy = centroid(poly)
    k = 0.18
    a2, b2 = (a[0] + (cx - a[0]) * k, a[1] + (cy - a[1]) * k), (b[0] + (cx - b[0]) * k, b[1] + (cy - b[1]) * k)
    d.line([a2, b2], fill=(255, 255, 255, int(150 * alpha)), width=3)
    if spark > 0.5:
        r = 9 + 10 * (spark - 0.5)
        for dx, dy in ((r, 0), (0, r)):
            d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(255, 255, 255, 230), width=2)


# ------------------------------------------------------------ the world
class World:
    """Holds a precomputed list of per-frame states."""

    def __init__(self, kind, seed=1):
        self.kind = kind
        self.rng = random.Random(seed)
        self.floor = 770.0
        self.shards, self.drops = [], []
        self.fra = {}
        if kind == "cup":
            self.setup_cup()
        else:
            self.setup_bottles()

    # ---------------- cup
    def setup_cup(self):
        cx, top, bot = 560, 420.0, self.floor
        self.cup = dict(cx=cx, top=top, bot=bot, wt=15, base=30, wtop=205, wbot=150)
        c = self.cup

        def edge(y, outer, side):
            t = (y - top) / (bot - top)
            half = (c["wtop"] + (c["wbot"] - c["wtop"]) * t) / 2
            return cx + side * (half if outer else half - c["wt"])
        self.edge = edge
        L = [(edge(top, 1, -1), top), (edge(top, 0, -1), top), (edge(bot - c["base"], 0, -1), bot - c["base"]), (edge(bot, 1, -1), bot)]
        R = [(edge(top, 0, 1), top), (edge(top, 1, 1), top), (edge(bot, 1, 1), bot), (edge(bot - c["base"], 0, 1), bot - c["base"])]
        B = [(edge(bot - c["base"], 0, -1), bot - c["base"]), (edge(bot - c["base"], 0, 1), bot - c["base"]), (edge(bot, 1, 1), bot), (edge(bot, 1, -1), bot)]
        self.intact = [L, R, B]
        self.water_y = 520.0
        self.water_poly = [(edge(self.water_y, 0, -1), self.water_y), (edge(self.water_y, 0, 1), self.water_y), (edge(bot - c["base"], 0, 1), bot - c["base"]), (edge(bot - c["base"], 0, -1), bot - c["base"])]
        self.I = (edge(610, 1, -1), 610.0)
        self.E = (edge(610, 1, 1), 610.0)
        self.v_dir = np.array([1.0, 0.0])
        self.t_hit_x = self.I[0]

    def blast_cup(self):
        rng = self.rng
        I, E = np.array(self.I), np.array(self.E)
        L, R, B = self.intact
        for poly, which in ((L, "L"), (R, "R"), (B, "B")):
            ref = I if which != "R" else E
            cells = fracture(poly, tuple(ref), rng, n_max=30 if which != "B" else 14, dmin0=13, dmin_k=0.17)
            for cell in cells:
                c = np.array(centroid(cell))
                dvec = c - ref
                r = float(np.hypot(*dvec)) + 1e-6
                if which == "R":
                    delay = 0.34 + r / 1500
                    direc = dvec / r * 0.5 + np.array([1.0, -0.15])
                    sp = 950 * math.exp(-r / 150) + 140
                elif which == "L":
                    delay = r / 1500
                    direc = dvec / r + np.array([0.35, 0.0])
                    sp = 1000 * math.exp(-r / 160) + 90
                else:
                    delay = 0.22 + r / 1000
                    direc = dvec / r + np.array([0.3, -0.2])
                    sp = 700 * math.exp(-r / 220) + 60
                direc = direc / (np.hypot(*direc) + 1e-9)
                vel = direc * sp + np.array([rng.uniform(-40, 40), rng.uniform(-120, 20)])
                self.shards.append(Shard(cell, delay, vel, rng.uniform(-9, 9) * (1 + sp / 500), self.floor))
        # water
        wy0 = self.water_y
        n = 0
        while n < 380:
            y = rng.uniform(wy0, self.floor - self.cup["base"])
            x = rng.uniform(self.edge(y, 0, -1), self.edge(y, 0, 1))
            r = rng.uniform(6.5, 11.5)
            dvec = np.array([x, y]) - I
            d = float(np.hypot(*dvec)) + 1
            delay = 0.03 + d / 900 + (0.02 if x > self.cup["cx"] else 0)
            push = np.array([1.0, 0.0]) * 520 * math.exp(-d / 170)
            side = np.array([(x - self.cup["cx"]) * 2.2, rng.uniform(-80, 30)])
            vel = push + side + np.array([rng.uniform(-60, 60), rng.uniform(-140, 0)])
            self.drops.append(Drop(x, y, r, delay, vel))
            n += 1

    # ---------------- bottles
    def bottle_poly(self, sign):
        pts = [(-210, -60), (30, -60), (60, -56), (85, -44), (105, -28), (118, -19), (205, -17), (213, -22), (213, 22), (205, 17), (118, 19), (105, 28), (85, 44), (60, 56), (30, 60), (-210, 60)]
        # bottle with neck pointing +x, flip for sign=-1
        return [(sign * x * BS, y * BS) for x, y in pts] if sign > 0 else [(-x * BS, y * BS) for x, y in pts][::-1]

    def liquid_poly(self, sign):
        pts = [(-200, -50), (30, -50), (55, -46), (75, -37), (92, -22), (92, 22), (75, 37), (55, 46), (30, 50), (-200, 50)]
        return [(sign * x * BS, y * BS) for x, y in pts] if sign > 0 else [(-x * BS, y * BS) for x, y in pts][::-1]

    def setup_bottles(self):
        self.floor = 800.0
        self.yc = 460.0
        self.cx0 = 480.0
        self.contact_gap = 213.0 * BS

    def bottle_x(self, which, u):
        end = 480 - self.contact_gap if which == 0 else 480 + self.contact_gap
        start = end - 330 if which == 0 else end + 330
        return start + (end - start) * (u ** 1.5)

    def blast_bottles(self):
        rng = self.rng
        I = np.array([480.0, self.yc])
        cols = [(255, 170, 40), (120, 230, 120)]
        for which in (0, 1):
            sign = 1 if which == 0 else -1
            cx = 480 - self.contact_gap if which == 0 else 480 + self.contact_gap
            poly = [(cx + x, self.yc + y) for x, y in self.bottle_poly(sign)]
            cells = fracture(poly, tuple(I), rng, n_max=34, dmin0=14, dmin_k=0.13)
            v_in = np.array([300.0 * sign, 0.0])
            for cell in cells:
                c = np.array(centroid(cell))
                dvec = c - I
                r = float(np.hypot(*dvec)) + 1e-6
                direc = dvec / r
                sp = 430 * math.exp(-r / 200) + 80
                vel = direc * sp * 0.85 + v_in * 0.25 + np.array([rng.uniform(-50, 50), rng.uniform(-200, -20)])
                self.shards.append(Shard(cell, r / 1700, vel, rng.uniform(-8, 8) * (1 + sp / 500), self.floor))
            lp = [(cx + x, self.yc + y) for x, y in self.liquid_poly(sign)]
            xs, ys = [p[0] for p in lp], [p[1] for p in lp]
            n = 0
            while n < 210:
                x, y = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
                if not inside(lp, x, y):
                    continue
                r = rng.uniform(6.5, 11)
                dvec = np.array([x, y]) - I
                d = float(np.hypot(*dvec)) + 1
                vel = dvec / d * (360 * math.exp(-d / 220) + 70) + v_in * 0.3 + np.array([rng.uniform(-60, 60), rng.uniform(-220, -10)])
                d_ = Drop(x, y, r, 0.03 + d / 1600, vel)
                d_.col = cols[which]
                self.drops.append(d_)
                n += 1

    # ---------------- simulate
    def run(self, seconds):
        if self.kind == "cup":
            self.blast_cup()
        else:
            self.blast_bottles()
        frames = []
        n = int(seconds * FPS)
        dt = SLOW / FPS / SUB
        s = 0.0
        for f in range(n):
            snap_s = [(sh.verts(), sh.release <= s) for sh in self.shards]
            snap_d = [(d.p.copy(), d.r, d.flat, d.release <= s, getattr(d, "col", None)) for d in self.drops]
            frames.append((snap_s, snap_d))
            for _ in range(SUB):
                s += dt
                for sh in self.shards:
                    sh.step(dt, s)
                for d in self.drops:
                    d.step(dt, s, self.floor)
        return frames


# ------------------------------------------------------------ reel
class ShatterReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.W = cfg["world"]
        self.frames = cfg["frames"]
        self.t_hit = cfg["t_hit"]
        # static background
        bg = Image.new("RGB", (VS, VS), (8, 6, 22))
        arr = np.array(bg, float)
        yy, xx = np.mgrid[0:VS, 0:VS]
        cone = np.clip(1 - np.hypot((xx - VS * 0.5) / (VS * 0.62), (yy - 80) / (VS * 0.95)), 0, 1) ** 1.6
        arr += cone[..., None] * np.array([70, 62, 100])
        floor = self.W.floor
        arr[int(floor):] = arr[int(floor):] * 0.55 + np.array([14, 10, 30])
        arr[int(floor):int(floor) + 3] += 90
        vig = np.clip(np.hypot((xx - VS / 2) / (VS * 0.75), (yy - VS / 2) / (VS * 0.75)), 0, 1) ** 2
        arr *= (1 - 0.5 * vig)[..., None]
        self.sbg = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    def sim_idx(self, t):
        return int((t - self.t_hit) * FPS)

    def shake(self, t):
        u = t - self.t_hit
        if 0 <= u < 0.5:
            k = (1 - u / 0.5) ** 2 * 16
            return math.sin(u * 90) * k, math.cos(u * 77) * k * 0.6
        return 0.0, 0.0

    def drops_layer(self, items, ox, oy):
        S = 2
        L = Image.new("L", (VS // S, VS // S), 0)
        d = ImageDraw.Draw(L)
        flats = []
        for (p, r, flat, rel, col) in items:
            if not rel:
                continue
            x, y = (p[0] + ox) / S, (p[1] + oy) / S
            if flat:
                d.ellipse([x - r * 1.35 / S, y - r * 0.45 / S, x + r * 1.35 / S, y + r * 0.45 / S], fill=255)
            else:
                d.ellipse([x - r / S, y - r / S, x + r / S, y + r / S], fill=255)
        return L

    def content(self, t):
        W = self.W
        ox, oy = self.shake(t)
        im = self.sbg.copy()
        ov = Image.new("RGBA", (VS, VS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov, "RGBA")
        i = self.sim_idx(t)
        P = lambda pts: [(x + ox, y + oy) for x, y in pts]  # noqa: E731
        water_items = []
        wcol = (80, 170, 235)
        if W.kind == "cup":
            c = W.cup
            # shadow
            d.ellipse([c["cx"] - 150 + ox, W.floor - 14 + oy, c["cx"] + 150 + ox, W.floor + 18 + oy], fill=(0, 0, 0, 120))
            if i < 0:
                wp = W.water_poly
                wob = 3 * math.sin(t * 5)
                d.polygon(P([(wp[0][0], wp[0][1] + wob), (wp[1][0], wp[1][1] - wob)] + wp[2:]), fill=wcol + (150,))
                d.line(P([(wp[0][0], wp[0][1] + wob), (wp[1][0], wp[1][1] - wob)]), fill=(210, 240, 255, 230), width=3)
                for poly in W.intact:
                    glass_poly(d, P(poly), 1.0, 0.0)
            # bullet + approach
            bx = self.bullet_x(t)
            if bx is not None:
                by = 610.0
                tail = 130
                for k in range(14):
                    a = int(190 * (1 - k / 14) ** 2)
                    d.ellipse([bx - tail * k / 14 - 7 + ox, by - 5 + oy - k * 0.1, bx - tail * k / 14 + 7 + ox, by + 5 + oy], fill=(255, 190, 90, a))
                d.rounded_rectangle([bx - 20 + ox, by - 8 + oy, bx + 14 + ox, by + 8 + oy], radius=7, fill=(236, 186, 80, 255), outline=(255, 245, 200, 255), width=2)
                d.polygon([(bx + 14 + ox, by - 8 + oy), (bx + 32 + ox, by + oy), (bx + 14 + ox, by + 8 + oy)], fill=(210, 140, 60, 255))
                if i < 0:
                    ph = (t * 3) % 1
                    for k in range(2):
                        rr = 30 + 60 * ((ph + k * 0.5) % 1)
                        d.ellipse([bx + 40 - rr + ox, by - rr + oy, bx + 40 + rr + ox, by + rr + oy], outline=(255, 220, 140, int(120 * (1 - ((ph + k * 0.5) % 1)))), width=2)
        else:
            u = clamp((t - self.cfg["t0"]) / (self.t_hit - self.cfg["t0"]))
            if i < 0:
                for which in (0, 1):
                    sign = 1 if which == 0 else -1
                    x = W.bottle_x(which, u)
                    poly = [(x + px, W.yc + py) for px, py in W.bottle_poly(sign)]
                    lp = [(x + px, W.yc + py) for px, py in W.liquid_poly(sign)]
                    col = (255, 170, 40) if which == 0 else (120, 230, 120)
                    d.polygon(P(lp), fill=col + (170,))
                    glass_poly(d, P(poly), 1.0, 0.0)
                    d.line(P([(x - sign * 110, W.yc - 29), (x + sign * 15, W.yc - 29)]), fill=(255, 255, 255, 160), width=4)
        # shake zoom flash
        if 0 <= t - self.t_hit < 0.12:
            q = 1 - (t - self.t_hit) / 0.12
            fx, fy = (W.I if W.kind == "cup" else (480, W.yc))
            r = 60 + 190 * (1 - q)
            d.ellipse([fx - r + ox, fy - r + oy, fx + r + ox, fy + r + oy], fill=(255, 250, 230, int(220 * q)))
        if i >= 0:
            i = min(i, len(self.frames) - 1)
            snap_s, snap_d = self.frames[i]
            # water metaballs
            L = self.drops_layer(snap_d, ox, oy)
            Lb = L.filter(ImageFilter.GaussianBlur(5))
            m = Lb.point(lambda v: 0 if v < 70 else (255 if v > 130 else int((v - 70) / 60 * 255)))
            m = m.resize((VS, VS), Image.BILINEAR)
            if W.kind == "cup":
                col = Image.new("RGBA", (VS, VS), wcol + (165,))
            else:
                col = Image.new("RGBA", (VS, VS), (255, 190, 60, 165))
            wl = Image.new("RGBA", (VS, VS), (0, 0, 0, 0))
            wl.paste(col, (0, 0), m)
            if W.kind == "bottles":
                # tint by side: recolor the right half green
                gl = Image.new("RGBA", (VS, VS), (120, 230, 120, 165))
                half = Image.new("L", (VS, VS), 0)
                ImageDraw.Draw(half).rectangle([int(480 + ox), 0, VS, VS], fill=255)
                wl.paste(gl, (0, 0), ImageChops.multiply(m, half))
            hi = Lb.point(lambda v: 0 if v < 118 else (255 if v > 150 else int((v - 118) / 32 * 255))).resize((VS, VS), Image.BILINEAR)
            hi = ImageChops.offset(hi, 0, 0)
            inner = hi.filter(ImageFilter.GaussianBlur(4))
            rim = ImageChops.subtract(m, inner.point(lambda v: min(255, int(v * 1.15))))
            wl.paste(Image.new("RGBA", (VS, VS), (225, 245, 255, 190)), (0, 0), rim.point(lambda v: int(v * 0.8)))
            ov = Image.alpha_composite(ov, wl)
            d = ImageDraw.Draw(ov, "RGBA")
            for k, (verts, rel) in enumerate(snap_s):
                sp = (math.sin(t * 7 + k * 1.7) + 1) / 2
                glass_poly(d, P(verts), 1.0, sp if sp > 0.9 else 0.0)
        im = Image.alpha_composite(im, ov)
        cx0, cy0, cs = self.cfg["crop"]
        return im.convert("RGB").crop((cx0, cy0, cx0 + cs, cy0 + cs)).resize((VS, VS), Image.LANCZOS)

    def bullet_x(self, t):
        c = self.cfg
        if t < c["t0"]:
            return None
        u = (t - c["t0"]) / (self.t_hit - c["t0"])
        if u <= 1.0:
            return 190 + (self.W.I[0] - 40 - 190) * (u ** 1.6)
        # after hit, bullet keeps going to the right (slow-mo)
        return self.W.I[0] - 40 + (t - self.t_hit) * 360

    def stats(self, t):
        i = self.sim_idx(t)
        pre = i < 0
        frac = clamp((t - self.cfg["t0"]) / (self.t_hit - self.cfg["t0"])) if pre else 1.0
        lab = "WAIT FOR IT..." if pre else "SLOW MOTION"
        return [("PHYSICS SIM", YEL), ("SLOW-MO", CYAN)], frac, lab, ORANGE if pre else GREEN


# ------------------------------------------------------------ audio
def noise(n, rng):
    return rng.standard_normal(n).astype(np.float32)


def bp(x, lo, hi):
    b, a = butter(2, [lo / (SR / 2), min(hi, SR / 2 - 100) / (SR / 2)], btype="band")
    return lfilter(b, a, x).astype(np.float32)


def add(buf, t0, x):
    i = int(t0 * SR)
    if i < 0 or i >= len(buf):
        return
    buf[i:i + len(x)] += x[: len(buf) - i]


def burst(rng, dur, lo, hi, amp, decay):
    n = int(dur * SR)
    x = bp(noise(n, rng), lo, hi)
    env = np.exp(-np.arange(n) / SR * decay)
    return x * env * amp


def sfx_track(buf, cfg, W, frames):
    rng = np.random.default_rng(5)
    t_hit = cfg["t_hit"]
    t0 = cfg["t0"]
    # riser / whoosh before the hit
    n = int((t_hit - t0 + 0.2) * SR)
    ramp = np.linspace(0, 1, n) ** 2
    x = bp(noise(n, rng), 300, 3000) * ramp * 0.07
    tt = np.arange(n) / SR
    x += (np.sin(2 * np.pi * (90 + 160 * ramp) * tt) * ramp * 0.06).astype(np.float32)
    add(buf, t0 - 0.1, x)
    # heartbeat-ish ticks building tension
    k = 0
    tk = t0
    while tk < t_hit - 0.3:
        add(buf, tk, (np.sin(2 * np.pi * 70 * np.arange(int(0.12 * SR)) / SR) * np.exp(-np.arange(int(0.12 * SR)) / SR * 30) * 0.12).astype(np.float32))
        tk += max(0.28, 0.8 - 0.1 * (tk - t0))
    # impact: crack + thump + crash
    add(buf, t_hit, burst(rng, 0.25, 2500, 11000, 0.5, 22))
    thump = (np.sin(2 * np.pi * 55 * np.arange(int(0.6 * SR)) / SR) * np.exp(-np.arange(int(0.6 * SR)) / SR * 7) * 0.5).astype(np.float32)
    add(buf, t_hit, thump)
    add(buf, t_hit + 0.02, burst(rng, 1.2, 1500, 9000, 0.30, 4.5))
    # tinkles from shard bounces
    ev = []
    for sh in W.shards:
        for (s, sp) in sh.ev[:3]:
            ev.append((s, sp))
    ev.sort()
    last = -1
    for s, sp in ev:
        if s - last < 0.012:
            continue
        last = s
        f = rng.uniform(2200, 7500)
        dur = 0.12
        n = int(dur * SR)
        tt = np.arange(n) / SR
        x = (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 40) * min(0.12, sp / 9000)).astype(np.float32)
        add(buf, t_hit + s / SLOW, x)
    drops = sorted(d.ev[0] for d in W.drops if d.ev)
    last = -1
    for s in drops[::3]:
        if s - last < 0.03:
            continue
        last = s
        add(buf, t_hit + s / SLOW, burst(rng, 0.08, 800, 5000, 0.05, 55))


# ------------------------------------------------------------ builders
def build(which, model):
    W = World("cup" if which == "cup" else "bottles", seed=4)
    SIM_T = 9.5
    frames = W.run(SIM_T)
    if which == "cup":
        T, t0, t_hit = 27.0, 2.0, 6.4
    else:
        T, t0, t_hit = 27.0, 2.0, 7.0
    nshards = len(W.shards)
    print("shards", nshards, "drops", len(W.drops), "bounce events", sum(len(s.ev) for s in W.shards))
    cfg = dict(T=T, t0=t0, t_hit=t_hit, world=W, frames=frames, c1=ORANGE if which == "cup" else YEL, c2=PINK, payoff=None, seed=61 if which == "cup" else 67,
               still_times=[0.5, 3.5, 5.8, t_hit + 0.15, t_hit + 0.6, t_hit + 1.5, t_hit + 3.5, t_hit + 8], crop=(250, 330, 640) if which == "cup" else (130, 130, 700))
    if which == "cup":
        t_fact = 11.4
        cfg["header"] = lambda t: (([("WATCH THIS GLASS.", (255, 255, 255)), ("DON'T BLINK.", YEL)], "WAIT FOR IT") if t < t_fact else
                                   ([("CRACKS TRAVEL", (255, 255, 255)), ("~1,500 M/S", PINK)], "MEASURED IN TEMPERED GLASS"))
        lines = [(0.2, "Watch this glass. Don't look away."),
                 (3.1, "This whole scene is a physics simulation I coded."),
                 (t_fact, "In real glass, cracks race at about fifteen hundred meters per second."),
                 (t_fact + 5.0, "So fast, your eyes see it as one instant."),
                 (t_fact + 8.2, "Comment GLASS and I'll send the prompt so you can build it."),
                 (t_fact + 12.6, "Follow for more.")]
        title = "SHATTER-1  Bullet vs glass of water (slow-mo physics sim)"
        post_caption = ("Wait for it... a slow-motion bullet hits a glass of water. This is a physics simulation I coded with AI, not real footage. In real glass, cracks travel at roughly 1,500 m/s (measured in tempered glass with high-speed cameras), which is why it looks like it shatters in one instant.\n\n"
                        "Building your own app? Use code AT100 on Base44 (link in bio).\n\nComment GLASS and I'll send the prompt so you can build it.")
        pinned = ("Build me a single-file HTML canvas page: a side-view glass of water on a table. A bullet crawls in slow motion from the left, hits the left wall, and the glass shatters. Split the glass walls into Voronoi shards (smaller near the impact), release them with a delay that spreads out from the impact point, give each shard velocity, spin, gravity and a floor bounce, and turn the water into ~400 metaball droplets that splash. Add screen shake and a white flash on impact.")
        notes = f"Stylized 2D simulation, NOT real footage (labeled 'PHYSICS SIM' on screen and in the voiceover). {nshards} shards, {len(W.drops)} water droplets.\nFact: cracks in tempered glass ~1,458 m/s (Slow Mo Guys high-speed video, ~78,600 fps) and ~1,536 m/s in a separate study. Source: https://ceramics.org/ceramic-tech-today/video-speed-of-cracks-in-glass/\n'Your eyes see it as one instant' is a plain-language statement, not a measured claim."
        sound_name = "GLASS"
    else:
        N = nshards
        t_guess = t_hit - 3.0
        t_after = t_hit + 4.4
        cfg["header"] = lambda t: (([("2 GLASS BOTTLES.", (255, 255, 255)), ("1 COLLISION.", YEL)], "GUESS THE PIECES") if t < t_after else
                                   ([(f"{N} PIECES.", (255, 255, 255)), ("DID YOU GUESS?", PINK)], "IN THIS SIMULATION"))
        lines = [(0.2, "Two glass bottles. One collision."),
                 (2.8, "How many pieces will it make? Guess in the comments."),
                 (t_after, f"In this simulation, the two bottles broke into {N} pieces."),
                 (t_after + 4.0, "Glass can't bend, so the energy goes into cracks and flying pieces."),
                 (t_after + 8.2, "Comment BOTTLE and I'll send the prompt to build it."),
                 (t_after + 12.0, "Follow for more.")]
        title = "SHATTER-2  Two glass bottles collide (slow-mo physics sim)"
        post_caption = (f"Guess how many pieces two glass bottles break into when they collide. In this physics simulation I coded with AI (not real footage): {N} pieces. Did you guess right?\n\n"
                        "Building your own app? Use code AT100 on Base44 (link in bio).\n\nComment your guess, and comment BOTTLE if you want the prompt to build it.")
        pinned = "Guess the number of pieces BEFORE you watch the end! Comment your number. Closest guess gets a shoutout. Want the prompt to build this? Comment BOTTLE."
        notes = f"Stylized 2D simulation, NOT real footage. {N} shards total (both bottles) and {len(W.drops)} liquid droplets, generated by Voronoi fracture around the contact point with a delay that spreads outward. The piece count belongs to this run only and is not a prediction for real bottles."
        sound_name = "BOTTLE"
    buf, vo = make_audio(cfg, lines, model)
    sfx_track(buf, cfg, W, frames)
    chime(buf, cfg["t_hit"] + 0.1) if False else None
    reel = ShatterReel(cfg, vo)
    post = dict(title=title, caption=post_caption, tags=["slowmotion", "satisfying", "physics", "coding", "ai", "shipwithpip"], pinned=pinned, notes=notes)
    return reel, buf, cfg, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = build(a.which, a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
