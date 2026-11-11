"""Four real-simulation explainer reels with a richer look (aurora bg, glass card, bloom, confetti).

usage: python scenes.py gd|km|ql|of out.mp4 --model <kokoro-multi-lang-v1_0 dir> [--stills out.png]
"""
import argparse
import math
import os
import random
import re
import subprocess
import sys

import numpy as np
import soundfile as sf
from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sim"))
import voice  # noqa: E402
from make_reel import music_bed  # noqa: E402
from simfx import (F_BOLD, F_MONO, F_SEMI, FPS, H, INK, SR, VS, VX, VY, W, clamp, ease_out, font, smooth, tone, tw)  # noqa: E402

PINK, CYAN, GREEN, YEL, ORANGE, VIOLET = (255, 76, 170), (60, 222, 255), (70, 255, 170), (255, 226, 96), (255, 140, 60), (150, 110, 255)


# ---------------------------------------------------------------- shared look
class Reel2:
    def __init__(self, cfg, vo):
        self.cfg, self.vo, self.T = cfg, vo, cfg["T"]
        c1, c2 = cfg["c1"], cfg["c2"]
        h, w = H + 240, W + 160
        yy = np.linspace(0, 1, h)[:, None, None]
        img = np.broadcast_to(np.array([18, 10, 46]) * (1 - yy) + np.array([5, 4, 18]) * yy, (h, w, 3)).copy().astype(float)
        Y, X = np.mgrid[0:h, 0:w]
        for cx, cy, r, col, a in ((0.15 * w, 0.22 * h, 0.62 * w, c1, 0.55), (0.9 * w, 0.52 * h, 0.7 * w, c2, 0.5), (0.35 * w, 0.97 * h, 0.8 * w, c1, 0.4)):
            m = np.clip(1 - np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / r, 0, 1) ** 2 * a
            img += (np.array(col, float)[None, None, :] * 0.55 - img) * m[..., None]
        rs = np.random.default_rng(4)
        for _ in range(180):
            x, y = rs.integers(0, w), rs.integers(0, h)
            img[y:y + 2, x:x + 2] += rs.integers(40, 120)
        self.bg = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
        # glow + gradient ring + rounded mask
        glow = Image.new("L", (W, H), 0)
        ImageDraw.Draw(glow).rounded_rectangle([VX - 4, VY - 4, VX + VS + 4, VY + VS + 4], radius=48, fill=150)
        self.glow_a = glow.filter(ImageFilter.GaussianBlur(34))
        self.glow_rgb = Image.new("RGB", (W, H), tuple(c1))
        n = VS + 16
        gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        t = (gx + gy) / 2
        ring = (np.array(c1, float)[None, None, :] * (1 - t[..., None]) + np.array(c2, float)[None, None, :] * t[..., None]).astype(np.uint8)
        self.ring_rgb = Image.fromarray(ring, "RGB")
        rm = Image.new("L", (n, n), 0)
        d = ImageDraw.Draw(rm)
        d.rounded_rectangle([0, 0, n - 1, n - 1], radius=46, fill=255)
        d.rounded_rectangle([7, 7, n - 8, n - 8], radius=40, fill=0)
        self.ring_a = rm
        self.cmask = Image.new("L", (VS, VS), 0)
        ImageDraw.Draw(self.cmask).rounded_rectangle([0, 0, VS - 1, VS - 1], radius=40, fill=255)
        r = random.Random(3)
        cols = [c1, c2, YEL, GREEN, (255, 255, 255)]
        self.conf = [dict(a=r.uniform(-math.pi, 0), s=r.uniform(260, 1100), c=r.choice(cols), w=r.uniform(8, 18), rot=r.uniform(0, 6), dl=r.uniform(0, 0.25)) for _ in range(140)]

    # -- pieces
    def header(self, d, t):
        lines, tag = self.cfg["header"](t)
        pulse = 0.6 + 0.4 * math.sin(t * 6)
        f = font(F_SEMI, 34)
        w = tw(d, tag, f) + 96
        d.rounded_rectangle([(W - w) / 2, 70, (W + w) / 2, 136], radius=33, fill=YEL + (255,))
        d.ellipse([(W - w) / 2 + 24, 94, (W - w) / 2 + 46, 116], fill=(255, 60, 90, int(255 * pulse)))
        d.text((W / 2 + 14, 103), tag, font=f, fill=INK, anchor="mm")
        for i, (txt, col) in enumerate(lines):
            y = 238 + i * 122
            d.text((W / 2, y + 7), txt, font=font(F_BOLD, 104), fill=(0, 0, 0, 120), anchor="mm", stroke_width=9, stroke_fill=(0, 0, 0, 120))
            d.text((W / 2, y), txt, font=font(F_BOLD, 104), fill=col + (255,), anchor="mm", stroke_width=8, stroke_fill=INK + (255,))

    def pills(self, d, items):
        f = font(F_MONO, 38)
        total = sum(tw(d, s, f) + 52 + 16 for s, _ in items) - 16
        x = (W - total) / 2
        for txt, col in items:
            w = tw(d, txt, f) + 52
            d.rounded_rectangle([x, 1412, x + w, 1486], radius=37, fill=(14, 8, 38, 200), outline=col + (255,), width=4)
            d.text((x + w / 2, 1449), txt, font=f, fill=(255, 255, 255, 255), anchor="mm")
            x += w + 16

    def bar(self, d, frac, label, col):
        d.rounded_rectangle([VX, 1516, VX + VS, 1558], radius=21, fill=(14, 8, 38, 200), outline=(255, 255, 255, 50), width=2)
        if frac > 0.01:
            d.rounded_rectangle([VX + 3, 1519, VX + 3 + max(36, (VS - 6) * clamp(frac)), 1555], radius=18, fill=col + (255,))
        d.text((VX + VS / 2, 1537), label, font=font(F_MONO, 24), fill=(255, 255, 255, 255) if frac < 0.55 else INK + (255,), anchor="mm")

    def caption(self, d, t):
        for st, en, text in self.vo:
            if st <= t <= en + 0.2:
                words = text.split()
                wts = np.array([len(w) + 2 for w in words], float)
                cum = np.cumsum(wts) / wts.sum() * (en - st)
                k = min(int(np.searchsorted(cum, t - st)), len(words) - 1)
                g0 = (k // 4) * 4
                chunk = words[g0:g0 + 4]
                size = 66
                while True:
                    f = font(F_BOLD, size)
                    ws = [tw(d, w, f) + 24 for w in chunk]
                    if sum(ws) <= W - 110 or size <= 44:
                        break
                    size -= 4
                x = (W - sum(ws)) / 2
                d.rounded_rectangle([x - 26, 1612, x + sum(ws) + 2, 1708], radius=48, fill=(10, 6, 28, 190))
                for i, w in enumerate(chunk):
                    cur = g0 + i == k
                    d.text((x + 6, 1660), w, font=f, fill=(YEL if cur else (255, 255, 255)) + (255,), anchor="lm")
                    x += ws[i]
                return

    def cta(self, d, t):
        if t > self.T - 4.2:
            q = ease_out((t - (self.T - 4.2)) / 0.4)
            f = font(F_BOLD, 52)
            txt = "@shipwithpip"
            w = tw(d, txt, f) + 100
            y = 1760 - (1 - q) * 30
            d.rounded_rectangle([(W - w) / 2, y, (W + w) / 2, y + 92], radius=46, fill=PINK + (int(255 * q),))
            d.text((W / 2, y + 46), txt, font=f, fill=(255, 255, 255, int(255 * q)), anchor="mm")

    def confetti(self, d, t):
        tp = self.cfg.get("payoff")
        if tp is None or t < tp:
            return
        tau = t - tp
        cx, cy = W / 2, 900
        for p in self.conf:
            q = tau - p["dl"]
            if q < 0 or q > 2.6:
                continue
            x = cx + math.cos(p["a"]) * p["s"] * q * 0.8
            y = cy + math.sin(p["a"]) * p["s"] * q * 0.8 + 620 * q * q
            a = int(255 * clamp(1.6 - q / 1.6))
            r = p["rot"] + q * 9
            w = p["w"]
            pts = [(x + math.cos(r + k * 1.57) * w, y + math.sin(r + k * 1.57) * w * 0.55) for k in range(4)]
            d.polygon(pts, fill=p["c"] + (a,))

    def frame(self, t):
        ox, oy = int(80 + 60 * math.sin(t * 0.25)), int(120 + 90 * math.cos(t * 0.18))
        img = self.bg.crop((ox, oy, ox + W, oy + H))
        img.paste(self.glow_rgb, (0, 0), self.glow_a)
        img.paste(self.ring_rgb, (VX - 8, VY - 8), self.ring_a)
        content = self.content(t)
        bl = content.resize((240, 240), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5)).resize((VS, VS), Image.BICUBIC)
        content = ImageChops.screen(content, bl.point(lambda v: int(v * 0.6)))
        img.paste(content, (VX, VY), self.cmask)
        d = ImageDraw.Draw(img, "RGBA")
        self.header(d, t)
        pills, frac, label, col = self.stats(t)
        self.pills(d, pills)
        self.bar(d, frac, label, col)
        self.caption(d, t)
        self.cta(d, t)
        self.confetti(d, t)
        d.rectangle([0, 0, W * clamp(t / self.T), 10], fill=YEL + (255,))
        return img


def cmap(u, stops):
    u = np.clip(u, 0, 1)
    xs = [s[0] for s in stops]
    return np.stack([np.interp(u, xs, [s[1][c] for s in stops]) for c in range(3)], -1)


def glow_dot(d, x, y, r, col, a=255):
    for k, (m, al) in enumerate(((3.2, 40), (2.2, 70), (1.5, 120))):
        d.ellipse([x - r * m, y - r * m, x + r * m, y + r * m], fill=col + (int(al * a / 255),))
    d.ellipse([x - r, y - r, x + r, y + r], fill=col + (a,), outline=(255, 255, 255, a), width=3)


# ---------------------------------------------------------------- 1. gradient descent
def f_land(x, y):
    return -1.6 * np.exp(-((x - 1.1) ** 2 + (y + 0.9) ** 2) / 0.7) - 0.9 * np.exp(-((x + 1.2) ** 2 + (y - 1.0) ** 2) / 0.5) + 0.1 * (x ** 2 + y ** 2) + 0.08 * np.sin(2.5 * x) * np.sin(2.5 * y)


class GDReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        lin = np.linspace(-2.5, 2.5, VS)
        X, Y = np.meshgrid(lin, lin[::-1])
        Z = f_land(X, Y)
        self.zmin, self.zmax = Z.min(), Z.max()
        v = (Z - self.zmin) / (self.zmax - self.zmin)
        col = cmap(1 - v, [(0, (12, 7, 38)), (0.3, (58, 22, 120)), (0.55, (160, 40, 150)), (0.78, (255, 110, 90)), (1, (255, 232, 160))])
        band = np.abs(((v * 18) % 1) - 0.5) < 0.04
        col = np.where(band[..., None], col * 0.55 + 105, col)
        self.base = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB")
        self.balls = []
        for lr, c, nm in ((0.02, CYAN, "TOO SMALL"), (0.2, GREEN, "JUST RIGHT"), (1.3, PINK, "TOO BIG")):
            p = np.array([-1.0, -2.0])
            tr = [p.copy()]
            for _ in range(70):
                h = 1e-4
                g = np.array([(f_land(p[0] + h, p[1]) - f_land(p[0] - h, p[1])) / (2 * h), (f_land(p[0], p[1] + h) - f_land(p[0], p[1] - h)) / (2 * h)])
                p = np.clip(p - lr * g, -2.5, 2.5)
                tr.append(p.copy())
            self.balls.append(dict(lr=lr, c=c, nm=nm, tr=np.array(tr)))

    def px(self, p):
        return ((p[0] + 2.5) / 5 * VS, (2.5 - p[1]) / 5 * VS)

    def pos(self, b, t):
        s = clamp((t - self.cfg["t0"]) * self.cfg["rate"], 0, 70)
        i = min(int(s), 69)
        return b["tr"][i] * (1 - (s - i)) + b["tr"][i + 1] * (s - i), s

    def content(self, t):
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        gx, gy = self.px((1.1, -0.9))
        pulse = 0.5 + 0.5 * math.sin(t * 4)
        d.ellipse([gx - 44 - pulse * 6, gy - 44 - pulse * 6, gx + 44 + pulse * 6, gy + 44 + pulse * 6], outline=(255, 255, 255, 150), width=3)
        d.text((gx, gy + 70), "BOTTOM", font=font(F_MONO, 28), fill=(255, 255, 255, 230), anchor="mm", stroke_width=4, stroke_fill=INK)
        for b in self.balls:
            p, s = self.pos(b, t)
            pts = [self.px(b["tr"][k]) for k in range(int(s) + 1)] + [self.px(p)]
            big = b["lr"] > 1
            if len(pts) > 1:
                d.line(pts[-60:] if big else pts, fill=b["c"] + ((70 if big else 150),), width=(3 if big else 7), joint="curve")
            x, y = self.px(p)
            glow_dot(d, x, y, 15, b["c"])
            ox, oy = {"TOO SMALL": (-10, 46), "JUST RIGHT": (96, -6), "TOO BIG": (0, -46)}[b["nm"]]
            d.text((x + ox, y + oy), b["nm"], font=font(F_MONO, 24), fill=b["c"] + (255,), anchor="mm", stroke_width=5, stroke_fill=INK)
        if t > self.cfg["payoff"] and t < self.cfg["payoff"] + 3.5:
            b = self.balls[1]
            x, y = self.px(self.pos(b, t)[0])
            q = (t - self.cfg["payoff"]) / 1.2
            for k in range(3):
                rr = 30 + 160 * clamp(q - k * 0.18)
                d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=GREEN + (int(220 * (1 - clamp(q - k * 0.18))),), width=5)
        return im

    def stats(self, t):
        items = []
        best = -9
        for b in self.balls:
            p, _ = self.pos(b, t)
            z = float(f_land(p[0], p[1]))
            best = max(best, -z)
            items.append((f"lr {b['lr']:g}", b["c"]))
        frac = (best - (-self.zmax)) / (-self.zmin + self.zmax)
        return items, frac, f"BEST LOSS {-best:0.2f}", GREEN


# ---------------------------------------------------------------- 2. k-means
KCOL = [PINK, CYAN, YEL, GREEN]


class KMReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        r = np.random.default_rng(8)
        ctr = np.array([[0.24, 0.3], [0.74, 0.24], [0.3, 0.74], [0.76, 0.7]])
        self.P = np.concatenate([ctr[i] + r.normal(0, 0.065, (80, 2)) for i in range(4)])
        c = np.array([[0.5, 0.52], [0.56, 0.44], [0.42, 0.6], [0.6, 0.6]])
        self.its = []
        prev = None
        for _ in range(14):
            dist = np.linalg.norm(self.P[:, None, :] - c[None], axis=2)
            a = dist.argmin(1)
            new = np.array([self.P[a == k].mean(0) if (a == k).any() else c[k] for k in range(4)])
            inertia = float(((self.P - c[a]) ** 2).sum())
            self.its.append(dict(c0=c.copy(), a=a, c1=new, inertia=inertia))
            if prev is not None and np.allclose(new, c, atol=1e-4):
                break
            prev, c = a, new
        self.n = len(self.its)
        self.t0, self.D = cfg["t0"], cfg["D"]
        self.t1 = self.t0 + self.n * self.D

    def px(self, p):
        return (50 + p[0] * (VS - 100), 50 + p[1] * (VS - 100))

    def state(self, t):
        k = int(clamp((t - self.t0) / self.D, 0, self.n - 1e-6)) if t >= self.t0 else -1
        return k

    def content(self, t):
        im = Image.new("RGB", (VS, VS), (14, 9, 40))
        d = ImageDraw.Draw(im, "RGBA")
        for g in range(1, 10):
            d.line([(g * 96, 0), (g * 96, VS)], fill=(255, 255, 255, 9), width=2)
            d.line([(0, g * 96), (VS, g * 96)], fill=(255, 255, 255, 9), width=2)
        k = self.state(t)
        t_in = (t - self.t0) - max(k, 0) * self.D
        it = self.its[max(k, 0)]
        phase_a = clamp(t_in / (0.4 * self.D)) if k >= 0 else 0
        mv = smooth((t_in - 0.45 * self.D) / (0.45 * self.D)) if k >= 0 else 0
        cpos = it["c0"] * (1 - mv) + it["c1"] * mv if k >= 0 else self.its[0]["c0"]
        prev_a = self.its[k - 1]["a"] if k >= 1 else None
        for i, p in enumerate(self.P):
            x, y = self.px(p)
            if k < 0:
                d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(205, 205, 235, 235))
                continue
            a = it["a"][i]
            changed = prev_a is None or prev_a[i] != a
            col = KCOL[a]
            if phase_a < 1 and (prev_a is None or changed):
                u = phase_a
                base = (205, 205, 235) if prev_a is None else KCOL[prev_a[i]]
                col = tuple(int(base[c] * (1 - u) + KCOL[a][c] * u) for c in range(3))
            cx, cy = self.px(cpos[a])
            d.line([(x, y), (cx, cy)], fill=col + (26,), width=2)
            rr = 8 + (6 * math.sin(phase_a * math.pi) if changed and phase_a < 1 else 0)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col + (240,))
        if t >= self.t0 - 1.0:
            q = ease_out((t - (self.t0 - 1.0)) / 0.6)
            for j in range(4):
                x, y = self.px(cpos[j])
                glow_dot(d, x, y, 20 * q, KCOL[j])
                d.text((x, y), "+", font=font(F_BOLD, 30), fill=INK + (255,), anchor="mm")
        return im

    def stats(self, t):
        k = self.state(t)
        if k < 0:
            return [("UNLABELED", (205, 205, 235)), ("4 CENTERS", YEL)], 0.0, "ERROR", CYAN
        it = self.its[k]
        i0 = self.its[0]["inertia"]
        done = t >= self.t1
        shown = self.its[-1]["inertia"] if done else it["inertia"]
        label = "STOPPED CHANGING" if done else f"ROUND {k + 1}"
        return [(label, YEL), (f"ERROR {shown:0.1f}", CYAN)], 1 - shown / i0, "ERROR SHRINKING", CYAN


# ---------------------------------------------------------------- 3. Q-learning maze
MAZE = ["........G", ".####.##.", ".....T...", "##.####..", "...T.....", ".##.####.", "....#....", ".##.#.##.", "S...#...."]
N = 9
CS = VS / N


class QReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.wall = {(r, c) for r in range(N) for c in range(N) if MAZE[r][c] == "#"}
        self.trap = {(r, c) for r in range(N) for c in range(N) if MAZE[r][c] == "T"}
        self.goal = next((r, c) for r in range(N) for c in range(N) if MAZE[r][c] == "G")
        self.start = next((r, c) for r in range(N) for c in range(N) if MAZE[r][c] == "S")
        rng = np.random.default_rng(5)
        Q = np.zeros((N, N, 4))
        mv = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.mv = mv
        eps_list, recs = [], {}
        show = cfg["show"]

        def step(s, a):
            r, c = s[0] + mv[a][0], s[1] + mv[a][1]
            if not (0 <= r < N and 0 <= c < N) or (r, c) in self.wall:
                return s, -0.05, False
            if (r, c) == self.goal:
                return (r, c), 1.0, True
            if (r, c) in self.trap:
                return (r, c), -1.0, True
            return (r, c), -0.02, False

        for ep in range(1, 401):
            s, path, eps = self.start, [self.start], max(0.05, 0.4 * (1 - ep / 250))
            q_before = Q.copy() if ep in show else None
            for _ in range(250):
                a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s[0], s[1]] + rng.random(4) * 1e-6))
                s2, rw, done = step(s, a)
                Q[s[0], s[1], a] += 0.35 * (rw + 0.96 * (0 if done else Q[s2[0], s2[1]].max()) - Q[s[0], s[1], a])
                s = s2
                path.append(s)
                if done:
                    break
            if ep in show:
                recs[ep] = dict(path=path, qb=q_before, qa=Q.copy())
        # final greedy run
        s, path = self.start, [self.start]
        for _ in range(60):
            a = int(np.argmax(Q[s[0], s[1]]))
            s, _, done = step(s, a)
            path.append(s)
            if done:
                break
        recs["final"] = dict(path=path, qb=Q.copy(), qa=Q.copy())
        self.recs = recs
        self.sched = []
        t = cfg["t0"]
        for key, dur in zip(show + ["final"], cfg["durs"]):
            self.sched.append((t, t + dur, key))
            t += dur + 0.4
        self.first_len = len(recs[show[0]]["path"]) - 1
        self.final_len = len(path) - 1
        self.end_t = t - 0.4
        self.reached = path[-1] == self.goal

    def cur(self, t):
        for i, (a, b, key) in enumerate(self.sched):
            if t <= b + 0.4 or i == len(self.sched) - 1:
                if t < a:
                    return i, key, 0.0, t < a - 0.0
                return i, key, clamp((t - a) / (b - a)), False
        return len(self.sched) - 1, self.sched[-1][2], 1.0, False

    def cell(self, r, c):
        return (c * CS, r * CS, (c + 1) * CS, (r + 1) * CS)

    def content(self, t):
        im = Image.new("RGB", (VS, VS), (12, 8, 34))
        d = ImageDraw.Draw(im, "RGBA")
        i, key, prog, pre = self.cur(t)
        rec = self.recs[key]
        mix = smooth((prog - 0.75) / 0.25)
        Q = rec["qb"] * (1 - mix) + rec["qa"] * mix
        V = Q.max(2)
        for r in range(N):
            for c in range(N):
                x0, y0, x1, y1 = self.cell(r, c)
                if (r, c) in self.wall:
                    d.rounded_rectangle([x0 + 4, y0 + 4, x1 - 4, y1 - 4], radius=16, fill=(92, 70, 170, 255), outline=(190, 160, 255, 200), width=3)
                    continue
                d.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 12), width=2)
                v = V[r, c]
                if v > 0.02:
                    u = clamp(v / 0.8)
                    col = tuple(int(x) for x in cmap(u, [(0, (20, 80, 130)), (0.5, (40, 200, 190)), (1, (255, 230, 120))]))
                    d.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=col + (int(40 + 150 * u),))
                    a = int(np.argmax(Q[r, c]))
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    dy, dx = self.mv[a]
                    L = CS * 0.26
                    d.line([(cx - dx * L, cy - dy * L), (cx + dx * L, cy + dy * L)], fill=(255, 255, 255, 190), width=5)
                    hx, hy = cx + dx * L, cy + dy * L
                    d.polygon([(hx + dx * 12, hy + dy * 12), (hx - dy * 10 - dx * 4, hy + dx * 10 - dy * 4), (hx + dy * 10 - dx * 4, hy - dx * 10 - dy * 4)], fill=(255, 255, 255, 220))
        for r, c in self.trap:
            x0, y0, x1, y1 = self.cell(r, c)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            glow_dot(d, cx, cy, 20, (255, 70, 90))
            d.text((cx, cy), "x", font=font(F_BOLD, 34), fill=INK + (255,), anchor="mm")
        gx, gy = self.cell(*self.goal)[0] + CS / 2, self.cell(*self.goal)[1] + CS / 2
        pulse = 0.5 + 0.5 * math.sin(t * 5)
        glow_dot(d, gx, gy, 22 + pulse * 4, GREEN)
        d.text((gx, gy + 62), "GEM", font=font(F_MONO, 22), fill=GREEN + (255,), anchor="mm", stroke_width=4, stroke_fill=INK)
        sx, sy = self.cell(*self.start)[0] + CS / 2, self.cell(*self.start)[1] + CS / 2
        d.ellipse([sx - 30, sy - 30, sx + 30, sy + 30], outline=(255, 255, 255, 150), width=3)
        path = rec["path"]
        s_f = prog * (len(path) - 1)
        k = min(int(s_f), len(path) - 2) if len(path) > 1 else 0
        fr = s_f - k
        def center(p):
            return (p[1] * CS + CS / 2, p[0] * CS + CS / 2)
        if len(path) > 1:
            a, b = center(path[k]), center(path[k + 1])
            pos = (a[0] * (1 - fr) + b[0] * fr, a[1] * (1 - fr) + b[1] * fr)
            pts = [center(path[j]) for j in range(max(0, k - 12), k + 1)] + [pos]
            for j in range(len(pts) - 1):
                al = int(40 + 180 * j / max(1, len(pts) - 1))
                d.line([pts[j], pts[j + 1]], fill=CYAN + (al,), width=8)
            glow_dot(d, pos[0], pos[1], 17, (255, 255, 255))
        return im

    def stats(self, t):
        i, key, prog, pre = self.cur(t)
        rec = self.recs[key]
        n = len(rec["path"]) - 1
        steps = int(prog * n)
        title = "FINAL RUN" if key == "final" else f"EPISODE {key}"
        done = prog >= 1.0
        info = f"{n} STEPS" if done else f"STEP {steps}"
        ok = rec["path"][-1] == self.goal
        col = GREEN if (done and ok) else YEL
        frac = 1 - min(1, (n if done else steps) / self.first_len) if key != self.sched[0][2] else 0.0
        return [(title, YEL), (info, col)], frac, "FEWER STEPS TO THE GEM", GREEN


# ---------------------------------------------------------------- 4. overfitting
class OFReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        r = np.random.default_rng(25)
        self.xt = np.sort(r.uniform(0.04, 0.96, 14))
        self.yt = np.sin(2 * np.pi * self.xt) + r.normal(0, 0.28, 14)
        self.xs = np.sort(r.uniform(0.08, 0.92, 40))
        self.ys = np.sin(2 * np.pi * self.xs) + r.normal(0, 0.28, 40)
        self.xx = np.linspace(0, 1, 260)
        self.degs = [1, 2, 3, 4, 6, 9, 11]
        self.curves, self.tr_err, self.te_err = [], [], []
        for dg in self.degs:
            P = np.polynomial.Polynomial.fit(self.xt, self.yt, dg, domain=[0, 1])
            self.curves.append(np.clip(P(self.xx), -6, 6))
            self.tr_err.append(float(np.mean((P(self.xt) - self.yt) ** 2)))
            self.te_err.append(float(np.mean((P(self.xs) - self.ys) ** 2)))
        self.t0 = cfg["t0"]
        self.hold, self.morph = cfg["hold"], cfg["morph"]

    def deg_state(self, t):
        u = (t - self.t0)
        if u <= 0:
            return 0, 0, 0.0
        first = self.hold + 0.4
        if u < first:
            return 0, 0, 0.0
        u -= first
        per = self.morph + self.hold
        k = int(u // per)
        if k >= len(self.degs) - 1:
            return len(self.degs) - 1, len(self.degs) - 1, 0.0
        w = (u - k * per) / self.morph
        return k, k + 1, smooth(w)

    def px(self, x, y):
        return (70 + x * 820, 480 - y * 160)

    def content(self, t):
        im = Image.new("RGB", (VS, VS), (14, 9, 40))
        d = ImageDraw.Draw(im, "RGBA")
        for g in range(-2, 3):
            y = self.px(0, g)[1]
            d.line([(60, y), (VS - 40, y)], fill=(255, 255, 255, 14 if g else 40), width=2)
        for g in range(0, 6):
            x = self.px(g / 5, 0)[0]
            d.line([(x, 120), (x, VS - 120)], fill=(255, 255, 255, 10), width=2)
        a, b, w = self.deg_state(t)
        cur = self.curves[a] * (1 - w) + self.curves[b] * w
        # true pattern reveal
        if t >= self.cfg["true_t"]:
            q = clamp((t - self.cfg["true_t"]) / 0.8)
            pts = [self.px(x, math.sin(2 * math.pi * x)) for x in np.linspace(0, 1, 120)]
            for i in range(0, len(pts) - 1, 2):
                d.line([pts[i], pts[i + 1]], fill=(255, 255, 255, int(190 * q)), width=4)
            d.text((VS - 70, 120), "TRUE PATTERN", font=font(F_MONO, 26), fill=(255, 255, 255, int(230 * q)), anchor="rm", stroke_width=4, stroke_fill=INK)
        pts = [self.px(x, y) for x, y in zip(self.xx, cur)]
        d.line(pts, fill=PINK + (70,), width=22, joint="curve")
        d.line(pts, fill=PINK + (255,), width=8, joint="curve")
        if t >= self.cfg["test_t"]:
            q = clamp((t - self.cfg["test_t"]) / 0.8)
            for x, y in zip(self.xs, self.ys):
                px_, py_ = self.px(x, y)
                d.ellipse([px_ - 11, py_ - 11, px_ + 11, py_ + 11], outline=YEL + (int(255 * q),), width=4)
        for x, y in zip(self.xt, self.yt):
            px_, py_ = self.px(x, y)
            glow_dot(d, px_, py_, 9, CYAN)
        if t >= self.cfg["payoff"]:
            q = ease_out((t - self.cfg["payoff"]) / 0.5)
            ratio = self.te_err[-1] / min(self.te_err)
            txt = f"{ratio:0.0f}x WORSE ON NEW DATA"
            f = font(F_BOLD, 50)
            w = tw(d, txt, f) + 70
            d.rounded_rectangle([(VS - w) / 2, VS - 150 + (1 - q) * 40, (VS + w) / 2, VS - 60 + (1 - q) * 40], radius=45, fill=(10, 6, 28, int(235 * q)), outline=YEL + (int(255 * q),), width=4)
            d.text((VS / 2, VS - 105 + (1 - q) * 40), txt, font=f, fill=(255, 255, 255, int(255 * q)), anchor="mm")
        # error panel
        tre = self.tr_err[a] * (1 - w) + self.tr_err[b] * w
        tee = self.te_err[a] * (1 - w) + self.te_err[b] * w
        d.rounded_rectangle([28, 28, 520, 168], radius=26, fill=(10, 6, 28, 215), outline=(255, 255, 255, 40), width=2)
        d.text((52, 62), "TRAINING ERROR", font=font(F_MONO, 24), fill=CYAN + (255,), anchor="lm")
        d.rounded_rectangle([52, 84, 352, 100], radius=8, fill=(255, 255, 255, 30))
        d.rounded_rectangle([52, 84, 52 + 300 * clamp(math.sqrt(tre) / 1.0), 100], radius=8, fill=CYAN + (255,))
        d.text((498, 92), f"{tre:0.2f}", font=font(F_MONO, 28), fill=(255, 255, 255, 255), anchor="rm")
        if t >= self.cfg["test_t"]:
            d.text((52, 128), "NEW-DATA ERROR", font=font(F_MONO, 24), fill=YEL + (255,), anchor="lm")
            d.rounded_rectangle([52, 148, 352, 160], radius=6, fill=(255, 255, 255, 30))
            d.rounded_rectangle([52, 148, 52 + 300 * clamp(math.sqrt(tee) / 3.0), 160], radius=6, fill=YEL + (255,))
            d.text((498, 154), f"{tee:0.2f}", font=font(F_MONO, 28), fill=(255, 255, 255, 255), anchor="rm")
        return im

    def stats(self, t):
        a, b, w = self.deg_state(t)
        deg = self.degs[b] if w > 0.5 else self.degs[a]
        tre = self.tr_err[a] * (1 - w) + self.tr_err[b] * w
        tee = self.te_err[a] * (1 - w) + self.te_err[b] * w
        items = [(f"COMPLEXITY {deg}", YEL), (f"TRAIN {tre:0.2f}", CYAN)]
        if t >= self.cfg["test_t"]:
            items.append((f"NEW {tee:0.1f}", PINK))
        frac = 1 - min(1, tre / self.tr_err[0])
        return items, frac, "FITS THE TRAINING DOTS", CYAN


# ---------------------------------------------------------------- audio + render
def make_audio(cfg, lines, model_dir):
    T = cfg["T"]
    n = int(T * SR)
    buf = np.zeros(n, np.float32)
    bed = music_bed(T, cfg.get("seed", 5))[:n]
    buf[:len(bed)] += bed * 0.42
    vo = []
    for st, text in lines:
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        parts = []
        for s in sents:
            x, sr = voice.say(model_dir, s, cfg.get("voice", "af_heart"), cfg.get("speed", 0.98))
            parts += [x / (np.max(np.abs(x)) or 1) * 0.95, np.zeros(int(SR * (0.34 if s.endswith("?") else 0.26)), np.float32)]
        x = np.concatenate(parts[:-1])
        i = int(st * SR)
        buf[i:i + len(x)] += x[: n - i]
        vo.append((st, st + len(x) / SR, text))
    for (a, b, _), (c, _, _) in zip(vo, vo[1:]):
        if b > c:
            print(f"WARNING vo overlap {b:.2f} > {c:.2f}")
    print("vo end", vo[-1][1])
    return buf, vo


def render(reel, out, buf, T, stills=None):
    if stills:
        ts = reel.cfg["still_times"]
        fr = [reel.frame(t).resize((300, 533)) for t in ts]
        sheet = Image.new("RGB", (300 * len(fr), 533))
        for i, f_ in enumerate(fr):
            sheet.paste(f_, (i * 300, 0))
        sheet.save(stills)
        return
    buf = buf / (np.max(np.abs(buf)) or 1) * 0.92
    wav = os.path.splitext(out)[0] + "_a.wav"
    sf.write(wav, buf, SR)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-15:TP=-1.2:LRA=11",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for i in range(int(T * FPS)):
        ff.stdin.write(reel.frame(i / FPS).tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)


def chime(b, t0, base=523):
    for k, f in enumerate((1, 1.26, 1.5, 2.0)):
        tone(b, t0 + k * 0.07, 1.4, base * f, base * f, 0.14, "rich")


def build(which, model):
    if which == "gd":
        cfg = dict(T=32.0, c1=ORANGE, c2=PINK, t0=4.2, rate=3.3, payoff=13.4, seed=3, still_times=[0.3, 5, 9, 13, 14.5, 18, 24, 30],
                   header=lambda t: (([("3 AIs. 1 HILL.", (255, 255, 255)), ("WHO WINS?", YEL)], "WATCH THE 3 BALLS") if t < 13.4 else
                                     ([("STEP SIZE", (255, 255, 255)), ("DECIDES IT.", GREEN)], "LEARNING RATE")))
        lines = [(0.2, "Three AIs. One hill. Only one of them will find the bottom."),
                 (4.4, "They all use gradient descent. Feel the slope, then step downhill."),
                 (9.4, "The only difference is the step size. Engineers call it the learning rate."),
                 (14.6, "There. The right step size glides straight to the bottom."),
                 (18.8, "Too small, and it crawls. Too big, and it bounces around and never settles."),
                 (25.0, "That one number can decide whether training works at all."),
                 (28.8, "Follow for more AI, explained in motion.")]
        buf, vo = make_audio(cfg, lines, model)
        reel = GDReel(cfg, vo)
        for k in range(1, 5):
            tone(buf, cfg["t0"] + k * 2.0, 0.18, 400 + 40 * k, 440 + 40 * k, 0.05)
        chime(buf, 13.4)
        return reel, buf, cfg
    if which == "km":
        cfg = dict(T=31.0, c1=CYAN, c2=VIOLET, t0=5.4, D=3.3, payoff=18.8, seed=6, still_times=[0.3, 4, 6, 9, 13, 17, 19.2, 28],
                   header=lambda t: (([("NOBODY LABELED", (255, 255, 255)), ("THIS DATA.", YEL)], "WATCH IT SORT ITSELF") if t < 18.8 else
                                     ([("IT FOUND 4", (255, 255, 255)), ("GROUPS ALONE.", GREEN)], "NO LABELS NEEDED")))
        lines = [(0.2, "Nobody labeled this data. Watch the AI sort it anyway."),
                 (4.6, "It drops four random centers, and every dot joins its nearest one."),
                 (10.0, "Then each center moves to the middle of its group. And again."),
                 (14.4, "Join. Move. Repeat. Each round, the groups get tighter."),
                 (19.0, "That's k-means. It finds patterns with no labels, which is how apps can group customers or songs."),
                 (26.8, "Follow for more AI, explained in motion.")]
        buf, vo = make_audio(cfg, lines, model)
        reel = KMReel(cfg, vo)
        for k in range(reel.n):
            tone(buf, cfg["t0"] + k * reel.D, 0.2, 380 + 50 * k, 520 + 50 * k, 0.06)
        chime(buf, 18.8)
        print("kmeans iterations", reel.n)
        return reel, buf, cfg
    if which == "ql":
        cfg = dict(T=36.0, c1=GREEN, c2=CYAN, t0=5.0, show=[1, 2, 3, 8, 25, 80, 200], durs=[4.0, 2.2, 2.0, 1.8, 1.8, 1.8, 1.8, 3.8], payoff=None, seed=2,
                   still_times=[0.3, 4, 7, 11, 15, 19, 24, 28, 33])
        cfg["header"] = lambda t: (([("IT GETS LOST.", (255, 255, 255)), ("THEN LEARNS.", YEL)], "WATCH 200 TRIES") if t < cfg["pay_t"] else
                                   ([("NOW IT KNOWS", (255, 255, 255)), ("THE WAY.", GREEN)], "NOBODY TOLD IT"))
        cfg["pay_t"] = 99.0
        reel = QReel(cfg, [])
        r1 = reel.recs[1]["path"]
        end1 = "falls into a trap" if r1[-1] in reel.trap else ("finds the gem by luck" if r1[-1] == reel.goal else "runs out of steps")
        lines = [(0.2, "This AI has no map. It just wanders, and only gets a reward at the gem."),
                 (5.2, f"Episode one. It's lost. It wanders for {len(r1) - 1} steps, and then it {end1}."),
                 (11.0, "But every step teaches it something. Good moves get a higher score. Bad moves get a lower one."),
                 (17.4, "Those scores spread backward from the gem, like a trail of breadcrumbs."),
                 (23.0, f"By the last run, it walks straight there in {reel.final_len} steps."),
                 (27.4, "That's reinforcement learning. Try, get a reward, update, repeat."),
                 (32.0, "Follow for more AI, explained in motion.")]
        buf, vo = make_audio(cfg, lines, model)
        reel.vo = vo
        cfg["pay_t"] = reel.sched[-1][0]
        cfg["payoff"] = reel.sched[-1][0] + 0.2
        for a_, b_, key in reel.sched:
            tone(buf, a_, 0.2, 400, 560, 0.05)
        chime(buf, cfg["payoff"])
        print("first len", reel.first_len, "final len", reel.final_len, "reached", reel.reached, "sched", [(round(a, 1), round(b, 1), k) for a, b, k in reel.sched])
        return reel, buf, cfg
    cfg = dict(T=36.0, c1=PINK, c2=VIOLET, t0=4.6, hold=1.9, morph=0.8, test_t=16.4, true_t=26.2, payoff=26.4, seed=8, still_times=[0.3, 4, 8, 13, 17, 21, 25, 28, 33],
               header=lambda t: (([("PERFECT SCORE.", (255, 255, 255)), ("THEN IT FAILS.", YEL)], "WATCH THE CURVE") if t < 24.0 else
                                 ([("IT MEMORIZED.", (255, 255, 255)), ("NOT LEARNED.", PINK)], "OVERFITTING")))
    lines = [(0.2, "This AI is about to get a perfect score. Then fail in real life."),
             (4.8, "The filled dots are its training data. It draws a curve through them."),
             (9.4, "Make the curve more complex, and the error on training data keeps dropping."),
             (15.2, "But look at the hollow dots. That's new data it has never seen."),
             (20.4, "Watch what happens. The curve hits every training dot, and misses everything else."),
             (26.6, "That's overfitting. It memorized the answers instead of learning the pattern."),
             (32.4, "Follow for more AI, explained in motion.")]
    buf, vo = make_audio(cfg, lines, model)
    reel = OFReel(cfg, vo)
    for k in range(6):
        tone(buf, cfg["t0"] + 0.4 + 1.9 + k * 2.7, 0.2, 380 + 40 * k, 480 + 40 * k, 0.05)
    chime(buf, cfg["payoff"], 440)
    print("train", [round(x, 3) for x in reel.tr_err], "test", [round(x, 3) for x in reel.te_err])
    return reel, buf, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg = build(a.which, a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    print("done", a.out)


if __name__ == "__main__":
    main()
