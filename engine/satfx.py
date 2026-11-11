"""Satisfying real-simulation reels: sorting race, boids flock, self-organizing color map.
usage: python satfx.py sort|boids|som out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
"""
import argparse
import bisect
import colorsys
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import pw, inv, write_post  # noqa: E402
from scenes import CYAN, GREEN, PINK, VIOLET, YEL, ORANGE, Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, INK, SR, VS, clamp, ease_out, font, smooth, tone  # noqa: E402


def hsv(h, s=0.85, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


# ============================================================ 1. sorting race
N = 64


def rec_bubble(a):
    a = list(a)
    steps, c = [(0, tuple(a), ())], 0
    n = len(a)
    for i in range(n - 1):
        sw = False
        for j in range(n - 1 - i):
            c += 1
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                sw = True
            steps.append((c, tuple(a), (j, j + 1)))
        if not sw:
            break
    steps.append((c, tuple(a), ()))
    return steps, c


def rec_quick(a):
    a = list(a)
    st = {"c": 0}
    steps = [(0, tuple(a), ())]

    def qs(lo, hi):
        if lo >= hi:
            return
        p = a[hi]
        i = lo
        for j in range(lo, hi):
            st["c"] += 1
            if a[j] < p:
                a[i], a[j] = a[j], a[i]
                i += 1
            steps.append((st["c"], tuple(a), (j, hi)))
        a[i], a[hi] = a[hi], a[i]
        steps.append((st["c"], tuple(a), (i,)))
        qs(lo, i - 1)
        qs(i + 1, hi)
    qs(0, len(a) - 1)
    steps.append((st["c"], tuple(a), ()))
    return steps, st["c"]


def rec_merge(a):
    a = list(a)
    st = {"c": 0}
    steps = [(0, tuple(a), ())]

    def ms(lo, hi):
        if hi - lo < 1:
            return
        mid = (lo + hi) // 2
        ms(lo, mid)
        ms(mid + 1, hi)
        L, R = a[lo:mid + 1], a[mid + 1:hi + 1]
        i = j = 0
        k = lo
        while i < len(L) and j < len(R):
            st["c"] += 1
            if L[i] <= R[j]:
                a[k] = L[i]
                i += 1
            else:
                a[k] = R[j]
                j += 1
            steps.append((st["c"], tuple(a), (k,)))
            k += 1
        while i < len(L):
            a[k] = L[i]
            i += 1
            steps.append((st["c"], tuple(a), (k,)))
            k += 1
        while j < len(R):
            a[k] = R[j]
            j += 1
            steps.append((st["c"], tuple(a), (k,)))
            k += 1
    ms(0, len(a) - 1)
    steps.append((st["c"], tuple(a), ()))
    return steps, st["c"]


ALG = [("BUBBLE SORT", rec_bubble, ORANGE), ("QUICK SORT", rec_quick, CYAN), ("MERGE SORT", rec_merge, GREEN)]


class SortReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.pal = [hsv(v / N * 0.85) for v in range(N)]

    def state(self, k, t):
        c = (t - self.cfg["t0"]) * self.cfg["R"]
        c = max(0.0, c)
        steps = self.cfg["S"][k]
        keys = [s[0] for s in steps]
        i = min(len(steps) - 1, bisect.bisect_right(keys, c) - 1)
        # within same comparison count there may be several write steps -> advance by fractional position
        j = i
        while j + 1 < len(steps) and steps[j + 1][0] == steps[i][0] and (j - i) < 1000:
            j += 1
        # spread steps sharing a count across the time to next count
        return steps[i], steps[i][0] >= self.cfg["tot"][k] and i >= len(steps) - 2

    def content(self, t):
        im = Image.new("RGB", (VS, VS), (12, 7, 34))
        d = ImageDraw.Draw(im, "RGBA")
        ph = 296
        for k, (nm, _, col) in enumerate(ALG):
            y0 = 14 + k * (ph + 20)
            d.rounded_rectangle([14, y0, VS - 14, y0 + ph], radius=22, fill=(18, 10, 48, 255), outline=col + (200,), width=3)
            (c, arr, hl), done = self.state(k, t)
            d.text((38, y0 + 30), nm, font=font(F_MONO, 30), fill=col + (255,), anchor="lm")
            d.text((VS - 38, y0 + 30), f"COMPARISONS {c}", font=font(F_MONO, 30), fill=(255, 255, 255, 255), anchor="rm")
            bw = (VS - 60) / N
            base = y0 + ph - 18
            for i, v in enumerate(arr):
                h = 30 + v / (N - 1) * 190
                x = 30 + i * bw
                colr = (255, 255, 255) if (i in hl and not done) else (self.pal[v] if not done else (90, 255, 170))
                d.rounded_rectangle([x + 1, base - h, x + bw - 1.5, base], radius=3, fill=colr + (255,))
            if done:
                d.text((VS / 2, y0 + 78), "DONE", font=font(F_BOLD, 44), fill=GREEN + (255,), anchor="mm", stroke_width=6, stroke_fill=INK)
        return im

    def stats(self, t):
        c = (t - self.cfg["t0"]) * self.cfg["R"]
        frac = clamp(c / max(self.cfg["tot"]))
        return [("SAME LIST", YEL), (f"{N} NUMBERS", CYAN)], frac, "BUBBLE SORT PROGRESS (COMPARISONS)", ORANGE


def build_sort(model):
    rng = random.Random(7)
    arr = list(range(N))
    rng.shuffle(arr)
    S, tot = [], []
    for nm, fn, col in ALG:
        s, c = fn(arr)
        assert [x[1] for x in s][-1] == tuple(range(N)), nm
        S.append(s)
        tot.append(c)
    print("comparisons", dict(zip([a[0] for a in ALG], tot)))
    t0 = 3.2
    R = tot[0] / 21.0
    tq, tm, tb = (t0 + tot[1] / R), (t0 + tot[2] / R), (t0 + tot[0] / R)
    cfg = dict(T=33.0, c1=CYAN, c2=GREEN, t0=t0, R=R, S=S, tot=tot, payoff=round(min(tq, tm), 1), seed=31, still_times=[0.3, 3.5, 6, 9, round(tb - 1, 1), round(tb + 0.5, 1), 26, 31],
               header=lambda t: (([("3 WAYS TO SORT.", (255, 255, 255)), ("WHO WINS?", YEL)], "WATCH THE RACE") if t < tb else
                                 ([("BUBBLE SORT", (255, 255, 255)), ("IS SLOW.", PINK)], "SAME LIST, 3 METHODS")))
    fastest = "QUICK SORT" if tot[1] < tot[2] else "MERGE SORT"
    lines = [(0.2, "Three ways to sort the same shuffled list. Watch the race."),
             (4.0, "The score is simple. Fewer comparisons means a faster method."),
             (8.4, f"{'Quick' if tot[1] < tot[2] else 'Merge'} sort is done. Just {min(tot[1], tot[2])} comparisons."),
             (12.0, f"{'Merge' if tot[1] < tot[2] else 'Quick'} sort finishes with {max(tot[1], tot[2])}."),
             (15.6, f"Bubble sort keeps comparing neighbors again and again. It needs {tot[0]}."),
             (max(tb + 0.4, 22.6), "And the gap explodes as the list gets longer. That's why the method matters."),
             (29.6, "Follow for more, explained in motion.")]
    buf, vo = make_audio(cfg, lines, model)
    # sonification: pitch = compared value
    for k, (nm, _, col) in enumerate(ALG):
        base = [220, 330, 440][k]
        last = -1
        for c_, a_, hl in S[k]:
            if hl and c_ != last or (hl and k == 2):
                tt = t0 + c_ / R
                if tt < cfg["T"] - 1 and (c_ % (3 if k == 0 else 1) == 0):
                    v = a_[hl[0]] if hl[0] < N else 0
                    tone(buf, tt, 0.05, base * (1 + v / N), base * (1 + v / N), 0.012 if k == 0 else 0.02)
            last = c_
    chime(buf, cfg["payoff"])
    reel = SortReel(cfg, vo)
    post = dict(title="SAT-1  Same list. 3 sorting methods. Who wins?",
                caption=f"Bubble sort vs quick sort vs merge sort on the same shuffled list of {N} numbers. Counting comparisons: bubble {tot[0]}, quick {tot[1]}, merge {tot[2]}. The gap gets much bigger on longer lists.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nWhich sound is more satisfying? Comment BUBBLE, QUICK or MERGE.",
                tags=["coding", "algorithms", "computerscience", "satisfying", "programming", "shipwithpip"],
                pinned="Want this as an app? Comment SORT and I'll send the prompt. Which one is the most satisfying: BUBBLE, QUICK or MERGE?",
                notes=f"List of {N} shuffled distinct numbers (seed 7). Comparisons counted: bubble (with early exit) {tot[0]}, quicksort (Lomuto, last-element pivot) {tot[1]}, merge sort {tot[2]}. Fastest by comparisons: {fastest}.\nAnimation speed is tied to comparison count, not wall-clock time; real speed also depends on swaps/memory. Pitch of each tick = value being compared.")
    return reel, buf, cfg, post


# ============================================================ 2. boids flock
NB = 170


def sim_boids(T=33.0, fps=30, seed=3):
    r = np.random.default_rng(seed)
    p = r.random((NB, 2))
    ang = r.uniform(0, 2 * np.pi, NB)
    v = np.stack([np.cos(ang), np.sin(ang)], 1) * 0.12
    dt = 1.0 / fps
    P, V, PH = [], [], []
    for s in range(int(T * fps)):
        d = p[None, :, :] - p[:, None, :]
        d -= np.round(d)
        dist = np.sqrt((d ** 2).sum(-1)) + np.eye(NB) * 9
        nb = dist < 0.12
        close = dist < 0.05
        cnt = nb.sum(1, keepdims=True).clip(min=1)
        coh = (d * nb[..., None]).sum(1) / cnt
        ali = (v[None, :, :] * nb[..., None]).sum(1) / cnt - v
        sep = -(d * close[..., None] / (dist[..., None] ** 2 + 1e-4)).sum(1)
        warm = min(1.0, s / (fps * 1.0))
        v = v + dt * (0.45 * coh * (nb.sum(1, keepdims=True) > 0) + 2.4 * ali * (nb.sum(1, keepdims=True) > 0) + 0.0016 * sep) * warm
        v += r.normal(0, 0.015, v.shape)
        sp = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        v = v / sp * np.clip(sp, 0.10, 0.16)
        p = (p + v * dt) % 1.0
        P.append(p.copy())
        V.append(v.copy())
        u = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        PH.append(float(np.linalg.norm(u.mean(0))))
    return np.array(P), np.array(V), np.array(PH)


class BoidReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.P, self.V, self.PH = cfg["P"], cfg["V"], cfg["PH"]

    def idx(self, t):
        return int(clamp((t - 0.3) * 30, 0, len(self.P) - 1))

    def content(self, t):
        im = Image.new("RGB", (VS, VS), (10, 6, 30))
        d = ImageDraw.Draw(im, "RGBA")
        i = self.idx(t)
        for lag in range(14, -1, -1):
            j = max(0, i - lag)
            a = int(255 * (1 - lag / 15) ** 1.6)
            w = max(2, int(10 * (1 - lag / 16)))
            for k in range(NB):
                x, y = self.P[j, k] * VS
                if lag:
                    pj = self.P[max(0, j - 1), k] * VS
                    if abs(pj[0] - x) > VS / 2 or abs(pj[1] - y) > VS / 2:
                        continue
                    ang = math.atan2(self.V[j, k, 1], self.V[j, k, 0]) / (2 * math.pi) % 1
                    d.line([(pj[0], pj[1]), (x, y)], fill=hsv(ang, 0.8, 1.0) + (int(a * 0.55),), width=w)
        for k in range(NB):
            x, y = self.P[i, k] * VS
            ang = math.atan2(self.V[i, k, 1], self.V[i, k, 0])
            col = hsv(ang / (2 * math.pi) % 1, 0.7, 1.0)
            tip = (x + math.cos(ang) * 17, y + math.sin(ang) * 17)
            l = (x + math.cos(ang + 2.5) * 11, y + math.sin(ang + 2.5) * 11)
            r_ = (x + math.cos(ang - 2.5) * 11, y + math.sin(ang - 2.5) * 11)
            d.polygon([tip, l, r_], fill=col + (255,), outline=(255, 255, 255, 230))
        return im

    def stats(self, t):
        ph = self.PH[self.idx(t)]
        return [("NO LEADER", YEL), (f"IN SYNC {ph * 100:0.0f}%", CYAN)], ph, "HOW ALIGNED THE FLOCK IS", CYAN


def build_boids(model):
    P, V, PH = sim_boids()
    print("order param t=0,3,8,12,20,30:", [round(PH[int(x * 30)], 2) for x in (0.1, 3, 8, 12, 20, 30)])
    t_sync = next((i / 30 + 0.3 for i, x in enumerate(PH) if x > 0.9), None)
    print("t_sync", t_sync)
    cfg = dict(T=34.0, c1=VIOLET, c2=CYAN, P=P, V=V, PH=PH, payoff=round(t_sync or 20, 1), seed=41, still_times=[0.5, 3, 7, 11, 15, 20, 26, 31],
               header=lambda t: (([("NOBODY LEADS", (255, 255, 255)), ("THIS FLOCK.", YEL)], "WATCH IT ORGANIZE") if t < 20 else
                                 ([("3 SIMPLE RULES.", (255, 255, 255)), ("ONE FLOCK.", GREEN)], "NO LEADER NEEDED")))
    lines = [(0.2, f"Nobody is leading this flock. At the start it's chaos, only {PH[15] * 100:0.0f} percent in sync."),
             (7.6, "Each dot follows just three rules. Steer away from close neighbors. Match the direction of the group. And drift toward its center."),
             (16.8, "Watch. No leader and no plan, and the flock lines up on its own."),
             (21.6, f"Now it's {PH[22 * 30] * 100:0.0f} percent in sync."),
             (24.2, "Computer scientist Craig Reynolds came up with these three rules in 1986."),
             (29.8, "Follow for more, explained in motion.")]
    buf, vo = make_audio(cfg, lines, model)
    chime(buf, cfg["payoff"])
    reel = BoidReel(cfg, vo)
    post = dict(title="SAT-2  Nobody leads this flock.",
                caption=f"{NB} dots, three simple rules, no leader: steer away from close neighbors, match the group's direction, drift toward its center. The flock starts as chaos ({PH[15] * 100:.0f}% in sync) and organizes itself. These are Craig Reynolds' boids rules (1986).\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment FLOCK and I'll send the prompt so you can build it.",
                tags=["ai", "simulation", "coding", "satisfying", "emergence", "shipwithpip"],
                pinned="Build me a single-file HTML canvas page with 170 boids on a wrap-around screen. Rules: separation, alignment, cohesion (Craig Reynolds' boids). Color each boid by its heading, draw short fading trails, and show a live 'in sync' percentage (length of the average heading vector). Make it glow.",
                notes=f"{NB} boids, torus world, neighbor radius 0.12, separation radius 0.05, small random noise, speeds clamped 0.10-0.16 units/s.\n'In sync %' = length of the mean unit heading vector x100 (0% = random headings, 100% = identical headings). t=0.5s: {PH[15] * 100:.0f}%, t=22s: {PH[22 * 30] * 100:.0f}%.\nBoids: Craig Reynolds 1986 (paper at SIGGRAPH 1987).")
    return reel, buf, cfg, post


# ============================================================ 3. self-organizing color map
G = 30


def run_som(steps=4000, seed=2):
    r = np.random.default_rng(seed)
    Wt = r.random((G, G, 3))
    yy, xx = np.mgrid[0:G, 0:G]
    snaps_at = sorted(set(int(round(x)) for x in np.geomspace(1, steps, 130)) | {0})
    out = []

    def mism(W_):
        a = np.linalg.norm(W_[:, 1:] - W_[:, :-1], axis=2).mean()
        b = np.linalg.norm(W_[1:] - W_[:-1], axis=2).mean()
        return float((a + b) / 2)
    for s in range(steps + 1):
        if s in snaps_at:
            out.append((s, mism(Wt), (np.clip(Wt, 0, 1) * 255).astype(np.uint8).copy()))
        x = r.random(3)
        d = ((Wt - x) ** 2).sum(2)
        by, bx = np.unravel_index(np.argmin(d), d.shape)
        frac = s / steps
        sig = 3.0 * (0.8 / 3.0) ** frac
        lr = 0.6 * (0.02 / 0.6) ** frac
        h = np.exp(-((yy - by) ** 2 + (xx - bx) ** 2) / (2 * sig ** 2))
        Wt += lr * h[..., None] * (x - Wt)
    return out


class SomReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.snaps = cfg["snaps"]
        self.imgs = [Image.fromarray(s[2]).resize((VS, VS), Image.BICUBIC) for s in self.snaps]

    def idx(self, t):
        return int(min(len(self.snaps) - 1, max(0, round(pw(self.cfg["imap"], t)))))

    def content(self, t):
        i = self.idx(t)
        im = self.imgs[i].copy()
        d = ImageDraw.Draw(im, "RGBA")
        d.text((VS - 30, 44), f"STEP {self.snaps[i][0]}", font=font(F_MONO, 40), fill=(255, 255, 255, 255), anchor="rm", stroke_width=5, stroke_fill=INK)
        return im

    def stats(self, t):
        i = self.idx(t)
        m = self.snaps[i][1] / self.snaps[0][1]
        return [(f"STEP {self.snaps[i][0]}", YEL), (f"NEIGHBORS DIFFER {m * 100:0.0f}%", PINK)], 1 - m, "HOW ORGANIZED", PINK


def build_som(model):
    out = run_som()
    nS = len(out)
    m0 = out[0][1]
    mf = out[-1][1]
    es = [o[0] for o in out]

    def idx_of(e):
        return min(range(nS), key=lambda k: abs(es[k] - e))
    imap = [(0.8, 0), (3.0, idx_of(3)), (9.0, idx_of(60)), (16.0, idx_of(600)), (22.0, idx_of(2000)), (26.0, nS - 1)]
    print("mism0", m0, "final", mf, "ratio", mf / m0, "steps", es[-1])
    cfg = dict(T=35.0, c1=PINK, c2=VIOLET, snaps=out, imap=imap, payoff=round(inv(imap, idx_of(1200)), 1), seed=51, still_times=[0.5, 3, 8, 12, 17, 22, 27, 31],
               header=lambda t: (([("THIS AI HAS NEVER", (255, 255, 255)), ("SEEN A COLOR.", YEL)], "WATCH IT ORGANIZE") if t < 24 else
                                 ([("IT SORTED", (255, 255, 255)), ("COLORS ALONE.", GREEN)], "NO LABELS, NO TEACHER")))
    lines = [(0.2, "This AI has never seen a color. It starts as pure noise."),
             (4.2, "Now it's shown random colors, one at a time. The closest cell moves toward each color, and drags its neighbors along."),
             (13.0, "Watch. Similar colors drift together, and the noise melts into a smooth map."),
             (19.4, f"Now every cell differs from its neighbors by only {mf / m0 * 100:0.0f} percent of what it did at the start."),
             (25.4, "That's a self-organizing map, invented by Teuvo Kohonen in the nineteen eighties. No labels. No teacher."),
             (32.6, "Follow for more.")]
    buf, vo = make_audio(cfg, lines, model)
    chime(buf, cfg["payoff"])
    reel = SomReel(cfg, vo)
    post = dict(title="SAT-3  This AI has never seen a color.",
                caption=f"A {G}x{G} grid of random colors learns to organize itself with a self-organizing map (Teuvo Kohonen, 1980s). It's shown random colors one at a time, the closest cell moves toward each color and pulls its neighbors along. After {es[-1]} steps, neighboring cells differ by only {mf / m0 * 100:.0f}% of what they did at the start.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment COLORS and I'll send the prompt so you can build it.",
                tags=["ai", "machinelearning", "satisfying", "coding", "generativeart", "shipwithpip"],
                pinned="Build me a single-file HTML page with a 30x30 self-organizing map. Each cell holds an RGB color, starts random, and trains on random colors: find the closest cell, then move it and its neighbors toward the color (Gaussian neighborhood that shrinks, learning rate that decays). Redraw the grid smoothly every few steps and show how much neighboring cells differ.",
                notes=f"SOM grid {G}x{G}, {es[-1]} training steps, input = uniform random RGB, neighborhood sigma 3 -> 0.8, lr 0.6 -> 0.02.\n'Neighbors differ %' = mean RGB distance between adjacent cells, relative to step 0 (random start). Step 0: 100%; final: {mf / m0 * 100:.0f}%.\nSOM: Teuvo Kohonen, early 1980s (1982).")
    return reel, buf, cfg, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = {"sort": build_sort, "boids": build_boids, "som": build_som}[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
