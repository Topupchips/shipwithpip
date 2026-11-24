"""Satisfying visual reels: every motion is computed, every number comes from this file's own stats.

Kinds (rotate through them, see AUTOMATION.md section B2):
  pendulum  physics    pendulum wave, lengths from T = 2*pi*sqrt(L/g)
  galton    physics    Galton board, balls really bounce left/right with p=0.5 (seeded)
  fourier   math art   epicycles (DFT of a curve) drawing a shape with more and more circles
  sunflower math art   seeds placed at a fixed turn angle: 137, 138, then the golden angle
  pathfind  algorithm  BFS flood vs A* on the same seeded maze

usage:
  python3 satisfy.py stats  script.json              print the run's numbers (the only numbers a script may use)
  python3 satisfy.py check  script.json              every number in hook/head/chip/say/caption must be in stats or constants
  python3 satisfy.py sheet  script.json out.png --model DIR    contact sheet (also prints OOV words)
  python3 satisfy.py render script.json out.mp4 --model DIR

Script: {"id", "kind", "seed", "params": {...}, "sync": [[beat_index, u] or [beat_index, fraction, u], ...] (optional, maps video time to sim time),
         "beats": [{"say", "key": [...], "hook": [3 lines] (first beat), "head": [2 lines], "chip": "TEXT", "vis": {...}}],
         "constants": [...], "title", "caption", "hashtags", "sources"}
The last beat should be {"say": "Follow Ship With Pip for more.", "nocap": true}.
"""
import argparse
import colorsys
import json
import math
import os
import re
import subprocess
import sys
from functools import lru_cache

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import minimal as M  # noqa: E402
from minimal import W, H, FPS, SR, font, al, tc, clamp, ease, F_BOLD, F_SEMI, F_MED, F_MONO  # noqa: E402

TH = dict(M.THEMES["dark"]); TH["bg"] = (8, 8, 14); TH["grid"] = (16, 16, 26)
YEL = (255, 214, 64)
VX0, VX1, VY0, VY1 = 60, 1020, 400, 1330  # visual zone
CAP_Y = 1520


def hsv(h, s=0.78, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


# ============================================================ kinds
class Pendulum:
    """n pendulums; pendulum k swings K+k times per `cycle` seconds. Small-angle law T = 2 pi sqrt(L/g) gives each length."""
    def __init__(s, P, seed):
        s.n, s.K, s.cycle, s.g = P.get("n", 15), P.get("K", 20), P.get("cycle", 60), 9.81
        s.f = [(s.K + k) / s.cycle for k in range(s.n)]
        s.L = [s.g * (1 / f / (2 * math.pi)) ** 2 for f in s.f]
        s.event_u = s.cycle

    def stats(s):
        return dict(pendulums=s.n, swings_longest=s.K, swings_shortest=s.K + s.n - 1, cycle_s=s.cycle, half_cycle_s=s.cycle / 2,
                    third_cycle_s=s.cycle / 3, longest_m=round(s.L[0], 2), shortest_m=round(s.L[-1], 2),
                    longest_cm=round(s.L[0] * 100), shortest_cm=round(s.L[-1] * 100),
                    period_longest_s=round(1 / s.f[0], 2), period_shortest_s=round(1 / s.f[-1], 2))

    def draw(s, img, d, u, bi, bp, vis):
        n = s.n; ys = [VY0 + 40 + k * (VY1 - VY0 - 80) / (n - 1) for k in range(n)]; A = 400
        d.line([(540, VY0), (540, VY1)], fill=al((60, 60, 90), 0.6), width=2)
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
        for k in range(n):
            col = hsv(k / n * 0.85)
            for j in range(10, 0, -1):  # motion trail
                uu = u - j * 0.012
                x = 540 + A * math.cos(2 * math.pi * s.f[k] * uu)
                r = 22 * (1 - j / 14)
                gd.ellipse([x - r, ys[k] - r, x + r, ys[k] + r], fill=al(col, 0.06 * (11 - j)))
            x = 540 + A * math.cos(2 * math.pi * s.f[k] * u)
            gd.ellipse([x - 40, ys[k] - 40, x + 40, ys[k] + 40], fill=al(col, 0.35))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(14)))
        d = ImageDraw.Draw(img, "RGBA")
        for k in range(n):
            x = 540 + A * math.cos(2 * math.pi * s.f[k] * u); col = hsv(k / n * 0.85)
            d.line([(540, ys[k]), (x, ys[k])], fill=al(col, 0.25), width=2)
            d.ellipse([x - 24, ys[k] - 24, x + 24, ys[k] + 24], fill=col + (255,))
            d.ellipse([x - 9, ys[k] - 15, x + 3, ys[k] - 5], fill=(255, 255, 255, 120))
        tc(d, (900, VY1 + 40), f"t = {u:4.1f} s", font(F_MONO, 36), al(TH["muted"], 0.9))
        return d


class Galton:
    def __init__(s, P, seed):
        s.rows, s.N = P.get("rows", 12), P.get("balls", 800)
        rng = np.random.default_rng(seed)
        s.p = P.get("p", 0.5)  # chance of bouncing right at each peg (0.5 = fair board)
        s.ch = rng.random((s.N, s.rows)) < s.p  # True = bounce right
        s.bin = s.ch.sum(1); s.counts = np.bincount(s.bin, minlength=s.rows + 1)
        s.release = P.get("release_s", 40.0); s.fall = P.get("fall_s", 1.6); s.event_u = s.release + s.fall
        s.top, s.pegy1 = VY0 + 20, VY0 + 470; s.dx = 66; s.binw = s.dx
        s.cap = int(s.counts.max() * 1.1) + 1; s.per_row = 6

    def stats(s):
        return s._biased() or s._fair()

    def _fair(s):
        mid = s.rows // 2; exp = math.comb(s.rows, mid) / 2 ** s.rows
        return dict(balls=s.N, rows=s.rows, bins=s.rows + 1, middle_bin=int(s.counts[mid]), middle_pct=round(100 * s.counts[mid] / s.N, 1),
                    expected_middle_pct=round(100 * exp, 1), left_edge=int(s.counts[0]), right_edge=int(s.counts[-1]),
                    edges_total=int(s.counts[0] + s.counts[-1]), edge_odds_1_in=2 ** s.rows, paths=2 ** s.rows,
                    middle_paths=math.comb(s.rows, mid), counts=[int(c) for c in s.counts], max_bin=int(s.counts.max()))

    def _biased(s):
        if s.p != 0.5:  # biased board: exact binomial(rows, p) predictions
            p, n = s.p, s.rows; pmf = [math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(n + 1)]
            ex = [s.N * q for q in pmf]; pk = int(np.argmax(s.counts)); epk = int(np.argmax(pmf))
            st = dict(balls=s.N, rows=n, bins=n + 1, p_right_pct=round(100 * p), p_left_pct=round(100 * (1 - p)),
                      counts=[int(c) for c in s.counts], max_bin=int(s.counts.max()), peak_bin=pk, expected_peak_bin=epk,
                      peak_bin_label=pk + 1, expected_peak_label=epk + 1, expected_peak_count=round(ex[epk]),
                      mean_rights=round(n * p, 1), left_edge=int(s.counts[0]), right_edge=int(s.counts[-1]),
                      expected_right_edge=round(ex[-1], 1), expected_left_edge=round(ex[0], 4),
                      right_edge_odds_1_in=round(1 / p ** n), left_edge_odds_1_in=round(1 / (1 - p) ** n),
                      paths=2 ** n, fair_middle_paths=math.comb(n, n // 2), fair_edge_odds_1_in=2 ** n,
                      fair_expected_edge=round(s.N / 2 ** n, 2), middle_bin=int(s.counts[n // 2]),
                      expected_middle=round(ex[n // 2]), fair_expected_middle=round(s.N * math.comb(n, n // 2) / 2 ** n))
            return st
        return None

    def ballpos(s, i, u):
        t0 = i * s.release / s.N; a = (u - t0) / s.fall
        if a < 0: return None
        if a >= 1: return "done"
        r = a * s.rows; k = int(r); fr = r - k
        x = 540 + (s.ch[i, :k].sum() - k / 2) * s.dx
        nx = x + (s.dx / 2 if s.ch[i, k] else -s.dx / 2)
        y0 = s.top + k * (s.pegy1 - s.top) / s.rows
        return x + (nx - x) * fr, y0 + (s.pegy1 - s.top) / s.rows * fr - 26 * math.sin(math.pi * fr)

    def draw(s, img, d, u, bi, bp, vis):
        for k in range(s.rows):
            y = s.top + (k + 0.5) * (s.pegy1 - s.top) / s.rows + 10
            for j in range(k + 1):
                x = 540 + (j - k / 2) * s.dx
                d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(90, 96, 140, 255))
        by1 = VY1; nb = s.rows + 1; x0 = 540 - nb * s.binw / 2
        for j in range(nb + 1):
            d.line([(x0 + j * s.binw, s.pegy1 + 30), (x0 + j * s.binw, by1)], fill=(60, 64, 100, 255), width=3)
        d.line([(x0, by1), (x0 + nb * s.binw, by1)], fill=(60, 64, 100, 255), width=3)
        landed = np.zeros(nb, int); rr = (s.binw - 6) / s.per_row / 2
        rowh = min(2 * rr, (by1 - s.pegy1 - 50) / (s.cap / s.per_row))
        for i in range(s.N):
            p = s.ballpos(i, u)
            if p is None: break
            b = s.bin[i]; col = hsv(0.55 + 0.45 * b / s.rows)
            if p == "done":
                c = landed[b]; landed[b] += 1
                cx = x0 + b * s.binw + 3 + rr + (c % s.per_row) * 2 * rr; cy = by1 - 4 - rowh / 2 - (c // s.per_row) * rowh
                d.ellipse([cx - rr, cy - rowh / 2, cx + rr, cy + rowh / 2], fill=col + (255,))
            else:
                d.ellipse([p[0] - 10, p[1] - 10, p[0] + 10, p[1] + 10], fill=col + (255,))
        if vis.get("curve"):
            g = clamp(bp * 2.5); mid = s.rows // 2; pts = []
            for j in range(nb):
                h = s.N * math.comb(s.rows, j) * s.p ** j * (1 - s.p) ** (s.rows - j) / s.per_row * rowh
                pts.append((x0 + (j + 0.5) * s.binw, by1 - 4 - h))
            n = max(2, int(len(pts) * g)); d.line(pts[:n], fill=al(YEL, 0.95), width=7, joint="curve")
        if vis.get("count", True):
            tc(d, (540, by1 + 44), f"{int(landed.sum())} / {s.N} BALLS", font(F_MONO, 36), al(TH["muted"], 0.9))
        return d


def heart(t):
    return 16 * np.sin(t) ** 3, -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))


def star(t, k=5):
    pts = []
    for i in range(2 * k): r = 1.0 if i % 2 == 0 else 0.42; a = math.pi * i / k - math.pi / 2; pts.append((r * math.cos(a), r * math.sin(a)))
    pts.append(pts[0]); P = np.array(pts); seg = np.linspace(0, 1, len(t), endpoint=False) * (len(P) - 1)
    i = seg.astype(int); f = seg - i; Q = P[i] + (P[i + 1] - P[i]) * f[:, None]
    return Q[:, 0] * 16, Q[:, 1] * 16


class Fourier:
    def __init__(s, P, seed):
        s.M = 1024; t = np.linspace(0, 2 * np.pi, s.M, endpoint=False)
        x, y = (star if P.get("shape") == "star" else heart)(t); z = x + 1j * y
        s.z = z; s.C = np.fft.fft(z) / s.M; s.freqs = np.fft.fftfreq(s.M, 1 / s.M).astype(int)
        order = np.argsort(-np.abs(s.C)); order = order[s.freqs[order] != 0]
        s.order = order; s.scale = 380 / np.abs(z).max(); s.cx, s.cy = 540, (VY0 + VY1) / 2 + 20
        s.ns = sorted({b["vis"]["n"] for b in P.get("_beats", []) if b.get("vis", {}).get("n")})
        s.event_u = 0

    def recon(s, n, t):
        idx = s.order[:n]; c0 = s.C[s.freqs == 0][0]
        return c0 + sum(s.C[i] * np.exp(1j * s.freqs[i] * t) for i in idx)

    def err(s, n):
        t = np.linspace(0, 2 * np.pi, s.M, endpoint=False); r = s.recon(n, t)
        return float(np.mean(np.abs(r - s.z)) / (np.abs(s.z).max() * 2) * 100)

    def stats(s):
        out = dict(circles_available=len(s.order))
        for n in s.ns: out[f"error_pct_{n}"] = round(s.err(n), 1)
        return out

    def draw(s, img, d, u, bi, bp, vis):
        n = vis.get("n")
        if not n: n = s.ns[-1] if s.ns else 50; bp = 1.0
        a = clamp(bp / 0.9); T = 2 * np.pi * a
        tt = np.linspace(0, 2 * np.pi, 600)
        if vis.get("ghost", True):
            gx = s.cx + s.z.real * s.scale; gy = s.cy + s.z.imag * s.scale
            d.line(list(zip(gx[::4], gy[::4])) + [(gx[0], gy[0])], fill=al((70, 70, 110), 0.5), width=2)
        tr = s.recon(n, tt[tt <= T + 1e-9])
        if len(tr) > 1:
            pts = list(zip(s.cx + tr.real * s.scale, s.cy + tr.imag * s.scale))
            glow = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
            gd.line(pts, fill=al(YEL, 0.8), width=16, joint="curve"); img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(10)))
            d = ImageDraw.Draw(img, "RGBA"); d.line(pts, fill=(255, 240, 180, 255), width=6, joint="curve")
        if a < 1:
            c0 = s.C[s.freqs == 0][0]; p = c0
            for j, i in enumerate(s.order[:n]):
                q = p + s.C[i] * np.exp(1j * s.freqs[i] * T); r = abs(s.C[i]) * s.scale
                px, py = s.cx + p.real * s.scale, s.cy + p.imag * s.scale
                col = hsv(0.55 + j / max(n, 1) * 0.4, 0.6, 1)
                if r > 1.5: d.ellipse([px - r, py - r, px + r, py + r], outline=al(col, 0.35), width=2)
                d.line([(px, py), (s.cx + q.real * s.scale, s.cy + q.imag * s.scale)], fill=al(col, 0.85), width=3)
                p = q
            ex, ey = s.cx + p.real * s.scale, s.cy + p.imag * s.scale
            d.ellipse([ex - 10, ey - 10, ex + 10, ey + 10], fill=(255, 255, 255, 255))
        return d


class Sunflower:
    def __init__(s, P, seed):
        s.N = P.get("seeds", 700); s.phi = (1 + 5 ** 0.5) / 2; s.golden = 360 * (1 - 1 / s.phi); s.event_u = 0
        s.cx, s.cy = 540, (VY0 + VY1) / 2 + 10; s.c = 430 / math.sqrt(s.N)

    def stats(s):
        return dict(seeds=s.N, golden_angle=round(s.golden, 3), golden_angle_1dp=round(s.golden, 1), golden_angle_2dp=round(s.golden, 2),
                    phi=round(s.phi, 3), turn_fraction=round(1 / s.phi ** 2, 3), full_turn=360,
                    golden_turn=round(1 / s.phi, 3), pi=round(math.pi, 5), pi_leftover=round(math.pi - 3, 5),
                    pi_leftover_deg=round(360 * (math.pi - 3), 2), one_seventh=round(1 / 7, 6),
                    seven_pi_turns=round(7 * math.pi, 2), seven_pi_miss_deg=round(360 * (22 - 7 * math.pi), 2))

    def draw(s, img, d, u, bi, bp, vis):
        ang = vis.get("angle", s.golden)
        if ang == "golden": ang = s.golden
        if "turn" in vis:  # fraction of a full turn per seed (number, or "pi" = pi turns, "golden" = 1/phi turns)
            tr = {"pi": math.pi, "golden": 1 / s.phi}.get(vis["turn"], vis["turn"]); ang = 360 * tr
        n = int(s.N * ease(clamp(bp / 0.75))) if vis.get("grow", True) else s.N
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
        for i in range(n):
            r = s.c * math.sqrt(i + 0.5); th = math.radians(i * ang)
            x, y = s.cx + r * math.cos(th), s.cy + r * math.sin(th)
            col = hsv(0.08 + 0.12 * i / s.N, 0.85, 1.0) if vis.get("warm", True) else hsv(i / s.N)
            rad = 7 + 6 * i / s.N
            gd.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col + (255,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(8))); img.alpha_composite(glow)
        d = ImageDraw.Draw(img, "RGBA")
        return d


class Pathfind:
    def __init__(s, P, seed):
        rng = np.random.default_rng(seed); C, R = P.get("cols", 39), P.get("rows", 37)
        g = np.ones((R, C), bool)  # True = wall
        stack = [(1, 1)]; g[1, 1] = False
        while stack:
            y, x = stack[-1]; nb = [(y + dy, x + dx, dy, dx) for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0)) if 0 < y + dy < R - 1 and 0 < x + dx < C - 1 and g[y + dy, x + dx]]
            if not nb: stack.pop(); continue
            ny, nx, dy, dx = nb[rng.integers(len(nb))]; g[y + dy // 2, x + dx // 2] = False; g[ny, nx] = False; stack.append((ny, nx))
        walls = [(y, x) for y in range(1, R - 1) for x in range(1, C - 1) if g[y, x] and ((g[y - 1, x] == g[y + 1, x] == False) != (g[y, x - 1] == g[y, x + 1] == False))]
        for k in rng.permutation(len(walls))[: int(len(walls) * P.get("loops", 0.12))]: g[walls[k]] = False
        s.g, s.R, s.C = g, R, C; s.start, s.goal = (1, 1), (R - 2, C - 2)
        s.bfs, s.bfs_path = s.search(False); s.ast, s.ast_path = s.search(True); s.event_u = 0
        s.cell = min((VX1 - VX0) / C, (VY1 - VY0) / R); s.ox = 540 - C * s.cell / 2; s.oy = (VY0 + VY1) / 2 - R * s.cell / 2
        base = Image.new("RGBA", (W, H), (0, 0, 0, 0)); bd = ImageDraw.Draw(base)
        for y in range(R):
            for x in range(C):
                if g[y, x]: bd.rectangle(s.box(y, x), fill=(36, 38, 64, 255))
        s.base = base

    def search(s, astar):
        import heapq
        h = (lambda p: abs(p[0] - s.goal[0]) + abs(p[1] - s.goal[1])) if astar else (lambda p: 0)
        pq = [(h(s.start), 0, 0, s.start)]; par = {s.start: None}; dist = {s.start: 0}; order = []; done = set(); cnt = 0
        while pq:
            f, _, gd, p = heapq.heappop(pq)
            if p in done: continue
            done.add(p); order.append(p)
            if p == s.goal: break
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                q = (p[0] + dy, p[1] + dx)
                if not s.g[q] and (q not in dist or gd + 1 < dist[q]):
                    dist[q] = gd + 1; par[q] = p; cnt += 1; heapq.heappush(pq, (gd + 1 + h(q), -gd - 1 if astar else cnt, gd + 1, q))
        path = []; p = s.goal
        while p: path.append(p); p = par[p]
        return order, path[::-1]

    def box(s, y, x):
        return [s.ox + x * s.cell, s.oy + y * s.cell, s.ox + (x + 1) * s.cell - 1, s.oy + (y + 1) * s.cell - 1]

    def stats(s):
        open_cells = int((~s.g).sum())
        return dict(open_cells=open_cells, bfs_explored=len(s.bfs), astar_explored=len(s.ast), path_len=len(s.bfs_path) - 1,
                    astar_path_len=len(s.ast_path) - 1, astar_pct_of_bfs=round(100 * len(s.ast) / len(s.bfs)),
                    saved_cells=len(s.bfs) - len(s.ast), bfs_pct_of_open=round(100 * len(s.bfs) / open_cells))

    def draw(s, img, d, u, bi, bp, vis):
        img.alpha_composite(s.base); d = ImageDraw.Draw(img, "RGBA")
        algo = vis.get("algo", "bfs"); order, path = (s.ast, s.ast_path) if algo == "astar" else (s.bfs, s.bfs_path)
        a = clamp(bp / 0.85) if vis.get("animate", True) else 1.0
        k = int(len(order) * a); hue0 = 0.5 if algo == "bfs" else 0.92
        for i, (y, x) in enumerate(order[:k]):
            d.rectangle(s.box(y, x), fill=hsv(hue0 + 0.35 * i / len(order), 0.7, 0.95) + (255,))
        if a >= 1 and vis.get("path", True):
            pts = [((s.box(y, x)[0] + s.box(y, x)[2]) / 2, (s.box(y, x)[1] + s.box(y, x)[3]) / 2) for y, x in path]
            m = max(2, int(len(pts) * clamp((bp - 0.85) / 0.12)))
            d.line(pts[:m], fill=(255, 255, 255, 255), width=max(4, int(s.cell * 0.4)), joint="curve")
        for (y, x), col in ((s.start, (80, 255, 140)), (s.goal, (255, 80, 110))):
            b = s.box(y, x); d.ellipse([b[0] - 4, b[1] - 4, b[2] + 4, b[3] + 4], fill=col + (255,))
        tc(d, (540, VY1 + 44), f"{'BFS' if algo == 'bfs' else 'A*'}  {k} cells checked", font(F_MONO, 36), al(TH["muted"], 0.9))
        return d


KINDS = dict(pendulum=Pendulum, galton=Galton, fourier=Fourier, sunflower=Sunflower, pathfind=Pathfind)


def make(S):
    P = dict(S.get("params", {})); P["_beats"] = S["beats"]
    return KINDS[S["kind"]](P, S.get("seed", 1))


# ============================================================ fact check
WORDNUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
           "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def numbers_in(text):
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text)]


def spoken(text):
    """Spoken integers: 'four thousand and ninety six' -> 4096, 'a hundred and thirty seven' -> 137."""
    out, total, cur, active = [], 0, 0, False
    for w in re.findall(r"[a-z]+|[.,:;!?]", text.lower()):
        if w in WORDNUM: cur += WORDNUM[w]; active = True
        elif w == "hundred": cur = (cur or 1) * 100; active = True
        elif w == "thousand": total += (cur or 1) * 1000; cur = 0; active = True
        elif w == "and" and active: continue
        elif w == "a" and not active: continue
        else:
            if active: out.append(total + cur)
            total, cur, active = 0, 0, False
    if active: out.append(total + cur)
    return [n for n in out if n not in (1, 2)]


def pool(st):
    p = set()
    def walk(x):
        if isinstance(x, bool): return
        if isinstance(x, (int, float)): p.update({float(x), float(round(x)), round(float(x), 1), round(float(x), 2)})
        elif isinstance(x, (list, tuple)): [walk(y) for y in x]
        elif isinstance(x, dict): [walk(v) for v in x.values()]
    walk(st); return p


def check(S):
    st = make(S).stats(); ok = pool(st) | {float(c) for c in S.get("constants", [])}
    for i in range(len(S["beats"])): ok.add(float(i + 1))
    bad = []
    def chk(where, txt, speak=False):
        if "—" in txt or "–" in txt: bad.append(f"{where}: em/en dash")
        for n in numbers_in(txt):
            if n not in ok: bad.append(f"{where}: number {n:g} not in stats/constants")
        if speak:
            for n in spoken(txt):
                if float(n) not in ok: bad.append(f"{where}: spoken number {n} not in stats/constants")
    for i, b in enumerate(S["beats"]):
        chk(f"beat {i} say", b.get("speak", b["say"]), True); chk(f"beat {i} cap", b["say"])
        for f in ("hook", "head"):
            for ln in b.get(f, []): chk(f"beat {i} {f}", ln)
        if b.get("chip"): chk(f"beat {i} chip", b["chip"])
        if b.get("card"): chk(f"beat {i} card", b["card"])
    for f in ("title", "caption"): chk(f, S.get(f, ""))
    if S["beats"][-1].get("nocap") is not True: bad.append("last beat must be the nocap follow line")
    print("\n".join(bad) if bad else "CHECK OK"); return not bad


# ============================================================ render
def umap(S, tl):
    """Sim time u for video time t: piecewise linear through (0,0) and each sync anchor (beat start -> u)."""
    def at(e): return tl[e[0]]["start"] + (e[1] * (tl[e[0]]["end"] - tl[e[0]]["start"]) if len(e) == 3 else 0.0)
    pts = [(0.0, 0.0)] + [(at(e), float(e[-1])) for e in S.get("sync", [])]  # [beat, u] or [beat, fraction_of_beat, u]
    if len(pts) == 1: pts.append((tl[-1]["end"], float(S.get("params", {}).get("u_end", tl[-1]["end"]))))
    def f(t):
        for (t0, u0), (t1, u1) in zip(pts, pts[1:]):
            if t <= t1: return u0 + (u1 - u0) * (t - t0) / max(t1 - t0, 1e-6)
        (t0, u0), (t1, u1) = pts[-2], pts[-1]; return u1 + (u1 - u0) / max(t1 - t0, 1e-6) * (t - t1)
    return f


def frame(t, S, tl, bg, sim, U, total):
    img = bg.copy().convert("RGBA"); d = ImageDraw.Draw(img, "RGBA")
    bi = 0
    for i, s in enumerate(tl):
        if t >= s["start"] - 0.1: bi = i
    seg = tl[bi]; b = seg["beat"]
    vb = next((tl[j]["beat"] for j in range(bi, -1, -1) if "vis" in tl[j]["beat"] or j == 0), b)
    vj = next((j for j in range(bi, -1, -1) if "vis" in tl[j]["beat"]), 0)
    v_end = next((tl[j]["start"] for j in range(vj + 1, len(tl)) if "vis" in tl[j]["beat"]), total)
    bp = clamp((t - tl[vj]["start"]) / max(v_end - tl[vj]["start"] - 0.25, 0.1))
    d = sim.draw(img, d, U(t), vj, bp, vb.get("vis", {})) or ImageDraw.Draw(img, "RGBA")
    head = next((tl[j]["beat"]["head"] for j in range(bi, -1, -1) if tl[j]["beat"].get("head")), None)
    if bi == 0 and b.get("hook"):
        g = clamp(t / 0.25); fh = font(F_BOLD, 96)
        wmax = max(d.textlength(ln, font=fh) for ln in b["hook"])
        if wmax > 960: fh = font(F_BOLD, int(96 * 960 / wmax))
        for i, ln in enumerate(b["hook"]):
            tc(d, (540, 130 + i * 100), ln, fh, al(YEL if i == 1 else TH["ink"], g))
    elif head:
        fh = font(F_BOLD, 84)
        for i, ln in enumerate(head):
            f2 = fh if d.textlength(ln, font=fh) < 960 else font(F_BOLD, int(84 * 960 / d.textlength(ln, font=fh)))
            tc(d, (540, 150 + i * 96), ln, f2, al(YEL if i else TH["ink"], 1))
    chip = b.get("chip")
    if chip:
        fc = font(F_BOLD, 44); w = d.textlength(chip, font=fc) + 56
        d.rounded_rectangle([540 - w / 2, 312, 540 + w / 2, 372], radius=30, fill=al(YEL, 1)); tc(d, (540, 342), chip, fc, (20, 16, 8, 255))
    if not b.get("nocap"):
        M.caption(d, seg, min(t, seg["end"]), TH)
    else:
        g = clamp((t - seg["start"]) / 0.3)
        if b.get("card"):  # series card (section F): episode name over the comment question, on a dark panel
            fq = font(F_BOLD, 64); words = b["say"].split(); lines = [""]
            for w_ in words:
                if d.textlength((lines[-1] + " " + w_).strip(), font=fq) > 900: lines.append(w_)
                else: lines[-1] = (lines[-1] + " " + w_).strip()
            pan = Image.new("RGBA", (W, H), (0, 0, 0, 0)); pd = ImageDraw.Draw(pan)
            pd.rounded_rectangle([50, 600, 1030, 830 + 80 * len(lines)], radius=40, fill=(8, 8, 14, int(225 * g)))
            img.alpha_composite(pan); d = ImageDraw.Draw(img, "RGBA")
            fcard = font(F_BOLD, 96); fcard = fcard if d.textlength(b["card"], font=fcard) < 920 else font(F_BOLD, int(96 * 920 / d.textlength(b["card"], font=fcard)))
            tc(d, (540, 700), b["card"], fcard, al(YEL, g))
            for k, ln in enumerate(lines): tc(d, (540, 830 + k * 80), ln, fq, al(TH["ink"], g))
        tc(d, (540, CAP_Y), "Follow @shipwithpip", font(F_BOLD, 80), al(TH["ink"], g))
    tc(d, (540, 1840), "@shipwithpip", font(F_MED, 34), al(TH["muted"], 0.8))
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
    S, tl, vox, total = prepare(S, model_dir); sim = make(S); U = umap(S, tl); bg = M.make_bg(TH)
    if only is not None:
        ts = only if only else [0.5] + [x["start"] + (x["end"] - x["start"]) * 0.85 for x in tl]
        return [frame(t, S, tl, bg, sim, U, total) for t in ts], tl, total
    audio = M.mix(vox, total, tl, [s["start"] for s in tl], sum(map(ord, S["id"])))
    wav = out[:-4] + ".wav"; sf.write(wav, audio, SR)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-af", "loudnorm=I=-14:TP=-1.2:LRA=11", "-c:a", "aac", "-b:a", "192k", "-shortest",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for fr in range(int(total * FPS)):
        if fr % 150 == 0: print(f"frame {fr} / {int(total * FPS)}", flush=True)
        ff.stdin.write(frame(fr / FPS, S, tl, bg, sim, U, total).convert("RGB").tobytes())
    ff.stdin.close(); ff.wait(); os.remove(wav)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("script"); ap.add_argument("out", nargs="?")
    ap.add_argument("--model"); a = ap.parse_args(); S = json.load(open(a.script, encoding="utf-8"))
    if a.cmd == "stats": print("STATS", json.dumps(make(S).stats()))
    elif a.cmd == "check": sys.exit(0 if check(S) else 1)
    elif a.cmd == "sheet":
        frames, tl, total = render(S, a.out, a.model, only=[])
        sh = Image.new("RGB", (270 * 5, 480 * math.ceil(len(frames) / 5)))
        for i, f in enumerate(frames): sh.paste(f.convert("RGB").resize((270, 480)), ((i % 5) * 270, (i // 5) * 480))
        sh.save(a.out); print(a.out, f"{total:.1f}s")
    elif a.cmd == "render":
        total = render(S, a.out, a.model); print(f"done {a.out}: {total:.1f}s")
