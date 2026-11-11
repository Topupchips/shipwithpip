"""Evolving race cars and a wall of pole-balancing AIs.
usage: python drivefx.py cars|cartpole out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evosim as E  # noqa: E402
from learnfx import write_post  # noqa: E402
from scenes import CYAN, GREEN, PINK, VIOLET, YEL, ORANGE, Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, INK, VS, clamp, font, tone  # noqa: E402

BG = (10, 6, 28)


def wins_at(wins, t):
    for k, (a, b, gi) in enumerate(wins):
        if a <= t < b:
            return k, gi, (t - a) / (b - a)
    if t < wins[0][0]:
        return -1, wins[0][2], 0.0
    return len(wins) - 1, wins[-1][2], 1.0


# ------------------------------------------------------------------ cars
class CarReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.H = cfg["hist"]
        yy, xx = np.mgrid[0:VS, 0:VS]
        base = np.zeros((VS, VS, 3), float) + np.array(BG)
        base += (np.clip(1 - np.hypot(xx - 480, yy - 480) / 640, 0, 1) ** 2 * 45)[..., None] * np.array([0.3, 1.0, 0.7])
        self.base = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(self.base, "RGBA")
        pts = [tuple(p) for p in E.CL] + [tuple(E.CL[0])]
        d.line(pts, fill=(130, 120, 220, 70), width=2 * E.HW + 14, joint="curve")
        d.line(pts, fill=(52, 46, 96, 255), width=2 * E.HW, joint="curve")
        d.line(pts, fill=(255, 255, 255, 40), width=2, joint="curve")
        sx, sy = E.START
        d.line([(sx - 8, sy - E.HW), (sx - 8, sy + E.HW)], fill=(255, 255, 255, 230), width=5)
        self.d_fin = None

    def content(self, t):
        k, gi, q = wins_at(self.cfg["wins"], t)
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        if k < 0:
            return im
        h = self.H[gi]
        a, b, _ = self.cfg["wins"][k]
        frac = clamp((t - a) / (b - a - 0.4))
        st = min(h["P"].shape[0] - 1, int(frac * (h["P"].shape[0] - 1)))
        order = np.argsort(-h["prog"])
        best = int(order[0])
        P, A = h["P"], h["A"]
        for i in range(P.shape[1]):
            x, y, ang = P[st, i]
            al = A[st, i]
            if i == best:
                continue
            col = (120, 235, 255) if al else (255, 90, 120)
            if al:
                self.car(d, x, y, ang, col, 255, 1.0)
            else:
                d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 70, 100, 120))
        x, y, ang = P[st, best]
        tr = [(float(P[s, best, 0]), float(P[s, best, 1])) for s in range(max(0, st - 40), st + 1, 2)]
        if len(tr) > 1 and A[st, best]:
            d.line(tr, fill=YEL + (130,), width=5, joint="curve")
        self.car(d, x, y, ang, YEL if A[st, best] else (255, 90, 120), 255, 1.25)
        return im

    @staticmethod
    def car(d, x, y, ang, col, a, s):
        c, sn = math.cos(ang), math.sin(ang)
        L, Wd = 24 * s, 13 * s
        pts = [(x + c * L - sn * Wd * (1 if j else -1) * (1 if i < 2 else 1), 0) for i, j in ()]
        corners = [(L, Wd), (L, -Wd), (-L, -Wd), (-L, Wd)]
        poly = [(x + c * u - sn * v, y + sn * u + c * v) for u, v in corners]
        d.polygon(poly, fill=col + (a,), outline=INK + (255,))

    def _gen(self, t):
        k, gi, q = wins_at(self.cfg["wins"], t)
        return k, gi

    def stats(self, t):
        k, gi = self._gen(t)
        if k < 0:
            return [("40 CARS", YEL), ("GEN 1", CYAN)], 0.0, "CARS STILL ON THE ROAD", CYAN
        h = self.H[gi]
        a, b, _ = self.cfg["wins"][k]
        frac = clamp((t - a) / (b - a - 0.4))
        st = min(h["P"].shape[0] - 1, int(frac * (h["P"].shape[0] - 1)))
        alive = int(h["A"][st].sum())
        fin = int((h["prog"] >= 2.0).sum())
        n = h["P"].shape[1]
        return [(f"GEN {gi + 1}", YEL), (f"{alive}/{n} DRIVING", CYAN if alive else PINK)], alive / n, "CARS STILL ON THE ROAD", CYAN


def build_cars(model):
    seeds = range(1, 7)
    reach = []
    for s in seeds:
        Hs = E.evolve_cars(seed=s, gens=8)
        reach.append(next((g + 1 for g, h in enumerate(Hs) if (h["prog"] >= 2.0).sum() >= 10), None))
    print("gens to get 10+ cars through 2 laps, seeds 1-6:", reach)
    ok = sum(1 for r in reach if r is not None)
    hist = E.evolve_cars(seed=1, gens=12)
    fin = [int((h["prog"] >= 2.0).sum()) for h in hist]
    best = [round(float(h["prog"].max()), 2) for h in hist]
    print("finishers by gen", fin, "best laps", best)
    show = [0, 1, 2, 3, 4, 7]
    durs = [5.4, 3.4, 3.4, 3.4, 3.4, 3.4]
    t = 4.6
    wins = []
    for gi, du in zip(show, durs):
        wins.append((t, t + du, gi))
        t += du
    t_end = t
    r = lambda i: fin[show[i]]  # noqa: E731
    pay = round(wins[-1][0] + 1.6, 1)
    cfg = dict(T=48.0, c1=CYAN, c2=YEL, hist=hist, wins=wins, payoff=pay, seed=151,
               still_times=[0.5, 5.5, 11, 15, 19, 23, 27, t_end + 1],
               header=lambda tt: (([("40 DUMB CARS.", (255, 255, 255)), ("1 TRACK.", YEL)], "WATCH THEM EVOLVE") if tt < wins[-1][0] else
                                  ([("NOBODY TAUGHT", (255, 255, 255)), ("THEM TO DRIVE.", GREEN)], "NATURAL SELECTION")))
    lines = [(0.2, "Forty dumb cars. One track. None of them know how to drive."),
             (4.6, f"Generation one. Random brains. {r(0)} of forty drive two laps."),
             (10.2, "Each generation, the best drivers pass on their brain, with small random mutations."),
             (17.2, f"Generation {show[3] + 1}. {r(3)} of forty."),
             (20.8, f"Generation {show[4] + 1}. {r(4)} of forty drive two full laps."),
             (24.6, "Nobody wrote the driving code. Each brain is just five distance sensors and a few numbers."),
             (30.6, f"Generation {show[5] + 1}. {r(5)} of forty."),
             (34.0, "Follow for more, explained in motion.")]
    buf, vo = make_audio(cfg, lines, model)
    for k_, (a, b, gi) in enumerate(wins):
        tone(buf, a, 0.2, 380 + 30 * k_, 520 + 30 * k_, 0.05)
    chime(buf, cfg["payoff"])
    cfg["T"] = vo[-1][1] + 2.4
    reel = CarReel(cfg, vo)
    post = dict(title="DRIVE-1  40 dumb cars learn to drive by natural selection",
                caption=f"40 cars with random brains, one track. Each brain is a tiny neural network that reads 5 distance sensors and decides how much to steer. Each generation, the best drivers pass on their brain with small random mutations. Generation 1: {r(0)} of 40 drive 2 laps. Generation {show[4] + 1}: {r(4)} of 40. Nobody wrote the driving code.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment CARS and I'll send the prompt so you can build it.",
                tags=["ai", "neuralnetwork", "naturalselection", "simulation", "coding", "shipwithpip"],
                pinned="Build me a single-file HTML canvas page where 40 cars evolve to drive a closed track. Each car has 5 distance sensors (rays) and a tiny neural net (5 inputs, 6 tanh hidden, 2 outputs: steering and speed). Score each car by distance driven around the track before it leaves the road. Each generation keep the top 6, copy them with Gaussian mutation, show generation number and how many cars are still on the road.",
                notes=f"Genetic algorithm on tiny neural nets: 40 cars, 5 distance sensors (rays up to 168 px), 5-6-2 tanh network (NW={E.NW} weights), steering turn 0.09 rad/step max, speed ~2.8-4 px/step, 1000 steps per generation (about 2.2 laps if the car never crashes). Selection: top 6 by laps completed, top 2 kept unchanged, rest mutated (30% of weights get N(0,0.25) noise). Initial weights N(0,0.25).\nCars finishing 2 laps per generation (gen 1-12): {fin}.\nBest laps per generation: {best}.\nRobustness: over seeds 1-6, 10+ cars reached 2 laps within 8 generations in {ok} of 6 runs (generation numbers: {reach}).\nThis is a simulation I wrote (not footage). Reels shows generations {[s + 1 for s in show]}.")
    return reel, buf, cfg, post


# ------------------------------------------------------------------ cartpole wall
class CartReel(Reel2):
    COLS, ROWS = 6, 5

    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.H = cfg["hist"]

    def content(self, t):
        k, gi, q = wins_at(self.cfg["wins"], t)
        im = Image.new("RGB", (VS, VS), BG)
        d = ImageDraw.Draw(im, "RGBA")
        if k < 0:
            h = self.H[0]
            st = 0
        else:
            h = self.H[gi]
            a, b, _ = self.cfg["wins"][k]
            frac = clamp((t - a) / (b - a - 0.5))
            st = min(h["traj"].shape[0] - 1, int(frac * (h["traj"].shape[0] - 1)))
        cw, ch = VS // self.COLS, VS // self.ROWS
        fonts = font(F_MONO, 22)
        for i in range(h["traj"].shape[1]):
            r_, c_ = divmod(i, self.COLS)
            x0, y0 = c_ * cw, r_ * ch
            al = h["A"][st, i]
            life = int(h["life"][i])
            if al:
                fill, edge = (18, 40, 52, 255), (80, 235, 160, 220)
            else:
                fill, edge = (52, 18, 36, 255), (255, 80, 110, 220)
            d.rounded_rectangle([x0 + 5, y0 + 5, x0 + cw - 5, y0 + ch - 5], radius=14, fill=fill, outline=edge, width=3)
            x = float(h["traj"][min(st, life), i, 0]) if not al else float(h["traj"][st, i, 0])
            th = float(h["traj"][min(st, max(life - 1, 0)) if not al else st, i, 1])
            gx = x0 + cw / 2 + x / 2.4 * (cw / 2 - 30)
            gy = y0 + ch - 48
            d.line([(x0 + 14, gy + 14), (x0 + cw - 14, gy + 14)], fill=(255, 255, 255, 50), width=2)
            col = (120, 240, 255) if al else (255, 110, 140)
            d.rounded_rectangle([gx - 22, gy - 9, gx + 22, gy + 13], radius=5, fill=col + (255,), outline=INK + (255,), width=2)
            L = 76
            px, py = gx + L * math.sin(th), gy - 9 - L * math.cos(th)
            d.line([(gx, gy - 9), (px, py)], fill=(255, 224, 90, 255) if al else (255, 150, 170, 255), width=7)
            d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(255, 255, 255, 255), outline=INK + (255,), width=2)
            if not al:
                d.text((x0 + cw / 2, y0 + 26), f"{life} steps", font=fonts, fill=(255, 150, 170, 255), anchor="mm")
            elif life >= 500 and st >= 499:
                d.text((x0 + cw / 2, y0 + 26), "500 steps", font=fonts, fill=(130, 255, 190, 255), anchor="mm")
        return im

    def stats(self, t):
        k, gi, q = wins_at(self.cfg["wins"], t)
        if k < 0:
            return [("30 AIs", YEL), ("GEN 1", CYAN)], 1.0, "STILL BALANCING", CYAN
        h = self.H[gi]
        a, b, _ = self.cfg["wins"][k]
        frac = clamp((t - a) / (b - a - 0.5))
        st = min(h["traj"].shape[0] - 1, int(frac * (h["traj"].shape[0] - 1)))
        alive = int(h["A"][st].sum())
        n = h["A"].shape[1]
        return [(f"GEN {gi + 1}", YEL), (f"{alive}/{n} BALANCING", CYAN if alive else PINK)], alive / n, "AIs STILL BALANCING", CYAN


def build_cart(model):
    seed = 22
    hist = E.evolve_cart(seed=seed, gens=14, n=30)
    n500 = [int((h["life"] >= 500).sum()) for h in hist]
    mean = [int(h["life"].mean()) for h in hist]
    print("n500", n500, "mean", mean)
    # random-policy rate
    rng = np.random.default_rng(5)
    tot = ok = 0
    for tr in range(5):
        W = rng.uniform(-1, 1, (300, 4))
        _, _, life = E.cart_run(W, seed=tr)
        tot += 300
        ok += int((life >= 500).sum())
    rate = ok / tot * 100
    print("random policy solve rate %", rate)
    show = [0, 1, 3, 5, 9]
    durs = [5.0, 3.6, 3.6, 3.6, 3.6]
    t = 4.4
    wins = []
    for gi, du in zip(show, durs):
        wins.append((t, t + du, gi))
        t += du
    t_end = t
    r = lambda i: n500[show[i]]  # noqa: E731
    life0 = hist[0]["life"]
    pay = round(wins[-1][0] + 1.4, 1)
    cfg = dict(T=40.0, c1=CYAN, c2=PINK, hist=hist, wins=wins, payoff=pay, seed=161,
               still_times=[0.5, 5.2, 9.5, 13, 17, 21, t_end + 1],
               header=lambda tt: (([("30 AIs. 1 POLE.", (255, 255, 255)), ("DON'T LET IT FALL.", YEL)], "WATCH THEM LEARN") if tt < wins[-1][0] else
                                  ([("NOBODY TOLD", (255, 255, 255)), ("THEM HOW.", GREEN)], "TRIAL, ERROR, REPEAT")))
    lines = [(0.2, "Thirty AIs. One pole on a cart. Keep it upright."),
             (4.4, f"Generation one. No idea. The best lasts {int(life0.max())} steps of five hundred."),
             (10.0, "The ones that last longest pass on their settings, with small random changes."),
             (15.2, f"Generation {show[2] + 1}. {r(2)} of thirty keep it up the whole time."),
             (19.0, f"Generation {show[3] + 1}. {r(3)} of thirty."),
             (22.6, f"Generation {show[4] + 1}. {r(4)} of thirty. Each one is just four numbers."),
             (27.0, "Follow for more, explained in motion.")]
    buf, vo = make_audio(cfg, lines, model)
    for k_, (a, b, gi) in enumerate(wins):
        tone(buf, a, 0.2, 380 + 30 * k_, 520 + 30 * k_, 0.05)
    chime(buf, cfg["payoff"])
    cfg["T"] = vo[-1][1] + 2.4
    reel = CartReel(cfg, vo)
    post = dict(title="DRIVE-2  30 AIs learn to balance a pole",
                caption=f"30 AIs try to keep a pole balanced on a cart. Each AI is just 4 numbers that decide push left or push right. Each generation, the ones that last longest pass on their numbers with small random changes. Generation 1: best lasts {int(life0.max())} of 500 steps. Generation {show[2] + 1}: {r(2)} of 30 balance it the whole time. Nobody told them how.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment POLE and I'll send the prompt so you can build it.",
                tags=["ai", "reinforcementlearning", "physics", "simulation", "coding", "shipwithpip"],
                pinned="Build me a single-file HTML page that shows 30 cart-pole simulations at once. Use the standard CartPole physics (gravity 9.8, cart mass 1, pole mass 0.1, half-pole length 0.5, force 5 N, dt 0.02, fail beyond 12 degrees or 2.4 units). Each AI is 4 weights: push right if w dot [x, xdot, theta, thetadot] is above 0. Each generation keep the top 5 by steps survived (max 500), copy them with Gaussian noise 0.4, and show how many are still balancing.",
                notes=f"Standard CartPole dynamics (Barto, Sutton, Anderson 1983; as in OpenAI Gym): gravity 9.8, cart 1.0, pole 0.1, half-length 0.5, dt 0.02, failure at |theta|>12 deg or |x|>2.4, episode cap 500 steps (10 s). I used a smaller push force (5 N instead of Gym's 10 N) so random policies fail more often.\nPolicy: linear, push right if w . state > 0 (4 weights). Evolution: 30 policies, top 5 kept as parents, 1 elite kept unchanged, others = parent + N(0, 0.4) noise.\nPolicies reaching 500 steps per generation (gen 1-14, seed {seed}): {n500}.\nMean steps survived per generation: {mean}.\nA random policy solves this in about {rate:.1f}% of cases (1500 random policies tested), so with 30 in generation 1 it is luck whether any succeeds; I used a seed where none did and the climb is similar across seeds I tried.\nSimulation only; video plays 500 steps (10 s real time) at about 2.5x speed.")
    return reel, buf, cfg, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = {"cars": build_cars, "cartpole": build_cart}[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
