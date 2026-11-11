"""Dumb-AI-vs-goal reels: evolving rockets, ants finding the short path.
usage: python goalfx.py rockets|ants out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import goalsim as GS  # noqa: E402
from learnfx import write_post  # noqa: E402
from scenes import CYAN, GREEN, PINK, VIOLET, YEL, ORANGE, Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, INK, VS, clamp, ease_out, font, tone  # noqa: E402

BG = (10, 6, 28)


def evolve(seed=3, gens=46):
    rng = np.random.default_rng(seed)

    def rv(n):
        a = rng.uniform(0, 2 * np.pi, n)
        return np.stack([np.cos(a), np.sin(a)], -1)
    NR, NG = GS.NR, GS.NG
    pop = rv(NR * NG).reshape(NR, NG, 2)
    hist = []
    for g in range(gens):
        P, ST, fit = GS.rocket_run(pop)
        reached = int((ST[-1] == 2).sum())
        hist.append(dict(P=P.astype(np.float32), ST=ST.astype(np.int8), reached=reached, crashed=int((ST[-1] == 1).sum())))
        order = np.argsort(-fit)
        parents = order[:20]
        mut = 0.03 if reached < 20 else 0.012
        newp = [pop[i].copy() for i in order[:5]]
        while len(newp) < NR:
            a, b = rng.choice(parents, 2)
            cut = rng.integers(1, NG)
            child = np.concatenate([pop[a][:cut], pop[b][cut:]])
            m = rng.random(NG) < mut
            child[m] = rv(int(m.sum()))
            newp.append(child)
        pop = np.array(newp)
    return hist


class RocketReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.H = cfg["hist"]
        yy, xx = np.mgrid[0:VS, 0:VS]
        base = np.zeros((VS, VS, 3), float) + np.array(BG)
        base += (np.clip(1 - np.hypot(xx - 480, yy - 90) / 520, 0, 1) ** 2 * 60)[..., None] * np.array([0.2, 1.0, 0.6])
        grid = ((xx % 60 < 2) | (yy % 60 < 2))
        base[grid] += 10
        self.base = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(self.base, "RGBA")
        for (x0, y0, x1, y1) in GS.WALLS:
            d.rounded_rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], radius=14, fill=(255, 76, 170, 70))
            d.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(120, 40, 140, 255), outline=(255, 140, 220, 255), width=3)
        d.rounded_rectangle([430, 880, 530, 910], radius=10, fill=(60, 50, 120, 255), outline=(255, 255, 255, 120), width=2)

    def win(self, t):
        for k, (a, b, gi) in enumerate(self.cfg["wins"]):
            if a <= t < b:
                return k, gi, (t - a) / (b - a)
        a, b, gi = self.cfg["wins"][-1]
        if t < self.cfg["wins"][0][0]:
            return -1, 0, 0.0
        return len(self.cfg["wins"]) - 1, gi, 1.0

    def step_at(self, t):
        k, gi, q = self.win(t)
        run = self.cfg["wins"][k][2] if k >= 0 else 0
        a, b, _ = self.cfg["wins"][max(k, 0)]
        frac = clamp((t - a) / (b - a - 0.5)) if k >= 0 else 0.0
        return k, gi, int(frac * GS.L)

    def content(self, t):
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        pul = 0.5 + 0.5 * math.sin(t * 4)
        gx, gy = GS.GOAL
        for r, a in ((GS.GR + 28 + 8 * pul, 50), (GS.GR + 12, 90), (GS.GR, 255)):
            d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=GREEN + (a,), width=5 if r == GS.GR else 3)
        d.text((gx, gy), "GOAL", font=font(F_MONO, 26), fill=GREEN + (255,), anchor="mm")
        k, gi, step = self.step_at(t)
        if k >= 0:
            h = self.H[gi]
            P, ST = h["P"], h["ST"]
            s = min(step, GS.L)
            for i in range(GS.NR):
                st = ST[s, i]
                x, y = P[s, i]
                if st == 1:
                    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 70, 90, 120))
                    continue
                if st == 2:
                    col = GREEN
                else:
                    col = (255, 255, 255)
                # trail
                pts = [tuple(P[max(0, s - j), i]) for j in range(10, -1, -2)]
                d.line(pts, fill=(CYAN if st == 0 else GREEN) + (110,), width=4)
                px, py = P[max(0, s - 1), i]
                ang = math.atan2(y - py, x - px) if (x != px or y != py) else -math.pi / 2
                tip = (x + math.cos(ang) * 14, y + math.sin(ang) * 14)
                l = (x + math.cos(ang + 2.5) * 9, y + math.sin(ang + 2.5) * 9)
                r_ = (x + math.cos(ang - 2.5) * 9, y + math.sin(ang - 2.5) * 9)
                d.polygon([tip, l, r_], fill=col + (255,), outline=(255, 255, 255, 255))
            # label gen
            d.text((30, 44), "GENERATION", font=font(F_MONO, 28), fill=(255, 255, 255, 230), anchor="lm", stroke_width=5, stroke_fill=INK)
            d.text((30, 100), f"{gi + 1}", font=font(F_BOLD, 96), fill=YEL + (255,), anchor="lm", stroke_width=8, stroke_fill=INK)
        return im

    def stats(self, t):
        k, gi, step = self.step_at(t)
        if k < 0:
            return [("GENERATION 1", YEL), ("0 / 100 REACHED", CYAN)], 0.0, "ROCKETS THAT REACHED THE GOAL", GREEN
        h = self.H[gi]
        s = min(step, GS.L)
        reached = int((h["ST"][s] == 2).sum())
        return [(f"GEN {gi + 1}", YEL), (f"{reached} / 100 REACHED", CYAN)], reached / 100, "ROCKETS THAT REACHED THE GOAL", GREEN


def build_rockets(model):
    hist = evolve()
    reached = [h["reached"] for h in hist]
    print("reached by gen:", reached)
    best_g = int(np.argmax(reached[:46]))
    show = [0, 2, 5, 9, 12, 15, 19, best_g]
    show = sorted(set(show))
    durs = [5.4] + [2.7] * (len(show) - 1)
    t = 5.0
    wins = []
    for gi, du in zip(show, durs):
        wins.append((t, t + du, gi))
        t += du
    t_end = t
    print("shown gens", [s + 1 for s in show], "reached", [reached[s] for s in show], "ends", t_end)
    r = lambda g: reached[g]  # noqa: E731
    cfg = dict(T=36.5, c1=VIOLET, c2=GREEN, hist=hist, wins=wins, payoff=round(wins[-1][0] + 1.9, 1), seed=101,
               still_times=[0.5, 5, 9.5, 13, 16, 20, 24, t_end + 1],
               header=lambda tt: (([("100 DUMB AIs.", (255, 255, 255)), ("1 GOAL.", YEL)], "WATCH THEM EVOLVE") if tt < wins[-1][0] else
                                  ([("NOBODY TOLD", (255, 255, 255)), ("THEM THE ROUTE.", GREEN)], "EVOLUTION FOUND IT")))
    w = [x[0] for x in wins]
    gs = [x[2] + 1 for x in wins]
    lines = [(0.2, "A hundred dumb rockets. One goal. They have no idea how to get there."),
             (5.2, f"Generation one. Random thrust. {r(show[0])} of a hundred make it."),
             (10.4, "Each generation, the ones that got closest pass on their thrust, with small random mutations."),
             (18.6, f"Generation {gs[4]}. {r(show[4])} of a hundred."),
             (21.8, f"Generation {gs[5]}. {r(show[5])} of a hundred reach the goal."),
             (25.4, "Nobody programmed the route. Evolution found it."),
             (28.6, f"Generation {gs[-1]}. {r(show[-1])} of a hundred."),
             (32.0, "Follow for more, explained in motion.")]
    buf, vo = make_audio(cfg, lines, model)
    for k_, (a, b, gi) in enumerate(wins):
        tone(buf, a, 0.2, 380 + 30 * k_, 520 + 30 * k_, 0.05)
    chime(buf, cfg["payoff"])
    reel = RocketReel(cfg, vo)
    post = dict(title="GOAL-1  100 dumb rockets vs 1 goal (they evolve)",
                caption=f"100 rockets with random thrust, one goal behind two walls. Each generation, the ones that got closest pass on their thrust patterns with small random mutations. Generation 1: {r(0)} of 100 reach the goal. Generation {gs[-1]}: {r(show[-1])} of 100. Nobody programmed the route.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment ROCKET and I'll send the prompt so you can build it.",
                tags=["ai", "geneticalgorithm", "evolution", "simulation", "coding", "shipwithpip"],
                pinned="Build me a single-file HTML canvas page where 100 rockets evolve to reach a goal behind two walls. Each rocket has a genome of 64 thrust directions (each held for 5 steps). Each generation: score by closest distance to the goal (bonus for reaching it), keep the top 5 unchanged, breed the rest from the top 20 with single-point crossover and 3% mutation. Show the generation number and how many reached the goal.",
                notes=f"Genetic algorithm: 100 rockets, genome = 64 random unit thrust vectors (5 steps each), 320 steps per run, two walls, goal radius {GS.GR:.0f}px. Selection: top 20 as parents, elitism 5, single-point crossover, mutation 3% (1.2% once 20+ reach the goal).\nReached the goal per generation (all shown): {[(gs[i], r(show[i])) for i in range(len(show))]}.\nFull history (gen 1-46): {reached}.\nResults fluctuate between generations (mutation breaks good genomes sometimes); this is one seeded run.")
    return reel, buf, cfg, post


# ------------------------------------------------------------------ ants
class AntReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.fr = cfg["frames"]
        self.short, self.long_ = GS.ant_paths()
        yy, xx = np.mgrid[0:VS, 0:VS]
        base = np.zeros((VS, VS, 3), float) + np.array([22, 16, 12.0])
        rng = np.random.default_rng(3)
        base += rng.normal(0, 4, (VS, VS, 1))
        base += (np.clip(1 - np.hypot(xx - 480, yy - 480) / 700, 0, 1) ** 1.5 * 24)[..., None] * np.array([1.0, 0.8, 0.5])
        self.base = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(self.base, "RGBA")
        # rocks in the middle between the branches
        for (x, y, r) in ((470, 570, 90), (330, 520, 54), (620, 540, 66), (480, 470, 40)):
            d.ellipse([x - r, y - r, x + r, y + r], fill=(48, 40, 52, 255), outline=(90, 80, 100, 255), width=3)
        d.ellipse([60, 420, 180, 540], fill=(110, 70, 40, 255), outline=(190, 130, 80, 255), width=4)
        d.text((120, 480), "NEST", font=font(F_MONO, 28), fill=(255, 235, 200, 255), anchor="mm")
        d.ellipse([780, 420, 900, 540], fill=(40, 120, 70, 255), outline=(120, 255, 170, 255), width=4)
        d.text((840, 480), "FOOD", font=font(F_MONO, 28), fill=(235, 255, 235, 255), anchor="mm")
        for (x, y) in ((820, 450), (860, 470), (835, 505), (870, 440)):
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 90, 120, 255))

    def content(self, t):
        i = int(min(len(self.fr) - 1, max(0, (t - 0.4) * 30)))
        f = self.fr[i]
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        for idx, (pts, col) in enumerate(((self.short, CYAN), (self.long_, ORANGE))):
            tau = f["tau"][idx]
            w = 5 + min(40, tau * 0.35)
            d.line(pts, fill=col + (int(40 + min(120, tau * 1.2)),), width=int(w + 14), joint="curve")
            d.line(pts, fill=col + (int(110 + min(145, tau * 1.5)),), width=int(w), joint="curve")
        for (path, dr, s) in f["ants"]:
            pts = self.short if path == 0 else self.long_
            x, y = GS.point_on(pts, s)
            col = (255, 255, 255) if dr == 1 else (255, 230, 120)
            d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=col + (255,), outline=INK + (255,), width=2)
        d.text((480, 290), f"SHORT PATH", font=font(F_MONO, 26), fill=CYAN + (255,), anchor="mm", stroke_width=4, stroke_fill=INK)
        d.text((480, 880), f"LONG PATH", font=font(F_MONO, 26), fill=ORANGE + (255,), anchor="mm", stroke_width=4, stroke_fill=INK)
        return im

    def stats(self, t):
        i = int(min(len(self.fr) - 1, max(0, (t - 0.4) * 30)))
        a = self.fr[i]["ants"]
        frac = sum(1 for x in a if x[0] == 0) / max(1, len(a))
        return [(f"{len(a)} ANTS", YEL), (f"SHORT PATH {frac * 100:0.0f}%", CYAN)], frac, "ANTS ON THE SHORT PATH", CYAN


def build_ants(model):
    # robustness over many seeds
    wins = 0
    for s in range(1, 31):
        fr, hs, _ = GS.ants_sim(seed=s, T=34.0)
        wins += hs[-1] > 0.8
    print("runs ending >80% on short path:", wins, "of 30")
    seed = 2
    fr, hs, (Ls, Ll) = GS.ants_sim(seed=seed, T=34.0)
    print("short/long len", Ls, Ll, "fractions", [round(hs[int(x * 30)], 2) for x in (1, 5, 10, 15, 20, 25, 30)])
    tc = next((i / 30 for i, x in enumerate(hs) if i > 150 and x > 0.85), 26.0) + 0.4
    f = lambda x: hs[min(len(hs) - 1, int((x - 0.4) * 30))] * 100  # noqa: E731
    cfg = dict(T=39.0, c1=ORANGE, c2=CYAN, frames=fr, payoff=round(tc, 1), seed=111, still_times=[0.6, 5, 10, 15, 20, 25, 30, 36],
               header=lambda t: (([("NO ANT KNOWS", (255, 255, 255)), ("THE SHORT WAY.", YEL)], "100 DUMB ANTS") if t < tc else
                                 ([("THEY FOUND IT", (255, 255, 255)), ("ANYWAY.", GREEN)], "NO LEADER. NO MAP.")))
    lines = [(0.2, "A hundred dumb ants. Food on one side, nest on the other. Two paths: one short, one long."),
             (6.8, "No ant knows which is shorter. Each one just leans toward the path with more scent."),
             (12.2, f"Early on, only {f(10.5):0.0f} percent pick the short path."),
             (16.0, "But ants on the short path finish their trips sooner, so they lay scent more often."),
             (21.2, "More scent, more ants. It snowballs."),
             (24.4, f"Now {f(26.0):0.0f} percent take the short path. Nobody told them."),
             (28.6, f"Real Argentine ants do this too. In my sim it worked in {wins} of 30 runs."),
             (33.4, "Follow for more.")]
    buf, vo = make_audio(cfg, lines, model)
    chime(buf, cfg["payoff"])
    cfg["T"] = vo[-1][1] + 2.4
    reel = AntReel(cfg, vo)
    post = dict(title="GOAL-2  100 dumb ants vs the shortest path",
                caption=f"100 ants, two paths to the food, and no ant knows which is shorter. Each ant leans toward the path with more scent, and the short path gets scent faster because trips finish sooner. More scent, more ants, and it snowballs. In this run ~{hs[-1] * 100:.0f}% end up on the short path. In my simulation it worked in {wins} of 30 random runs. Real Argentine ants show the same effect (Goss et al., 1989).\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment ANTS and I'll send the prompt so you can build it.",
                tags=["ai", "swarmintelligence", "emergence", "simulation", "coding", "shipwithpip"],
                pinned="Build me a single-file HTML canvas page that simulates 100 ants choosing between a short and a long path from a nest to food. Each ant picks a branch with probability proportional to (8 + pheromone)^2, adds 1 pheromone to its branch when it finishes a trip, and pheromone evaporates slowly. Draw each path's thickness by its pheromone and show the percent of ants on the short path.",
                notes=f"Double-bridge style model: branch choice p ~ (k+tau)^2 with k=8, deposit 1 per completed trip, evaporation 3%/s, ant speed 330 px/s, path lengths {Ls:.0f} vs {Ll:.0f} px, up to 100 ants (spawned 8/s). Over 30 seeds, {wins} ended with more than 80% of ants on the short path (the rest got stuck mixed or on the long path, which also happens in the literature). This displayed run: seed {seed}.\nSource: Goss, Aron, Deneubourg, Pasteels (1989) 'Self-organized shortcuts in the Argentine ant', Naturwissenschaften 76: 579-581 (https://link.springer.com/article/10.1007/BF00462870). The Deneubourg-style model is a simplified model, not a claim about exact ant behavior.")
    return reel, buf, cfg, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = {"rockets": build_rockets, "ants": build_ants}[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
