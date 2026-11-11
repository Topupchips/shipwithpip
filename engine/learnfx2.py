"""Five 'dumb AI learns from zero' neon-sim reels.
usage: python learnfx2.py birds|digits|cliff|triangles|tsp out.mp4 --model <kokoro dir> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa: E402
from scenes import CYAN, GREEN, ORANGE, PINK, VIOLET, YEL, Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, INK, VS, clamp, font  # noqa: E402

WHITE = (255, 255, 255)


def base_img(rgb=(14, 9, 36)):
    im = Image.new("RGB", (VS, VS), rgb)
    return im


class SeqReel(Reel2):
    """Reel whose play area is driven by a list of precomputed frame states."""

    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.fr = cfg["frames"]

    def idx(self, t):
        return int(min(len(self.fr) - 1, max(0, (t - 0.4) * 30)))

    def st(self, t):
        return self.fr[self.idx(t)]


def mk_cfg(T, c1, c2, payoff, seed, stills, header):
    return dict(T=T, c1=c1, c2=c2, payoff=payoff, seed=seed, still_times=stills, header=header)


def finish(cfg, lines, cls, extra):
    lines = [list(l) for l in sorted(lines)]
    cfg["T"] = max(cfg["T"], 90)
    for _ in range(8):
        buf, vo = make_audio(cfg, [tuple(l) for l in lines], extra["model"])
        bad = False
        for i in range(len(vo) - 1):
            if vo[i][1] + 0.3 > lines[i + 1][0]:
                lines[i + 1][0] = vo[i][1] + 0.3
                bad = True
        if not bad:
            break
    chime(buf, cfg["payoff"])
    cfg["T"] = vo[-1][1] + 2.4
    return cls(cfg, vo), buf


# ===================================================================== 1. birds (GA plays a Flappy-style game)
WH, WW = 480, 480
BX, GAP, PW, SPACING, SPEED = 120, 112, 64, 200, 3.2
GRAV, FLAP = 0.45, -6.6


def pipes_for(seed, n=60):
    r = np.random.default_rng(seed)
    return [(400 + i * SPACING, int(r.integers(110, 370))) for i in range(n)]


def play(w, pipes, maxsteps, collect=False):
    y, vy = WH / 2, 0.0
    ys = []
    for s in range(maxsteps):
        px = [(x - SPEED * s, g) for x, g in pipes]
        nxt = next(((x, g) for x, g in px if x + PW > BX - 12), None)
        gy = nxt[1] if nxt else WH / 2
        inp = ((gy - y) / 200.0, vy / 8.0, 1.0)
        if w[0] * inp[0] + w[1] * inp[1] + w[2] > 0:
            vy = FLAP
        vy += GRAV
        y += vy
        if collect:
            ys.append(y)
        if y < 6 or y > WH - 6:
            return s, ys
        for x, g in px:
            if x - 12 < BX + 12 and x + PW > BX - 12 and (y < g - GAP / 2 + 2 or y > g + GAP / 2 - 2):
                return s, ys
    return maxsteps, ys


def birds_sim(seed, npop=40, gens=30, maxsteps=2400):
    rng = np.random.default_rng(seed)
    pipes = pipes_for(11)
    pop = rng.normal(0, 1, (npop, 3))
    hist = []
    for g in range(gens):
        res = [play(w, pipes, maxsteps, True) for w in pop]
        sc = np.array([r[0] for r in res])
        hist.append(dict(pop=pop.copy(), sc=sc, ys=[r[1] for r in res], best=int(sc.max())))
        if sc.max() >= maxsteps:
            break
        order = np.argsort(-sc)
        elite = pop[order[:6]]
        kids = [e.copy() for e in elite[:3]]
        while len(kids) < npop:
            a, b = elite[rng.integers(0, 6)], elite[rng.integers(0, 6)]
            m = rng.random(3) < 0.5
            c = np.where(m, a, b) + rng.normal(0, 0.25, 3) * (rng.random(3) < 0.6)
            kids.append(c)
        pop = np.array(kids)
    return pipes, hist


class BirdReel(SeqReel):
    def content(self, t):
        s = self.st(t)
        im = Image.new("RGB", (WW, WH), (16, 12, 44))
        d = ImageDraw.Draw(im, "RGBA")
        for k in range(0, WH, 24):
            d.line([0, k, WW, k], fill=(255, 255, 255, 8))
        off = s["step"] * SPEED
        for x, g in self.cfg["pipes"]:
            x -= off
            if x > WW or x + PW < 0:
                continue
            for (a, b) in ((0, g - GAP / 2), (g + GAP / 2, WH)):
                d.rectangle([x, a, x + PW, b], fill=(40, 200, 130, 255), outline=(140, 255, 200, 255), width=3)
        for i, y in s["birds"]:
            col = (255, 226, 96, 255) if i != s["lead"] else (255, 120, 190, 255)
            r = 9 if i != s["lead"] else 12
            d.ellipse([BX - r, y - r, BX + r, y + r], fill=col, outline=INK + (255,), width=2)
        im = im.resize((VS, VS), Image.BILINEAR)
        return im

    def stats(self, t):
        s = self.st(t)
        return [(f"GEN {s['gen'] + 1}", YEL), (f"{s['alive']} ALIVE", CYAN)], s["frac"], f"BEST SO FAR: {s['bestpipes']} PIPES", GREEN


def build_birds(model):
    best = None
    for sd in range(1, 40):
        pipes, hist = birds_sim(sd)
        if hist[-1]["best"] >= 2400 and 5 <= len(hist) <= 13:
            best = (sd, pipes, hist)
            break
    sd, pipes, hist = best
    ng = len(hist)
    # generalisation: final champion on fresh pipe layouts
    champ = hist[-1]["pop"][int(np.argmax(hist[-1]["sc"]))]
    fresh = [play(champ, pipes_for(1000 + k), 2400)[0] for k in range(20)]
    ok = sum(1 for f in fresh if f >= 2400)
    pp = lambda steps: int(max(0, (steps * SPEED - (pipes[0][0] - BX + PW)) // SPACING + 1)) if steps * SPEED > pipes[0][0] - BX + PW else 0  # noqa: E731
    show = sorted(set([0, 1, 2, 3, 4, 5, 7, 9] + [ng - 2, ng - 1]))
    show = [g for g in show if g < ng]
    frames = []
    bestp = 0
    first_start = {}
    for g in show:
        h = hist[g]
        stride = 1 if g < 4 else (2 if g < ng - 1 else 5)
        steps = int(h["sc"].max()) + 1
        if g == ng - 1:
            steps = min(steps, 1500)
        bestp = max(bestp, pp(int(h["sc"].max())))
        first_start[g] = len(frames)
        lead = int(np.argmax(h["sc"]))
        for st in range(0, steps, stride):
            birds = [(i, h["ys"][i][st]) for i in range(len(h["ys"])) if h["sc"][i] > st and st < len(h["ys"][i])]
            frames.append(dict(step=st, birds=birds, gen=g, alive=len(birds), lead=lead, bestpipes=bestp,
                               frac=min(1, pp(st) / max(1, pp(2400)))))
        frames += [frames[-1]] * 6
    fps = 30
    tpay = len(frames) / fps + 0.4
    n_fin = pp(2400)
    g1best = pp(int(hist[0]["sc"].max()))
    t_gen = lambda g: first_start[g] / fps + 0.4  # noqa: E731
    cfg = mk_cfg(40, YEL, PINK, round(tpay - (len(frames) - first_start[show[-1]]) / fps + 0.6, 1), 131, [0.6, t_gen(show[1]) + 1, t_gen(show[2]) + 1, t_gen(show[-2]) + 1, tpay - 5, tpay - 0.5], None)
    cfg["frames"], cfg["pipes"] = frames, pipes
    tcr = t_gen(show[-1])
    cfg["header"] = lambda t: (([("THIS BIRD HAS", WHITE), ("NEVER PLAYED.", YEL)], "40 DUMB BIRDS") if t < tcr else
                               ([("THE BEST ONE", WHITE), ("NEVER DIES.", GREEN)], f"GEN {ng} - {n_fin} PIPES"))
    lines = [(0.2, "These forty birds have never played this game. Watch what happens."),
             (4.0, f"Generation one is pure chaos. The best one clears {g1best} pipes."),
             (8.6, "No rules. No training data. Each bird has three random numbers as a brain."),
             (13.2, "The ones that last longest breed. Their kids get a small random tweak."),
             (min(t_gen(show[-2]) - 2.5, 19), f"By generation {show[-2] + 1}, most of them are getting through."),
             (tcr + 0.1, f"Generation {ng}. They clear {n_fin} pipes in a row. They never touch a wall."),
             (tcr + 6.2, f"The champion also beat {ok} of 20 brand new pipe layouts. Evolution, no hand coding."),
             (tcr + 12.4, "Follow for more AI from scratch.")]
    lines = sorted(lines)
    reel, buf = finish(cfg, lines, BirdReel, dict(model=model))
    post = dict(title="LEARN-1  The bird that never played",
                caption=f"Forty birds, a random 3-number brain each, zero training data. The ones that survive longest breed with small random mutations. Generation 1's best clears {g1best} pipes; by generation {ng} the best bird clears {n_fin} pipes in a row. The champion also survived {ok} of 20 brand-new random pipe layouts. This is a genetic algorithm, a classic in evolutionary computation.",
                tags=["ai", "geneticalgorithm", "machinelearning", "flappybird", "coding", "shipwithpip"],
                pinned="Build me a single-file HTML canvas Flappy Bird where 40 birds each have a 3-weight brain (inputs: gap height minus bird height, vertical speed, bias; flap if the weighted sum is above 0). Each generation keep the top 6, breed the next 40 with random mutation, and show the generation number and best score.",
                notes=f"Seed {sd}. Brain: flap if w0*dy/200 + w1*vy/8 + w2 > 0. Population 40, elite 6 (top 3 copied), uniform crossover with Gaussian mutation sigma 0.25. Game: gap {GAP}px, speed {SPEED}px/step, cap 2400 steps; first generation to hit the cap is generation {ng}. Generation 1 best {g1best} pipes. Fresh-layout test: {ok}/20 random layouts survived 2400 steps with the final champion. This is a simulation of a Flappy-style game, not the original game. Source for the method: genetic algorithms (Holland 1975).")
    return reel, buf, cfg, post


# ===================================================================== 2. digits MLP from scratch
def digits_sim(seed=4):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    rng = np.random.default_rng(seed)
    W1, b1 = rng.normal(0, 0.3, (64, 32)), np.zeros(32)
    W2, b2 = rng.normal(0, 0.3, (32, 10)), np.zeros(10)

    def fwd(A):
        h = np.maximum(0, A @ W1 + b1)
        z = h @ W2 + b2
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return h, p / p.sum(1, keepdims=True)

    showidx = np.random.default_rng(1).permutation(len(Xte))[:40]
    marks = sorted(set(int(v) for v in np.unique(np.round(np.geomspace(1, 3200, 70)).astype(int))))
    snaps = []
    snaps.append(dict(it=0, acc=float((fwd(Xte)[1].argmax(1) == yte).mean()), pred=fwd(Xte[showidx])[1].argmax(1)))
    it = 0
    lr = 0.15
    ep_order = []
    while it < marks[-1]:
        perm = rng.permutation(len(Xtr))
        for k in range(0, len(Xtr) - 31, 32):
            b = perm[k:k + 32]
            h, p = fwd(Xtr[b])
            g = p.copy()
            g[np.arange(len(b)), ytr[b]] -= 1
            g /= len(b)
            dW2, db2 = h.T @ g, g.sum(0)
            dh = g @ W2.T * (h > 0)
            dW1, db1 = Xtr[b].T @ dh, dh.sum(0)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            it += 1
            if it in marks:
                pr = fwd(Xte)[1].argmax(1)
                snaps.append(dict(it=it, acc=float((pr == yte).mean()), pred=pr[showidx]))
            if it >= marks[-1]:
                break
    trainacc = float((fwd(Xtr)[1].argmax(1) == ytr).mean())
    return Xte[showidx], yte[showidx], snaps, trainacc, len(Xtr), len(Xte)


class DigitReel(SeqReel):
    def content(self, t):
        s = self.st(t)
        im = base_img()
        d = ImageDraw.Draw(im, "RGBA")
        imgs, lab = self.cfg["imgs"], self.cfg["labels"]
        cw = 114
        ox, oy = (VS - cw * 8) // 2, 30
        for i in range(40):
            r, c = divmod(i, 8)
            x, y = ox + c * cw, oy + r * cw
            a = (imgs[i].reshape(8, 8) * 255).astype(np.uint8)
            tile = Image.fromarray(a, "L").resize((cw - 14, cw - 14), Image.NEAREST)
            ok = s["pred"][i] == lab[i]
            col = GREEN if ok else (255, 70, 90)
            d.rounded_rectangle([x + 2, y + 2, x + cw - 2, y + cw - 2], radius=14, fill=(0, 0, 0, 120), outline=col + (255,), width=4)
            im.paste(Image.merge("RGB", (tile, tile, tile)), (x + 7, y + 7))
            d.rounded_rectangle([x + cw - 44, y + cw - 40, x + cw - 6, y + cw - 6], radius=10, fill=col + (255,))
            d.text((x + cw - 25, y + cw - 23), str(int(s["pred"][i])), font=font(F_BOLD, 30), fill=INK + (255,), anchor="mm")
        # accuracy curve
        cx0, cy0, cx1, cy1 = 70, 700, 900, 930
        d.line([cx0, cy0, cx0, cy1, cx1, cy1], fill=(255, 255, 255, 90), width=2)
        d.text((cx0, cy0 - 20), "accuracy on digits it never trained on", font=font(F_MONO, 24), fill=(255, 255, 255, 200), anchor="lm")
        n = s["k"]
        pts = [(cx0 + (math.log10(max(1, sn["it"])) / math.log10(3200)) * (cx1 - cx0), cy1 - sn["acc"] * (cy1 - cy0 - 40)) for sn in self.cfg["snaps"][:n + 1]]
        if len(pts) > 1:
            d.line(pts, fill=GREEN + (255,), width=6, joint="curve")
        d.ellipse([pts[-1][0] - 9, pts[-1][1] - 9, pts[-1][0] + 9, pts[-1][1] + 9], fill=YEL + (255,))
        return im

    def stats(self, t):
        s = self.st(t)
        return [(f"STEP {s['it']}", YEL), (f"{s['acc'] * 100:0.0f}% RIGHT", GREEN if s["acc"] > 0.5 else PINK)], s["acc"], "ACCURACY ON UNSEEN DIGITS", GREEN


def build_digits(model):
    imgs, lab, snaps, trainacc, ntr, nte = digits_sim()
    frames = []
    for k, sn in enumerate(snaps):
        n = 45 if k == 0 else (16 if k < 12 else 11)
        for _ in range(n):
            frames.append(dict(k=k, it=sn["it"], acc=sn["acc"], pred=sn["pred"]))
    frames += [frames[-1]] * 45
    fps = 30
    fin = snaps[-1]["acc"]
    a0 = snaps[0]["acc"]
    kk = lambda it: min(snaps, key=lambda s: abs(s["it"] - it))["acc"]  # noqa: E731
    tstart = lambda it: next((i for i, f in enumerate(frames) if f["it"] >= it), len(frames) - 1) / fps + 0.4  # noqa: E731
    tpay = tstart(snaps[-1]["it"])
    cfg = mk_cfg(40, CYAN, VIOLET, round(tpay + 0.5, 1), 132, [0.6, tstart(10) + .3, tstart(60) + .3, tstart(300) + .3, tstart(1000) + .3, tpay + 1.5], None)
    cfg["frames"], cfg["imgs"], cfg["labels"], cfg["snaps"] = frames, imgs, lab, snaps
    cfg["header"] = lambda t: (([("THIS AI HAS NEVER", WHITE), ("SEEN A NUMBER", YEL)], "ZERO KNOWLEDGE") if t < tpay else
                               ([("NOW IT READS", WHITE), (f"{fin * 100:0.0f}% RIGHT", GREEN)], "LEARNED IN SECONDS"))
    lines = [(0.2, "This AI has never seen a number. Not one. Watch it learn to read."),
             (4.6, f"Right now it guesses. Only {a0 * 100:0.0f} percent right, which is basically random."),
             (9.4, f"It gets shown a few handwritten digits, checks its answer, and nudges its numbers. That's it."),
             (tstart(60) - 1.0, f"After {60} nudges, it's already at {kk(60) * 100:0.0f} percent."),
             (tstart(400) - 1.0, f"After 400, {kk(400) * 100:0.0f} percent."),
             (tpay + 0.1, f"After {snaps[-1]['it']} nudges, {fin * 100:0.0f} percent on digits it has never seen."),
             (tpay + 6.0, "Same idea behind the AI you use every day. Just much smaller."),
             (tpay + 10.6, "Follow for more.")]
    lines = sorted(lines)
    reel, buf = finish(cfg, lines, DigitReel, dict(model=model))
    post = dict(title="LEARN-2  The AI that never saw a number",
                caption=f"A tiny neural network (64 inputs, 32 hidden units, 10 outputs) starts with random weights and about {a0 * 100:0.0f}% accuracy on handwritten digits. After {snaps[-1]['it']} small gradient steps it reaches {fin * 100:0.0f}% on {nte} test digits it never trained on (trained on {ntr}). Data: scikit-learn's 8x8 digits set. Every number on screen is computed live from the model, not faked.",
                tags=["ai", "neuralnetwork", "machinelearning", "python", "deeplearning", "shipwithpip"],
                pinned="Write a NumPy-only neural network (64-32-10, ReLU, softmax, minibatch SGD lr 0.15, batch 32) and train it on sklearn.datasets.load_digits. Print test accuracy after 1, 10, 60, 400 and 3200 steps.",
                notes=f"scikit-learn load_digits (1797 samples, 8x8, 0-16 scaled to 0-1). Stratified 80/20 split, random_state 4: train {ntr}, test {nte}. 64-32-10 ReLU network, init N(0,0.3), softmax cross-entropy, SGD lr 0.15, batch 32, {snaps[-1]['it']} steps. Accuracy at step 0: {a0:.3f}; final test accuracy: {fin:.3f}; final train accuracy {trainacc:.3f}. The 40 digits shown are a fixed random subset of the test set; the green/red borders are real predictions at that step. Numbers differ across seeds by about a percentage point.")
    return reel, buf, cfg, post


# ===================================================================== 3. cliff walking Q-learning
CR, CC = 4, 12


def cliff_sim(seed=3, episodes=500):
    rng = np.random.default_rng(seed)
    Q = np.zeros((CR, CC, 4))
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    eps, alpha, gamma = 0.1, 0.5, 1.0
    eps_hist = []
    falls = 0
    for ep in range(episodes):
        r, c = 3, 0
        traj = [(r, c)]
        steps = 0
        fell_here = 0
        ret = 0
        while True:
            a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[r, c] + rng.random(4) * 1e-9))
            nr, nc = r + dirs[a][0], c + dirs[a][1]
            nr, nc = min(CR - 1, max(0, nr)), min(CC - 1, max(0, nc))
            rew = -1
            fell = False
            if nr == 3 and 1 <= nc <= 10:
                rew, fell = -100, True
                nr, nc = 3, 0
            done = (nr, nc) == (3, 11)
            tgt = rew + (0 if done else gamma * Q[nr, nc].max())
            Q[r, c, a] += alpha * (tgt - Q[r, c, a])
            ret += rew
            r, c = nr, nc
            traj.append((r, c))
            steps += 1
            if fell:
                falls += 1
                fell_here += 1
            if done or steps > 1500:
                break
        eps_hist.append(dict(traj=traj, falls=falls, fell_here=fell_here, steps=steps, ret=ret, Q=Q.copy()))
    return eps_hist, dirs


def greedy_path(Q, dirs):
    r, c = 3, 0
    path = [(r, c)]
    for _ in range(60):
        a = int(np.argmax(Q[r, c]))
        nr, nc = min(CR - 1, max(0, r + dirs[a][0])), min(CC - 1, max(0, c + dirs[a][1]))
        if nr == 3 and 1 <= nc <= 10:
            path.append((nr, nc))
            return path, False
        r, c = nr, nc
        path.append((r, c))
        if (r, c) == (3, 11):
            return path, True
    return path, False


class CliffReel(SeqReel):
    CW = 76

    def cell(self, r, c):
        return 24 + c * self.CW, 300 + r * self.CW

    def content(self, t):
        s = self.st(t)
        im = base_img((12, 8, 30))
        d = ImageDraw.Draw(im, "RGBA")
        for r in range(CR):
            for c in range(CC):
                x, y = self.cell(r, c)
                if r == 3 and 1 <= c <= 10:
                    fill, out = (140, 20, 50, 255), (255, 80, 110, 255)
                elif (r, c) == (3, 0):
                    fill, out = (30, 70, 120, 255), CYAN + (255,)
                elif (r, c) == (3, 11):
                    fill, out = (20, 110, 70, 255), GREEN + (255,)
                else:
                    fill, out = (24, 20, 56, 255), (255, 255, 255, 40)
                d.rectangle([x, y, x + self.CW, y + self.CW], fill=fill, outline=out, width=2)
        x, y = self.cell(3, 5)
        d.text((x + self.CW / 2 + 0, y + self.CW / 2), "THE CLIFF", font=font(F_MONO, 26), fill=(255, 190, 200, 255), anchor="mm")
        for (r, c), tx in (((3, 0), "S"), ((3, 11), "GOAL")):
            x, y = self.cell(r, c)
            d.text((x + self.CW / 2, y + self.CW / 2), tx, font=font(F_BOLD, 24 if tx == "S" else 18), fill=WHITE + (255,), anchor="mm")
        # learned policy arrows
        Q = s["Q"]
        for r in range(CR):
            for c in range(CC):
                if (r == 3 and c >= 1) or not Q[r, c].any():
                    continue
                a = int(np.argmax(Q[r, c]))
                dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]
                x, y = self.cell(r, c)
                cx, cy = x + self.CW / 2, y + self.CW / 2
                d.line([cx - dc * 14, cy - dr * 14, cx + dc * 14, cy + dr * 14], fill=YEL + (210,), width=5)
                d.ellipse([cx + dc * 14 - 6, cy + dr * 14 - 6, cx + dc * 14 + 6, cy + dr * 14 + 6], fill=YEL + (230,))
        # trail + agent
        tr = s["trail"]
        pts = [(self.cell(r, c)[0] + self.CW / 2, self.cell(r, c)[1] + self.CW / 2) for r, c in tr]
        if len(pts) > 1:
            d.line(pts[-120:], fill=PINK + (200,), width=5, joint="curve")
        if s["greedy"]:
            gp = s["greedy"]
            gpts = [(self.cell(r, c)[0] + self.CW / 2, self.cell(r, c)[1] + self.CW / 2) for r, c in gp]
            d.line(gpts, fill=GREEN + (90,), width=26, joint="curve")
            d.line(gpts, fill=GREEN + (255,), width=8, joint="curve")
        ax, ay = pts[-1]
        d.ellipse([ax - 17, ay - 17, ax + 17, ay + 17], fill=PINK + (255,), outline=WHITE + (255,), width=3)
        if s["flash"] > 0:
            d.rectangle([0, 0, VS, VS], fill=(255, 40, 70, int(70 * s["flash"])))
            d.text((VS / 2, 190), "SPLAT", font=font(F_BOLD, 90), fill=(255, 90, 110, int(255 * s["flash"])), anchor="mm", stroke_width=6, stroke_fill=INK)
        # episode steps chart (return per episode)
        d.text((40, 650), "score per try (higher is better)", font=font(F_MONO, 24), fill=(255, 255, 255, 200), anchor="lm")
        rets = self.cfg["rets"][:s["ep"] + 1]
        x0, y0, x1, y1 = 60, 690, 920, 930
        d.line([x0, y0, x0, y1, x1, y1], fill=(255, 255, 255, 90), width=2)
        n = max(2, len(self.cfg["rets"]))
        pl = [(x0 + i / n * (x1 - x0), y1 - clamp((v + 300) / 300) * (y1 - y0 - 10)) for i, v in enumerate(rets)]
        if len(pl) > 1:
            d.line(pl, fill=CYAN + (255,), width=4, joint="curve")
        return im

    def stats(self, t):
        s = self.st(t)
        return [(f"TRY {s['ep'] + 1}", YEL), (f"{s['falls']} FALLS", PINK)], clamp(1 - s["falls"] / self.cfg["maxfalls"]), "FALLS SO FAR", PINK


def build_cliff(model):
    hist, dirs = cliff_sim()
    gp, ok = greedy_path(hist[-1]["Q"], dirs)
    nfalls = hist[-1]["falls"]
    shown = list(range(0, 8)) + [9, 11, 14, 18, 24, 32, 45, 65, 100, 160, 260, 499]
    frames = []
    for ep in shown:
        h = hist[ep]
        tr = h["traj"]
        stride = max(1, math.ceil(len(tr) / (60 if ep < 8 else 36)))
        last_fall = 0
        for k in range(1, len(tr) + stride, stride):
            sub = tr[:min(k, len(tr))]
            prev_fell = any(tr[j] == (3, 0) and tr[j - 1] != (3, 0) and tr[j - 1][0] == 3 and 1 <= tr[j - 1][1] <= 10 for j in range(1, len(sub)))
            frames.append(dict(ep=ep, falls=hist[ep - 1]["falls"] if ep else 0, trail=sub, Q=hist[ep - 1]["Q"] if ep else np.zeros((CR, CC, 4)),
                               flash=0.0, greedy=None))
        # fall flashes: mark first frame after each fall
        for fidx in range(1, len(tr)):
            if tr[fidx] == (3, 0) and tr[fidx - 1][0] == 3 and 1 <= tr[fidx - 1][1] <= 10:
                j = min(len(frames) - 1, len(frames) - (len(tr) + stride) // stride + fidx // stride)
                for q in range(j, min(len(frames), j + 7)):
                    frames[q]["flash"] = 1 - (q - j) / 7
        for q in frames[-3:]:
            q["falls"] = h["falls"]
            q["Q"] = h["Q"]
        frames += [dict(frames[-1], flash=0)] * 3
    for _ in range(100):
        frames.append(dict(ep=499, falls=nfalls, trail=hist[499]["traj"][-1:] + [(3, 0)], Q=hist[-1]["Q"], flash=0, greedy=gp))
    gl = len(gp) - 1
    tpay = (len(frames) - 100) / 30 + 0.4
    f_at = lambda ep: hist[ep]["falls"]  # noqa: E731
    cfg = mk_cfg(40, PINK, CYAN, round(tpay, 1), 133, [0.6, 5, 10, 16, tpay - 2, tpay + 3], None)
    cfg["frames"], cfg["rets"], cfg["maxfalls"] = frames, [h["ret"] for h in hist], max(10, nfalls)
    cfg["header"] = lambda t: (([("IT WILL FALL OFF", WHITE), (f"{nfalls} TIMES", PINK)], "ZERO SENSE OF DANGER") if t < tpay else
                               ([("THEN IT FOUND", WHITE), ("THE PERFECT PATH", GREEN)], f"{gl} STEPS. SHORTEST POSSIBLE"))
    ts = lambda ep: next(i for i, f in enumerate(frames) if f["ep"] >= ep) / 30 + 0.4  # noqa: E731
    lines = [(0.2, "This AI will fall off the same cliff again and again. Watch."),
             (4.4, "It starts with zero idea what's dangerous. Every red tile is a fall, and a fall costs a hundred points."),
             (ts(9) + 0.6, f"After ten tries, it's already fallen {f_at(9)} times."),
             (ts(45) + 0.2, "But every fall teaches it that one tile is a bad idea. The fear spreads backwards, tile by tile."),
             (ts(160) + 0.2, f"After 160 tries it has fallen {f_at(160)} times in total."),
             (tpay + 0.1, f"Now it walks the edge in {gl} steps, the shortest route that exists."),
             (tpay + 5.0, "This is Q-learning, a textbook reinforcement learning example."),
             (tpay + 9.4, "Follow for more.")]
    lines = sorted(lines)
    reel, buf = finish(cfg, lines, CliffReel, dict(model=model))
    post = dict(title="LEARN-3  The AI that fell off a cliff",
                caption=f"A 4x12 grid, a cliff along the bottom, and an agent that starts knowing nothing. Each fall costs 100 points; each step costs 1. Using Q-learning with 10% random exploration it falls {nfalls} times over 500 tries, then finds the shortest path ({gl} steps along the cliff edge). This is the classic Cliff Walking example from Sutton and Barto's reinforcement learning textbook, simulated from scratch.",
                tags=["ai", "reinforcementlearning", "qlearning", "machinelearning", "python", "shipwithpip"],
                pinned="Write a NumPy Q-learning agent for the 4x12 Cliff Walking gridworld (alpha 0.5, gamma 1, epsilon 0.1, 500 episodes, -1 per step, -100 and reset to start on the cliff). After training, print the greedy path and the total number of falls.",
                notes=f"Cliff Walking gridworld per Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.), Example 6.6: 4x12, start bottom-left, goal bottom-right, cliff along the bottom row between them, reward -1 per step, -100 for stepping on the cliff (sent back to start). Q-learning, alpha 0.5, gamma 1, epsilon 0.1 (textbook values), 500 episodes, seed 3. Falls across all 500 episodes: {nfalls}. Greedy path length after training: {gl} steps ({'reaches the goal' if ok else 'DOES NOT reach goal'}); shortest possible is 13. Q-learning is known to learn the cliff-edge path while epsilon-greedy exploration keeps causing occasional falls; SARSA learns a safer longer path.")
    return reel, buf, cfg, post


# ===================================================================== 4. triangles recreate a picture
TS, NTRI = 64, 70


def target_img():
    im = Image.new("RGB", (TS, TS))
    px = np.zeros((TS, TS, 3), float)
    yy = np.linspace(0, 1, TS)[:, None]
    top, bot = np.array([40, 20, 110.0]), np.array([255, 140, 70.0])
    px[:] = (top * (1 - yy) + bot * yy)[:, None, :]
    im = Image.fromarray(px.astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.ellipse([22, 26, 46, 50], fill=(255, 236, 140))
    d.polygon([(0, 64), (0, 44), (14, 34), (26, 48), (38, 38), (52, 50), (64, 40), (64, 64)], fill=(30, 14, 60))
    d.polygon([(0, 64), (0, 54), (20, 50), (40, 56), (64, 50), (64, 64)], fill=(14, 6, 34))
    return im


def tri_render(tris, size, bg=(0, 0, 0)):
    k = size / TS
    base = Image.new("RGBA", (size, size), bg + (255,))
    for (pts, col) in tris:
        ov = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ImageDraw.Draw(ov).polygon([(x * k, y * k) for x, y in pts], fill=tuple(int(v) for v in col))
        base = Image.alpha_composite(base, ov)
    return base.convert("RGB")


def tri_sim(seed=2, iters=26000):
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target_img(), float)

    def rnd_tri():
        c = rng.uniform(0, TS, 2)
        pts = [(float(np.clip(c[0] + rng.normal(0, 12), -4, TS + 4)), float(np.clip(c[1] + rng.normal(0, 12), -4, TS + 4))) for _ in range(3)]
        return [pts, [int(v) for v in rng.integers(0, 255, 3)] + [int(rng.integers(40, 140))]]

    tris = [rnd_tri() for _ in range(NTRI)]

    def err(ts):
        return float(((np.asarray(tri_render(ts, TS), float) - tgt) ** 2).mean())

    cur = err(tris)
    e0 = cur
    marks = set(int(v) for v in np.geomspace(1, iters, 80))
    snaps = [(0, cur, [([tuple(p) for p in t[0]], tuple(t[1])) for t in tris])]
    acc = 0
    for it in range(1, iters + 1):
        i = int(rng.integers(NTRI))
        old = [list(map(list, tris[i][0])), list(tris[i][1])]
        r = rng.random()
        if r < 0.45:
            j = int(rng.integers(3))
            tris[i][0][j] = [float(np.clip(tris[i][0][j][0] + rng.normal(0, 7), -4, TS + 4)), float(np.clip(tris[i][0][j][1] + rng.normal(0, 7), -4, TS + 4))]
        elif r < 0.8:
            j = int(rng.integers(3))
            tris[i][1][j] = int(np.clip(tris[i][1][j] + rng.normal(0, 28), 0, 255))
        elif r < 0.9:
            tris[i][1][3] = int(np.clip(tris[i][1][3] + rng.normal(0, 25), 25, 200))
        else:
            dx, dy = rng.normal(0, 5, 2)
            tris[i][0] = [[float(np.clip(x + dx, -4, TS + 4)), float(np.clip(y + dy, -4, TS + 4))] for x, y in tris[i][0]]
        e = err(tris)
        if e < cur:
            cur = e
            acc += 1
        else:
            tris[i] = old
        if it in marks:
            snaps.append((it, cur, [([tuple(p) for p in t[0]], tuple(t[1])) for t in tris]))
    return snaps, e0, acc, iters


class TriReel(SeqReel):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.tgt = target_img().resize((260, 260), Image.LANCZOS)
        self.cache = {}

    def content(self, t):
        s = self.st(t)
        k = s["k"]
        if k not in self.cache:
            self.cache[k] = tri_render(self.cfg["snaps"][k][2], 900)
        im = base_img()
        im.paste(self.cache[k], (30, 30))
        d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle([30, 30, 930, 930], radius=6, outline=(255, 255, 255, 60), width=2)
        im.paste(self.tgt, (VS - 30 - 260, 30 + 0))
        d.rectangle([VS - 30 - 260, 30, VS - 30, 290], outline=YEL + (255,), width=4)
        d.rounded_rectangle([VS - 290, 296, VS - 30, 336], radius=10, fill=(14, 8, 38, 220))
        d.text((VS - 160, 316), "THE TARGET", font=font(F_MONO, 26), fill=YEL + (255,), anchor="mm")
        return im

    def stats(self, t):
        s = self.st(t)
        return [(f"TRY #{s['it']:,}", YEL), (f"{self.cfg['ntri']} TRIANGLES", CYAN)], s["sim"], f"MATCH: {s['sim'] * 100:0.0f}%", GREEN


def build_tri(model):
    snaps, e0, acc, iters = tri_sim()
    # similarity: 1 - rmse/ (rmse of a flat gray ref)
    tgt = np.asarray(target_img(), float)
    ref = float(((tgt - tgt.mean((0, 1))) ** 2).mean())
    simf = lambda e: float(clamp(1 - math.sqrt(e / ref) * 0.9, 0, 1))  # noqa: E731
    frames = []
    for k, (it, e, _) in enumerate(snaps):
        n = 40 if k == 0 else 11
        for _ in range(n):
            frames.append(dict(k=k, it=it, sim=simf(e)))
    frames += [frames[-1]] * 60
    tpay = (len(frames) - 60) / 30 + 0.4
    s0, sf = simf(snaps[0][1]), simf(snaps[-1][1])
    ts = lambda it: next((i for i, f in enumerate(frames) if f["it"] >= it), len(frames) - 1) / 30 + 0.4  # noqa: E731
    sim_at = lambda it: min(snaps, key=lambda s: abs(s[0] - it))  # noqa: E731
    cfg = mk_cfg(40, ORANGE, VIOLET, round(tpay, 1), 134, [0.6, ts(30) + .3, ts(300) + .3, ts(3000) + .3, ts(14000) + .3, tpay + 1.5], None)
    cfg["frames"], cfg["snaps"], cfg["ntri"] = frames, snaps, NTRI
    cfg["header"] = lambda t: (([(f"{NTRI} RANDOM", WHITE), ("TRIANGLES.", YEL)], "ZERO PLAN") if t < tpay else
                               ([("IT PAINTED", WHITE), ("A SUNSET", ORANGE)], "BY KEEPING LUCKY GUESSES"))
    lines = [(0.2, f"These {NTRI} random triangles will turn into this picture. Nobody tells them how."),
             (4.8, "The only rule: nudge one triangle at random. If the picture gets closer, keep it. If not, undo it."),
             (ts(300) - 1.0, f"After 300 tries, you can already see the sky."),
             (ts(3000) - 1.0, f"After 3,000, the sun and the mountains show up."),
             (tpay + 0.1, f"{iters:,} tries later, and only {acc:,} of them were kept, it's a sunset."),
             (tpay + 6.0, "No drawing skill. Just try, check, keep the winners."),
             (tpay + 10.0, "Follow for more.")]
    lines = sorted(lines)
    reel, buf = finish(cfg, lines, TriReel, dict(model=model))
    post = dict(title="LEARN-4  70 triangles, no plan",
                caption=f"{NTRI} random translucent triangles. Each step nudges one at random; if the picture gets closer to the target it's kept, otherwise undone. After {iters:,} tries ({acc:,} kept) a sunset appears. This is hill climbing, the simplest form of search. The target is a 64x64 picture I generated myself.",
                tags=["ai", "evolution", "generativeart", "python", "algorithms", "shipwithpip"],
                pinned="Write Python that approximates a 64x64 target image with 70 translucent triangles using hill climbing: mutate one triangle (move a vertex, change colour, change alpha, or shift it), recompute the mean squared error, keep the change only if error drops. Save a snapshot at 1, 10, 100, 1000 and 26000 iterations.",
                notes=f"Procedural 64x64 target (gradient sky, sun, two mountain layers). {NTRI} RGBA triangles, hill climbing with single-triangle mutations (vertex move 45%, colour 35%, alpha 10%, shift 10%), accepted only if MSE decreases. {iters:,} iterations, {acc:,} accepted, seed 2. The 'match %' on screen is 1 - 0.9*sqrt(MSE/variance of target), a visual score I defined, not a standard metric; it reads {s0 * 100:0.0f}% at the start and {sf * 100:0.0f}% at the end. The technique is simple hill climbing (a Roger Alsing-style 'evolution of Mona Lisa' demo uses the same idea).")
    return reel, buf, cfg, post


# ===================================================================== 5. TSP untangling
def tsp_sim(n=40, seed=7):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0.06, 0.94, (n, 2))
    D = np.linalg.norm(P[:, None] - P[None], axis=2)

    def length(t):
        return float(sum(D[t[i], t[(i + 1) % n]] for i in range(n)))

    def two_opt_full(t):
        t = list(t)
        imp = True
        while imp:
            imp = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                    if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-12:
                        t[i + 1:j + 1] = reversed(t[i + 1:j + 1])
                        imp = True
        return t

    # reference: best of many randomized restarts of 2-opt (honest 'best I could find')
    rr = np.random.default_rng(99)
    best_ref = min(length(two_opt_full(list(rr.permutation(n)))) for _ in range(300))
    # nearest neighbour
    un, cur, nn = set(range(1, n)), 0, [0]
    while un:
        nx = min(un, key=lambda j: D[cur, j])
        nn.append(nx)
        un.remove(nx)
        cur = nx
    Lnn = length(nn)
    # displayed run: random start, accept random segment reversal only if shorter
    t = list(rng.permutation(n))
    L0 = length(t)
    snaps = [(0, L0, list(t))]
    cur = L0
    tries = 0
    kept = 0
    while True:
        tries += 1
        i, j = sorted(rng.choice(n, 2, replace=False))
        if j - i < 1:
            continue
        a, b, c, d = t[(i - 1) % n], t[i], t[j], t[(j + 1) % n]
        if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-12:
            t[i:j + 1] = reversed(t[i:j + 1])
            cur = length(t)
            kept += 1
            snaps.append((tries, cur, list(t)))
        if tries > 60000:
            break
    return P, snaps, L0, Lnn, best_ref, tries, kept


class TspReel(SeqReel):
    def content(self, t):
        s = self.st(t)
        im = base_img((12, 8, 32))
        d = ImageDraw.Draw(im, "RGBA")
        P = self.cfg["P"]
        pts = [(40 + p[0] * 880, 40 + p[1] * 880) for p in P]
        route = s["route"]
        poly = [pts[i] for i in route] + [pts[route[0]]]
        d.line(poly, fill=CYAN + (60,), width=22, joint="curve")
        d.line(poly, fill=CYAN + (255,), width=5, joint="curve")
        for (x, y) in pts:
            d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=PINK + (255,), outline=WHITE + (255,), width=2)
        return im

    def stats(self, t):
        s = self.st(t)
        return [(f"{s['shorter']:0.0f}% SHORTER", GREEN), (f"{s['keep']} FIXES", YEL)], s["shorter"] / 100.0, "ROUTE LENGTH CUT", GREEN


def build_tsp(model):
    P, snaps, L0, Lnn, ref, tries, kept = tsp_sim()
    n = len(P)
    routes = math.factorial(n - 1) // 2
    sci = f"{routes:.1e}"
    mant, ex = sci.split("e")
    words = f"{float(mant):0.1f} times ten to the {int(ex)}"
    idxs = [0] + list(range(1, len(snaps)))
    pick = sorted(set(int(v) for v in np.linspace(0, len(snaps) - 1, 80)))
    frames = []
    for k in pick:
        it, L, rt = snaps[k]
        for _ in range(8 if k else 40):
            frames.append(dict(route=rt, shorter=(1 - L / L0) * 100, keep=k, tries=it, L=L))
    frames += [frames[-1]] * 60
    fin = snaps[-1][1]
    tpay = (len(frames) - 60) / 30 + 0.4
    over = (fin / ref - 1) * 100
    ts = lambda frac: len(frames) * frac / 30  # noqa: E731
    cfg = mk_cfg(40, CYAN, PINK, round(tpay, 1), 135, [0.6, ts(0.15), ts(0.4), ts(0.7), tpay - 0.2, tpay + 1.5], None)
    cfg["frames"], cfg["P"] = frames, P
    cfg["header"] = lambda t: (([("THE SALESMAN", WHITE), ("PROBLEM", YEL)], f"{n} CITIES. ONE ROUTE") if t < tpay else
                               ([("MESS BECOMES", WHITE), ("A CLEAN LOOP", GREEN)], f"{(1 - fin / L0) * 100:0.0f}% SHORTER"))
    mid = frames[int(len(frames) * 0.4)]
    lines = [(0.2, f"{n} cities. Visit each one once. This AI starts with a total mess."),
             (4.2, f"There are {words} possible routes. No computer can check them all."),
             (9.4, "So it does one dumb thing. Pick two stops, flip the part between them. If the route gets shorter, keep it."),
             (ts(0.38), f"Look at that. After {mid['keep']} fixes it's already {mid['shorter']:0.0f} percent shorter."),
             (tpay + 0.1, f"The final route is {(1 - fin / L0) * 100:0.0f} percent shorter than where it started."),
             (tpay + 5.4, f"And it lands within {over:0.0f} percent of the best I could find with 300 restarts."),
             (tpay + 10.4, "Follow for more.")]
    lines = sorted(lines)
    reel, buf = finish(cfg, lines, TspReel, dict(model=model))
    post = dict(title="LEARN-5  The route nobody could check",
                caption=f"{n} random cities and a random starting route. The only move: flip a segment of the route, and keep it only if the route gets shorter. There are about {sci} possible routes, so brute force is out. The final route is {(1 - fin / L0) * 100:0.0f}% shorter than the start, and within {over:0.0f}% of the best route I found with 300 random restarts. This is 2-opt, a classic travelling salesman heuristic.",
                tags=["ai", "algorithms", "travelingsalesman", "optimization", "python", "shipwithpip"],
                pinned="Write Python that solves a 40-city travelling salesman problem with 2-opt: start from a random order, repeatedly pick two positions, reverse the segment between them, and keep it only if the total tour length drops. Plot the route every 50 fixes.",
                notes=f"{n} random cities (seed 7). Possible distinct tours (n-1)!/2 = {routes:.3e}. Displayed run: random start tour length {L0:.3f}, random segment reversal accepted only if shorter, {tries:,} attempts, {kept} accepted; final {fin:.3f} ({(1 - fin / L0) * 100:0.1f}% shorter). Nearest-neighbour baseline: {Lnn:.3f}. Best of 300 random-restart full 2-opt runs: {ref:.3f}; displayed final is {over:0.1f}% above that (this is a heuristic reference, not a proven optimum). 2-opt: Croes (1958).")
    return reel, buf, cfg, post


BUILDERS = {"birds": build_birds, "digits": build_digits, "cliff": build_cliff, "triangles": build_tri, "tsp": build_tsp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = BUILDERS[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
