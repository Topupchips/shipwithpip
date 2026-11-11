"""Learning-in-motion reels with honest numbers: spiral (train vs test), coordinate-net painter, 3 network sizes.
usage: python learnfx.py spiral|paint|sizes out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learn  # noqa: E402
import paintnet  # noqa: E402
from scenes import CYAN, GREEN, PINK, VIOLET, YEL, ORANGE, Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, INK, VS, clamp, ease_out, font, smooth, tone  # noqa: E402

GN = learn.GN


def heat_img(p, size):
    a = np.clip(np.abs(p - 0.5) * 2, 0, 1) ** 0.8 * 0.9
    base = np.array([16, 10, 44], float)
    col = np.where((p > 0.5)[..., None], np.array(PINK, float), np.array(CYAN, float))
    arr = base * (1 - a[..., None]) + col * a[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").resize((size, size), Image.BICUBIC)


def interp_grid(H, e):
    i = int(min(e, len(H["grid"]) - 2))
    fr = min(1.0, e - i)
    return H["grid"][i] * (1 - fr) + H["grid"][i + 1] * fr


def pw(pts, t):
    """piecewise-linear map through (t, v) points"""
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return pts[-1][1]


def inv(pts, v):
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if v <= v1:
            return t0 + (t1 - t0) * (v - v0) / max(1e-9, v1 - v0)
    return pts[-1][0]


def dot(d, x, y, col, filled, r=9):
    if filled:
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (255,), outline=(255, 255, 255, 255), width=3)
    else:
        d.ellipse([x - r, y - r, x + r, y + r], fill=(12, 6, 30, 200), outline=col + (255,), width=4)


def write_post(path, title, caption, tags, pinned, notes):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{title}\n\nCAPTION\n{caption}\n\n{' '.join('#' + t for t in tags)}\n\nPINNED COMMENT / PROMPT\n{pinned}\n\nFACT CHECK NOTES (real values from the run)\n{notes}\n")


# ===================================================================== 1. spiral, train vs test
class SpiralReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.D, self.H = cfg["D"], cfg["H"]

    def e_at(self, t):
        return pw(self.cfg["emap"], t)

    def content(self, t):
        e = self.e_at(t)
        im = heat_img(interp_grid(self.H, e), VS)
        d = ImageDraw.Draw(im, "RGBA")
        sc = VS / 2.3
        Xtr, ytr, Xte, yte = self.D
        for (px, py), c in zip(Xtr, ytr):
            dot(d, (px + 1.15) * sc, (1.15 - py) * sc, PINK if c > 0.5 else CYAN, True, 8)
        if t >= self.cfg["show_test"]:
            q = ease_out((t - self.cfg["show_test"]) / 0.6)
            for (px, py), c in zip(Xte, yte):
                dot(d, (px + 1.15) * sc, (1.15 - py) * sc, PINK if c > 0.5 else CYAN, False, int(5 + 5 * q))
        i = int(min(e, len(self.H["tr"]) - 1))
        d.text((30, 40), "EPOCH", font=font(F_MONO, 30), fill=(255, 255, 255, 230), anchor="lm", stroke_width=5, stroke_fill=INK)
        d.text((30, 98), f"{i}", font=font(F_BOLD, 96), fill=YEL + (255,), anchor="lm", stroke_width=8, stroke_fill=INK)
        return im

    def stats(self, t):
        i = int(min(self.e_at(t), len(self.H["tr"]) - 1))
        tr, te = self.H["tr"][i], self.H["te"][i]
        items = [(f"TRAIN {tr * 100:0.0f}%", YEL)]
        if t >= self.cfg["show_test"]:
            items.append((f"NEW DOTS {te * 100:0.0f}%", CYAN))
        return items, te if t >= self.cfg["show_test"] else tr, ("SCORE ON DOTS IT NEVER TRAINED ON" if t >= self.cfg["show_test"] else "SCORE ON ITS TRAINING DOTS"), CYAN if t >= self.cfg["show_test"] else YEL


def build_spiral(model):
    D, H = learn.run_spiral([32, 32], seed=1, epochs=300, lr=0.01, data_seed=0)
    te, tr = np.array(H["te"]), np.array(H["tr"])
    e95 = int(np.argmax(te >= 0.95))
    emap = [(0.8, 0), (4.0, 20), (11.0, 100), (15.5, e95 - 3), (17.0, e95 + 10), (21.0, 230), (25.5, 300)]
    t95 = inv(emap, e95)
    eS = int(pw(emap, 13.4))
    print("e95", e95, "t95", round(t95, 1), "te@sampled", te[eS], "tr", tr[eS], "final tr/te", tr[-1], te[-1], "ep0", tr[0], te[0])
    cfg = dict(T=34.0, c1=PINK, c2=CYAN, payoff=round(t95, 1), seed=11, D=D, H=H, emap=emap, show_test=7.6,
               still_times=[0.3, 3, 8, 11, round(t95 + 0.3, 1), 20, 26, 31],
               header=lambda t: (([("THIS AI KNOWS", (255, 255, 255)), ("NOTHING.", YEL)], "WATCH IT LEARN") if t < t95 else
                                 ([("IT LEARNED", (255, 255, 255)), ("A SPIRAL.", GREEN)], "TESTED ON NEW DOTS")))
    lines = [(0.2, "This AI knows nothing. Watch it learn a spiral."),
             (3.4, "Pink dots are one group, cyan dots are the other. It trains on the filled ones."),
             (7.8, "The hollow dots are new. It never trains on them, so they're the real test."),
             (12.4, f"For ages it's stuck, scoring only {te[eS] * 100:0.0f} percent on those."),
             (t95 + 0.2, f"There. It clicked. Now it gets {te[e95] * 100:0.0f} percent of new dots right."),
             (t95 + 4.6, "Each round it guesses, measures its error, and nudges its numbers. That's all training is."),
             (30.0, "Follow for more AI, explained in motion.")]
    buf, vo = make_audio(cfg, lines, model)
    reel = SpiralReel(cfg, vo)
    chime(buf, cfg["payoff"])
    post = dict(title="LEARN-1  This AI knows nothing. Watch it learn a spiral.",
                caption=f"A real tiny neural network (2 hidden layers, 32 neurons each) learns a spiral live. 300 dots to train on, 200 hidden dots it never trains on. It sits near {te[eS] * 100:.0f}% on the new dots for a long time, then clicks and reaches {te[-1] * 100:.0f}% on dots it has never seen. Guess, measure, nudge, repeat.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment SIM and I'll send the prompt to build this yourself.",
                tags=["ai", "neuralnetworks", "machinelearning", "deeplearning", "buildinpublic", "shipwithpip"],
                pinned="Build me a single-file HTML page with a small neural network (2 inputs, two hidden layers of 32 tanh neurons, 1 sigmoid output) trained live in the browser with Adam on a two-spiral dataset. Hold out 40% of the dots as a test set. Draw the decision boundary as a pink and cyan heatmap that updates every few epochs, show the training dots filled and the test dots hollow, and display epoch, train accuracy and test accuracy. Make it look neon and satisfying.",
                notes=f"train dots {len(D[0])}, test dots {len(D[2])} (random 60/40 split), net 2-32-32-1 tanh, Adam lr 0.01 full batch.\nEpoch 0: train {tr[0]:.3f}, test {te[0]:.3f}. Epoch {eS}: test {te[eS]:.3f}. First epoch >=95% test: {e95} (test {te[e95]:.3f}). Epoch 300: train {tr[-1]:.3f}, test {te[-1]:.3f}.\nHeld-out accuracy is on only {len(D[2])} dots, so it moves in 0.5% steps. Results varied only slightly across seeds 1-5 (all 99% test at 1500 epochs).")
    return reel, buf, cfg, post


# ===================================================================== 2. network paints a picture
class PaintReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.snaps, self.target = cfg["snaps"], cfg["target"]
        self.imgs = [Image.fromarray(s[3]).resize((VS, VS), Image.BICUBIC) for s in self.snaps]
        self.thumb = self.target.resize((230, 230), Image.LANCZOS)

    def idx(self, t):
        f = pw(self.cfg["imap"], t)
        return int(min(len(self.snaps) - 1, max(0, round(f))))

    def content(self, t):
        i = self.idx(t)
        im = self.imgs[i].copy()
        d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle([26, 26, 26 + 238, 26 + 238 + 46], radius=22, fill=(10, 6, 28, 215), outline=(255, 255, 255, 120), width=3)
        im.paste(self.thumb, (30, 30))
        d.text((26 + 119, 30 + 230 + 22), "TARGET", font=font(F_MONO, 26), fill=(255, 255, 255, 255), anchor="mm")
        d.text((VS - 30, 44), f"EPOCH {self.snaps[i][0]}", font=font(F_MONO, 40), fill=(255, 255, 255, 255), anchor="rm", stroke_width=5, stroke_fill=INK)
        return im

    def stats(self, t):
        i = self.idx(t)
        mae = self.snaps[i][2]
        frac = 1 - min(1, mae / self.snaps[0][2])
        return [(f"EPOCH {self.snaps[i][0]}", YEL), (f"COLOR ERROR {mae * 100:0.0f}%", PINK)], frac, "HOW CLOSE TO THE TARGET", PINK


def build_paint(model):
    sn = sorted(set(int(round(x)) for x in np.geomspace(1, 1200, 119)) | {0})
    im, out = paintnet.run(1200, snaps=set(sn), lr=0.004)
    es = [o[0] for o in out]
    mae = {o[0]: o[2] for o in out}
    nS = len(out)

    def idx_of(e):
        return min(range(nS), key=lambda k: abs(es[k] - e))
    imap = [(0.8, 0), (2.6, idx_of(3)), (8.0, idx_of(30)), (15.0, idx_of(150)), (21.0, idx_of(600)), (25.8, nS - 1)]
    eb = idx_of(60)
    t_blobs = inv(imap, eb)
    cfg = dict(T=35.0, c1=ORANGE, c2=PINK, payoff=round(inv(imap, idx_of(150)), 1), seed=14, snaps=out, target=im, imap=imap,
               still_times=[0.3, 3, 7, 10, 14, 19, 25, 30],
               header=lambda t: (([("THIS AI KNOWS", (255, 255, 255)), ("NOTHING.", YEL)], "WATCH IT PAINT") if t < 24.0 else
                                 ([("IT PAINTED", (255, 255, 255)), ("THE PICTURE.", GREEN)], "FROM SCRATCH")))
    print("blobs at epoch", es[eb], "t", round(t_blobs, 1), "mae0", out[0][2], "mae150", mae[es[idx_of(150)]], "final", out[-1][2], "payoff", cfg["payoff"])
    fin = out[-1]
    lines = [(0.2, "This AI knows nothing. Watch it learn to paint."),
             (3.4, "It never sees the picture. It only gets a spot on the canvas, and is told what color belongs there."),
             (9.6, "Round one is a messy pattern. Then big blobs of color start to appear."),
             (15.0, "The sun. The heart. The hills. Then the sharp edges fill in."),
             (20.0, f"After {fin[0]} rounds, each color is off by only about {fin[2] * 100:0.1f} percent."),
             (25.8, "That's called a neural field. NeRF, for 3D scenes, uses the same core idea."),
             (31.8, "Follow for more AI, explained in motion.")]
    buf, vo = make_audio(cfg, lines, model)
    reel = PaintReel(cfg, vo)
    chime(buf, cfg["payoff"])
    post = dict(title="LEARN-2  This AI knows nothing. Watch it learn to paint.",
                caption=f"A real tiny neural network learns to paint a picture. It never sees the image. For every spot on the canvas, it's just told what color belongs there. It starts as a messy pattern, then blobs of color, then sharp edges. After {fin[0]} rounds the average color error is {fin[2] * 100:.1f}%.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment PAINT and I'll send the prompt so you can build it.",
                tags=["ai", "neuralnetworks", "machinelearning", "creativecoding", "buildinpublic", "shipwithpip"],
                pinned="Build me a single-file HTML page where a small neural network (inputs: x and y plus sine and cosine features at 4 frequencies, 3 hidden layers of 64 tanh neurons, 3 outputs for RGB) learns to reproduce a simple 96x96 picture I draw in code (a sunset, a sun, a heart, dark hills). Train it live in the browser with Adam and redraw the network's output every few epochs next to the target. Show epoch and average color error.",
                notes=f"net: 20 inputs (x,y + sin/cos at freqs 1,2,4,8) -> 64 -> 64 -> 64 -> 3 tanh/linear, Adam lr 0.004, full batch of 9216 pixels, MSE loss.\n'Color error' = mean absolute difference between output and target colors, scale 0-1, shown as %. Epoch 0: {out[0][2] * 100:.1f}%. Epoch {es[idx_of(150)]}: {mae[es[idx_of(150)]] * 100:.1f}%. Epoch {fin[0]}: {fin[2] * 100:.1f}%.\nIt is fit to the pixels it trains on (a memorization-style demo by design, not a generalization claim).\nNeRF: published by Mildenhall et al. 2020, an MLP over coordinates with positional encoding. 'Same core idea' = coordinate network + positional encoding.")
    return reel, buf, cfg, post


# ===================================================================== 3. network sizes race
SC = [("TINY", 3, ORANGE), ("SMALL", 8, YEL), ("BIG", 32, GREEN)]


class SizesReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.R, self.D = cfg["R"], cfg["D"]

    def e_at(self, t):
        return pw(self.cfg["emap"], t)

    def content(self, t):
        e = self.e_at(t)
        im = Image.new("RGB", (VS, VS), (12, 7, 34))
        d = ImageDraw.Draw(im, "RGBA")
        pan, gap = 296, 36
        Xte, yte = self.D[2], self.D[3]
        for k, (nm, w, col) in enumerate(SC):
            x0 = k * (pan + gap)
            im.paste(heat_img(interp_grid(self.R[k], e), pan), (x0, 96))
            dd = ImageDraw.Draw(im, "RGBA")
            dd.rounded_rectangle([x0, 96, x0 + pan, 96 + pan], radius=14, outline=col + (255,), width=5)
            sc = pan / 2.3
            for (px, py), c in zip(Xte, yte):
                dot(dd, x0 + (px + 1.15) * sc, 96 + (1.15 - py) * sc, PINK if c > 0.5 else CYAN, False, 4)
            d.text((x0 + pan / 2, 52), f"{w} NEURONS", font=font(F_MONO, 30), fill=col + (255,), anchor="mm")
            i = int(min(e, len(self.R[k]["te"]) - 1))
            d.text((x0 + pan / 2, 96 + pan + 52), f"{self.R[k]['te'][i] * 100:0.0f}%", font=font(F_BOLD, 76), fill=col + (255,), anchor="mm", stroke_width=6, stroke_fill=INK)
        # chart
        cx0, cx1, cy0, cy1 = 90, VS - 30, 590, 910
        d.text((cx0 - 10, 520), "SCORE ON NEW DOTS", font=font(F_MONO, 28), fill=(255, 255, 255, 220), anchor="lm")
        for v in (0.5, 0.75, 1.0):
            y = cy1 - (v - 0.4) / 0.6 * (cy1 - cy0)
            d.line([(cx0, y), (cx1, y)], fill=(255, 255, 255, 50), width=2)
            d.text((cx0 - 12, y), f"{v * 100:0.0f}", font=font(F_MONO, 24), fill=(255, 255, 255, 190), anchor="rm")
        ne = self.cfg["emax"]
        for k, (nm, w, col) in enumerate(SC):
            te = self.R[k]["te"]
            n = int(min(e, len(te) - 1))
            pts = [(cx0 + j / ne * (cx1 - cx0), cy1 - (max(te[j], 0.4) - 0.4) / 0.6 * (cy1 - cy0)) for j in range(0, n + 1)]
            if len(pts) > 1:
                d.line(pts, fill=col + (255,), width=7, joint="curve")
            x, y = pts[-1]
            d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=col + (255,), outline=(255, 255, 255, 255), width=3)
        d.text((cx0, cy1 + 32), "0", font=font(F_MONO, 24), fill=(255, 255, 255, 190), anchor="mm")
        d.text((cx1, cy1 + 32), f"EPOCH {ne}", font=font(F_MONO, 24), fill=(255, 255, 255, 190), anchor="rm")
        return im

    def stats(self, t):
        i = int(min(self.e_at(t), self.cfg["emax"]))
        best = max(r["te"][min(i, len(r["te"]) - 1)] for r in self.R)
        return [(f"EPOCH {i}", YEL), ("HOLLOW DOTS = NEW", CYAN)], best, "BEST SCORE ON NEW DOTS", GREEN


def build_sizes(model):
    EM = 400
    R = []
    D = None
    for nm, w, col in SC:
        D, H = learn.run_spiral([w, w], seed=1, epochs=EM, lr=0.01, data_seed=0)
        R.append(H)
    te = [np.array(h["te"]) for h in R]
    first = [int(np.argmax(x >= 0.95)) if (x >= 0.95).any() else None for x in te]
    print("first>=95", first, "final te", [x[-1] for x in te], "final tr", [h["tr"][-1] for h in R], "max tiny", te[0].max())
    assert first[2] is not None and first[1] is not None and first[0] is None
    emap = [(0.8, 0), (8.0, 60), (14.0, 132), (21.0, 260), (24.5, 331), (27.0, EM)]
    t32, t8 = inv(emap, first[2]), inv(emap, first[1])
    cfg = dict(T=35.0, c1=GREEN, c2=ORANGE, payoff=round(t32, 1), seed=21, R=R, D=D, emap=emap, emax=EM,
               still_times=[0.3, 4, 8, round(t32 + 0.4, 1), 17, round(t8 + 0.4, 1), 26, 31],
               header=lambda t: (([("SAME TASK.", (255, 255, 255)), ("3 BRAIN SIZES.", YEL)], "WHO LEARNS IT?") if t < 24.5 else
                                 ([("TINY BRAINS", (255, 255, 255)), ("GET STUCK.", PINK)], "TESTED ON NEW DOTS")))
    lines = [(0.2, "Same task. Same spiral. Three AI brains, three sizes."),
             (4.8, "The hollow dots are new ones. The score is only on those."),
             (8.8, "The tiny brain has just three neurons per layer. The big one has thirty two."),
             (13.0, f"Watch the big one. It cracks it first, at epoch {first[2]}, with {te[2][first[2]] * 100:0.0f} percent."),
             (18.5, f"And the tiny brain? It never gets past about {te[0].max() * 100:0.0f} percent. Not enough neurons to bend around the spiral."),
             (t8 + 1.0, f"The small one gets there too, but later, at epoch {first[1]}."),
             (t8 + 5.2, "So size matters. Follow for more AI, explained in motion.")]
    buf, vo = make_audio(cfg, lines, model)
    reel = SizesReel(cfg, vo)
    chime(buf, cfg["payoff"])
    post = dict(title="LEARN-3  Same task. 3 AI brain sizes.",
                caption=f"Same spiral, same training, three network sizes (3, 8 and 32 neurons per layer). The big one cracks it first, the small one gets there later, and the tiny one never gets past {te[0].max() * 100:.0f}% on new dots. In this run. Results shift a bit with the random start, but the tiny net failed every time I tried.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nWhich brain would you bet on? Comment 3, 8 or 32.",
                tags=["ai", "neuralnetworks", "machinelearning", "deeplearning", "buildinpublic", "shipwithpip"],
                pinned="Comment 3, 8 or 32 - which brain would you bet on? Want the prompt to build this race yourself? Comment RACE.",
                notes=f"nets 2-w-w-1 tanh, w in 3, 8, 32; Adam lr 0.01 full batch; same data (300 train, 200 held-out test dots), seed 1.\nFirst epoch >=95% on test: 32 neurons: {first[2]}, 8 neurons: {first[1]}, 3 neurons: never (max {te[0].max():.3f}).\nFinal test acc @epoch {EM}: 3: {te[0][-1]:.3f}, 8: {te[1][-1]:.3f}, 32: {te[2][-1]:.3f}.\nAcross seeds 1-5 (other run): 32 reached 95% test at epochs 132-166; 8 at 199-331; 3 neurons never did (final test 0.62-0.845 at 800 epochs).")
    return reel, buf, cfg, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = {"spiral": build_spiral, "paint": build_paint, "sizes": build_sizes}[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
