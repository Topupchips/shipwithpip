"""Informative explainers with real computed visuals: temperature, BPE tokens, backprop.
usage: python infofx.py temp|bpe|backprop out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
"""
import argparse
import math
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import pw, write_post  # noqa: E402
from scenes import CYAN, GREEN, PINK, VIOLET, YEL, ORANGE, Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, INK, VS, clamp, ease_out, font, tone  # noqa: E402

BG = (12, 7, 34)
PAL = [PINK, CYAN, YEL, GREEN, ORANGE, VIOLET, (255, 120, 120), (120, 200, 255), (200, 255, 140), (255, 180, 255), (255, 210, 120), (140, 255, 220)]


def panel(d, box, col=(18, 10, 48, 255), outline=(255, 255, 255, 40)):
    d.rounded_rectangle(box, radius=24, fill=col, outline=outline, width=2)


# =================================================================== 1. temperature
WORDS = [("mat", 22), ("sofa", 9), ("floor", 7), ("bed", 5), ("roof", 3), ("table", 3), ("moon", 1)]


def probs(T):
    c = np.array([w[1] for w in WORDS], float)
    z = np.log(c) / T
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


class TempReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.u = np.random.default_rng(11).random(10)

    def T_at(self, t):
        return pw(self.cfg["tmap"], t)

    def content(self, t):
        T = self.T_at(t)
        p = probs(T)
        im = Image.new("RGB", (VS, VS), BG)
        d = ImageDraw.Draw(im, "RGBA")
        d.text((VS / 2, 62), "THE CAT SAT ON THE ...", font=font(F_BOLD, 50), fill=(255, 255, 255, 255), anchor="mm")
        top = 120
        for i, ((w, _), pi) in enumerate(zip(WORDS, p)):
            y = top + i * 60
            d.text((30, y + 24), w, font=font(F_MONO, 34), fill=(255, 255, 255, 255), anchor="lm")
            d.rounded_rectangle([190, y + 4, 190 + 560, y + 44], radius=10, fill=(26, 16, 62, 255))
            col = PAL[i % len(PAL)]
            d.rounded_rectangle([190, y + 4, 190 + max(10, 560 * float(pi)), y + 44], radius=10, fill=col + (255,))
            d.text((930, y + 24), f"{pi * 100:0.0f}%", font=font(F_MONO, 34), fill=(255, 255, 255, 255), anchor="rm")
        # temperature dial
        gy = 600
        d.text((30, gy), "TEMPERATURE", font=font(F_MONO, 30), fill=(255, 255, 255, 220), anchor="lm")
        d.text((930, gy), f"{T:0.1f}", font=font(F_BOLD, 72), fill=YEL + (255,), anchor="rm", stroke_width=6, stroke_fill=INK)
        d.rounded_rectangle([30, gy + 46, 930, gy + 70], radius=12, fill=(26, 16, 62, 255))
        q = (math.log(T) - math.log(0.15)) / (math.log(3.0) - math.log(0.15))
        d.rounded_rectangle([30, gy + 46, 30 + 900 * clamp(q), gy + 70], radius=12, fill=(PINK if T > 1.3 else (CYAN if T < 0.7 else GREEN)) + (255,))
        d.ellipse([30 + 900 * clamp(q) - 18, gy + 58 - 18, 30 + 900 * clamp(q) + 18, gy + 58 + 18], fill=(255, 255, 255, 255))
        d.text((30, gy + 100), "SAFE", font=font(F_MONO, 24), fill=CYAN + (255,), anchor="lm")
        d.text((930, gy + 100), "WILD", font=font(F_MONO, 24), fill=PINK + (255,), anchor="rm")
        # 10 samples
        cdf = np.cumsum(p)
        picks = [WORDS[int(min(np.searchsorted(cdf, u), len(WORDS) - 1))][0] for u in self.u]
        d.text((30, 770), "10 ROLLS OF THE DICE:", font=font(F_MONO, 28), fill=(255, 255, 255, 220), anchor="lm")
        x, y = 30, 810
        for k, w in enumerate(picks):
            idx = [a for a, _ in WORDS].index(w)
            tw_ = d.textlength(w, font=font(F_MONO, 30)) + 34
            if x + tw_ > 940:
                x, y = 30, y + 54
            d.rounded_rectangle([x, y, x + tw_, y + 44], radius=22, fill=PAL[idx % len(PAL)] + (255,))
            d.text((x + tw_ / 2, y + 22), w, font=font(F_MONO, 30), fill=INK + (255,), anchor="mm")
            x += tw_ + 10
        return im

    def stats(self, t):
        T = self.T_at(t)
        p = probs(T)
        ent = float(-(p * np.log(p + 1e-12)).sum())
        return [(f"TEMP {T:0.1f}", YEL), (f"~{math.exp(ent):0.1f} REAL CHOICES", CYAN)], float(p.max()), "CHANCE OF THE TOP WORD", PINK


def build_temp(model):
    p02, p1, p25 = probs(0.2), probs(1.0), probs(2.5)
    ent = lambda p: float(math.exp(-(p * np.log(p + 1e-12)).sum()))  # noqa: E731
    print("top p @0.2/1.0/2.5", p02.max(), p1.max(), p25.max(), "eff", ent(p02), ent(p1), ent(p25))
    tmap = [(0, 1.0), (7.0, 1.0), (9.5, 0.2), (15.5, 0.2), (19.5, 2.5), (26.0, 2.5), (29.0, 1.0), (40, 1.0)]
    cfg = dict(T=35.0, c1=ORANGE, c2=VIOLET, tmap=tmap, payoff=None, seed=71, still_times=[0.5, 5, 9.5, 12, 17, 21, 27, 31],
               header=lambda t: (([("AI PICKS WORDS", (255, 255, 255)), ("FROM A LIST.", YEL)], "HOW AI WRITES") if t < 15.5 else
                                 ([("TEMPERATURE", (255, 255, 255)), ("= THE DIAL.", PINK)], "SAFE VS WILD")))
    lines = [(0.2, "When AI writes, it doesn't just pick a word. It picks from a list of probabilities."),
             (6.4, "Here's a toy example. The cat sat on the... what?"),
             (10.0, f"Turn the temperature down, and the top word takes almost everything. {p02.max() * 100:0.0f} percent. Same answer every time."),
             (16.8, f"Turn it up, and the list flattens. Now the top word has only {p25.max() * 100:0.0f} percent, and strange words get a real chance."),
             (23.7, "The math is simple. The model's scores are divided by the temperature before they become probabilities."),
             (30.4, "Low is reliable. High is creative, or just weird."),
             (33.6, "Follow for more.")]
    buf, vo = make_audio(cfg, lines, model)
    reel = TempReel(cfg, vo)
    post = dict(title="INFO-1  What AI temperature actually does",
                caption=f"AI doesn't pick a word, it picks from a list of probabilities. Temperature reshapes that list: scores are divided by the temperature before they become probabilities. Low temperature = the top word wins almost every time. High temperature = flatter list, stranger picks. Toy example with made-up scores for 'The cat sat on the ...'.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment TEMP and I'll send the prompt so you can build this slider yourself.",
                tags=["ai", "chatgpt", "llm", "aitips", "machinelearning", "shipwithpip"],
                pinned="Build me a single-file HTML page with a temperature slider (0.1 to 3). Show 7 candidate next words as bars for 'The cat sat on the ...' using made-up counts (22, 9, 7, 5, 3, 3, 1). Probabilities = softmax(log(count) / temperature). Show 10 fixed random dice rolls re-sampled live as the slider moves, and the 'real choices' number exp(entropy).",
                notes=f"TOY example: the next-word scores (counts 22, 9, 7, 5, 3, 3, 1) are made up for illustration, not from a real model. Probabilities = softmax(log(count)/T), computed exactly. Top-word probability: T=0.2 -> {p02.max() * 100:.1f}%, T=1.0 -> {p1.max() * 100:.1f}%, T=2.5 -> {p25.max() * 100:.1f}%. 'Real choices' = exp(entropy): {ent(p02):.2f}, {ent(p1):.2f}, {ent(p25):.2f}.\nReal LLMs apply the same temperature scaling to their logits over a vocabulary of tens of thousands of tokens or more.")
    return reel, buf, cfg, post


# =================================================================== 2. BPE tokens
CORPUS = [("building", 5), ("builder", 3), ("builds", 4), ("app", 6), ("apps", 4), ("application", 2)]


def bpe_steps(n_merges):
    words = {w: list(w) + ["_"] for w, _ in CORPUS}
    cnt = dict(CORPUS)
    states = [{w: list(t) for w, t in words.items()}]
    merges = []
    for _ in range(n_merges):
        pc = Counter()
        for w, toks in words.items():
            for a, b in zip(toks, toks[1:]):
                pc[(a, b)] += cnt[w]
        if not pc:
            break
        best = max(pc.items(), key=lambda kv: (kv[1], kv[0]))
        (a, b), n = best
        for w, toks in words.items():
            i, out = 0, []
            while i < len(toks):
                if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(toks[i])
                    i += 1
            words[w] = out
        merges.append((a, b, n))
        states.append({w: list(t) for w, t in words.items()})
    return states, merges


class BpeReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.states, self.merges = cfg["states"], cfg["merges"]
        self.col = {}

    def step_at(self, t):
        c = self.cfg
        k = int((t - c["t0"]) // c["D"]) if t >= c["t0"] else -1
        ph = ((t - c["t0"]) % c["D"]) / c["D"] if t >= c["t0"] else 0
        return min(max(k, -1), len(self.merges)), ph

    def totals(self, st):
        tot = sum(c * len(st[w]) for w, c in CORPUS)
        vocab = len({tok for w in st for tok in st[w]})
        return tot, vocab

    def tcol(self, tok):
        if len(tok.replace("_", "")) <= 1 and tok != "_":
            return (40, 28, 90)
        if tok == "_":
            return (40, 28, 90)
        if tok not in self.col:
            self.col[tok] = PAL[len(self.col) % len(PAL)]
        return self.col[tok]

    def content(self, t):
        k, ph = self.step_at(t)
        done = k if k >= 0 else 0
        before = self.states[min(done, len(self.states) - 1)]
        merging = 0 <= k < len(self.merges) and ph < 0.5
        pair = self.merges[k][:2] if 0 <= k < len(self.merges) else None
        st = self.states[min(k + 1, len(self.states) - 1)] if (0 <= k and not merging) else before
        if k < 0:
            st = self.states[0]
        im = Image.new("RGB", (VS, VS), BG)
        d = ImageDraw.Draw(im, "RGBA")
        d.text((VS / 2, 44), "MOST COMMON PAIR:" if k >= 0 and k < len(self.merges) else "TEXT = TOKENS", font=font(F_MONO, 28), fill=(255, 255, 255, 220), anchor="mm")
        if 0 <= k < len(self.merges):
            a, b, n = self.merges[k]
            d.text((VS / 2, 100), f"{a.replace('_', '·')} + {b.replace('_', '·')}  ({n}x)", font=font(F_BOLD, 56), fill=YEL + (255,), anchor="mm", stroke_width=5, stroke_fill=INK)
        f = font(F_MONO, 40)
        y0 = 170
        for i, (w, c) in enumerate(CORPUS):
            y = y0 + i * 120
            d.text((30, y + 40), f"{c}x", font=font(F_MONO, 28), fill=(255, 255, 255, 170), anchor="lm")
            x = 110
            toks = st[w]
            # during highlight phase, mark pair occurrences from `before`
            seq = before[w] if merging else toks
            for j, tk in enumerate(seq):
                lab = tk.replace("_", "·")
                tw_ = d.textlength(lab, font=f) + 26
                hl = merging and pair and j < len(seq) - 1 and seq[j] == pair[0] and seq[j + 1] == pair[1]
                hl2 = merging and pair and j > 0 and seq[j - 1] == pair[0] and seq[j] == pair[1]
                col = self.tcol(tk)
                if hl or hl2:
                    col = YEL
                is_new = (not merging) and 0 <= k < len(self.merges) and tk == self.merges[k][0] + self.merges[k][1] and ph < 0.7
                pop = 1.0 + (0.12 * math.sin(ph / 0.7 * math.pi) if is_new else 0)
                yy = y + 40
                hh = 36 * pop
                d.rounded_rectangle([x, yy - hh, x + tw_ * pop, yy + hh], radius=14, fill=col + (255,), outline=(255, 255, 255, 80 if col != YEL else 255), width=2)
                dark = sum(col) > 380
                d.text((x + tw_ * pop / 2, yy), lab, font=f, fill=(INK if dark else (255, 255, 255)) + (255,), anchor="mm")
                x += tw_ * pop + 6
        return im

    def stats(self, t):
        k, ph = self.step_at(t)
        idx = max(0, min(k + (1 if ph >= 0.5 and k >= 0 else 0), len(self.states) - 1))
        tot, vocab = self.totals(self.states[idx])
        t0tot, _ = self.totals(self.states[0])
        return [(f"TOKENS {tot}", YEL), (f"VOCAB {vocab}", CYAN)], 1 - tot / t0tot, "TEXT SHRINKS AS TOKENS MERGE", GREEN


def build_bpe(model):
    NM = 10
    states, merges = bpe_steps(NM)
    tot0 = sum(c * len(states[0][w]) for w, c in CORPUS)
    totN = sum(c * len(states[-1][w]) for w, c in CORPUS)
    print("merges", [(a, b, n) for a, b, n in merges], "tokens", tot0, "->", totN)
    single = {}
    for k, st in enumerate(states):
        for w in st:
            if len(st[w]) == 1 and w not in single:
                single[w] = k
    print("single-token words (merge index)", single)
    t0, D = 8.2, 1.6
    cfg = dict(T=70.0, c1=GREEN, c2=CYAN, t0=t0, D=D, states=states, merges=merges, payoff=None, seed=81, still_times=[0.5, 5, 10, 13, 18, 23, 28, 33],
               header=lambda t: (([("AI DOESN'T READ", (255, 255, 255)), ("LETTERS.", YEL)], "HOW AI READS") if t < t0 + D * NM else
                                 ([("IT READS", (255, 255, 255)), ("TOKENS.", GREEN)], "BYTE-PAIR ENCODING")))
    first_w = min(single, key=single.get) if single else None
    lines = [(0.2, "AI doesn't read letters or words. It reads tokens."),
             (3.7, "Here's how it learns them. Merge the most common pair, then repeat."),
             (t0 + 0.2, "Watch the most common pair merge."),
             ]
    if first_w is not None:
        lines.append((t0 + D * (single[first_w]) + 0.6, f"Now '{first_w}' is a single token."))
    lines += [(t0 + D * NM + 0.4, f"After {NM} merges, the same text went from {tot0} tokens to {totN}."),
              (t0 + D * NM + 5.6, "GPT-2 used this idea, called byte-pair encoding, with about fifty thousand tokens."),
              ]
    lines.append((t0 + D * NM + 11.0, "It's one reason AI can stumble on counting letters. It sees tokens, not letters."))
    buf, vo = make_audio(cfg, lines, model)
    for k in range(NM):
        tone(buf, t0 + k * D + 0.5 * D, 0.2, 380 + 40 * k, 520 + 40 * k, 0.07)
    chime(buf, t0 + NM * D)
    cfg["T"] = vo[-1][1] + 2.4
    reel = BpeReel(cfg, vo)
    post = dict(title="INFO-2  How AI reads: tokens (byte-pair encoding)",
                caption=f"AI doesn't read letters or words, it reads tokens. A tokenizer learns them with a simple loop: find the most common pair of symbols, merge it into one, repeat. In this tiny demo, {tot0} tokens became {totN} after {NM} merges. GPT-2 used byte-level byte-pair encoding with a vocabulary of about 50,000 tokens.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment TOKENS and I'll send the prompt so you can build this yourself.",
                tags=["ai", "chatgpt", "llm", "tokenization", "aitips", "shipwithpip"],
                pinned="Build me a single-file HTML page that demonstrates byte-pair encoding. Start with the words building x5, builder x3, builds x4, app x6, apps x4, application x2 split into characters plus an end-of-word marker. Each step, find the most common adjacent pair weighted by word count, highlight it, merge it, and show total token count and vocabulary size. Do 10 merges.",
                notes=f"Real BPE run on a tiny corpus ({', '.join(f'{w} x{c}' for w, c in CORPUS)}). Merges in order: {[(a, b, n) for a, b, n in merges]}. Tokens {tot0} -> {totN}.\nBPE in compression: Gage 1994; adapted for subword units in NLP by Sennrich et al. 2015/16; GPT-2 used byte-level BPE with a 50,257-token vocabulary. 'One reason AI stumbles at counting letters' is a commonly cited explanation, hedged on purpose.")
    return reel, buf, cfg, post


# =================================================================== 3. backprop
class Net:
    def __init__(self, seed=7, H=3):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 0.9, (2, H))
        self.b1 = np.zeros(H)
        self.W2 = r.normal(0, 0.9, (H, 1))
        self.b2 = np.zeros(1)

    def fwd(self, x):
        z1 = x @ self.W1 + self.b1
        h = np.tanh(z1)
        z2 = h @ self.W2 + self.b2
        o = 1 / (1 + np.exp(-z2))
        return h, o

    def grads(self, x, y):
        h, o = self.fwd(x)
        dz2 = o - y
        gW2 = np.outer(h, dz2)
        gb2 = dz2
        dh = (self.W2 @ dz2)
        dz1 = dh * (1 - h ** 2)
        gW1 = np.outer(x, dz1)
        return h, o, dz2, dh, dz1, gW1, dz1.copy(), gW2, gb2

    def copy(self):
        n = Net.__new__(Net)
        n.W1, n.b1, n.W2, n.b2 = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()
        return n


XS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
YS = np.array([0, 1, 1, 0], float)


def bce(o, y):
    return float(-(y * math.log(o + 1e-9) + (1 - y) * math.log(1 - o + 1e-9)))


class BackReel(Reel2):
    NX = [(150, 330), (150, 630)]
    NH = [(480, 230), (480, 480), (480, 730)]
    NO = (810, 480)

    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)

    def phase(self, t):
        for name, a, b in self.cfg["phases"]:
            if a <= t < b:
                return name, (t - a) / (b - a)
        return "end", 1.0

    def edge(self, d, p0, p1, w, col_extra=None, pulse=None, rev=False):
        col = PINK if w >= 0 else CYAN
        wd = int(3 + min(18, abs(w) * 5))
        d.line([p0, p1], fill=col + (220,), width=wd)
        if pulse is not None:
            q = 1 - pulse if rev else pulse
            x, y = p0[0] + (p1[0] - p0[0]) * q, p0[1] + (p1[1] - p0[1]) * q
            c = YEL if not rev else ORANGE
            d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=c + (255,), outline=(255, 255, 255, 255), width=3)

    def node(self, d, p, label, col=(30, 20, 70), r=44):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col + (255,), outline=(255, 255, 255, 220), width=3)
        d.text(p, label, font=font(F_MONO, 28), fill=(255, 255, 255, 255), anchor="mm")

    def content(self, t):
        c = self.cfg
        im = Image.new("RGB", (VS, VS), BG)
        d = ImageDraw.Draw(im, "RGBA")
        name, q = self.phase(t)
        cyc = c["cyc_at"](t)
        Wn = c["net_at"](t)
        x = c["x_ex"]
        y = c["y_ex"]
        g = Wn.grads(x, y)
        h, o = g[0], float(g[1][0])
        fast = name in ("fast", "end")
        d.text((VS / 2, 44), "INPUT (1, 0).  CORRECT ANSWER: 1" if not fast else "XOR: 4 EXAMPLES, 1 NETWORK", font=font(F_MONO, 28), fill=(255, 255, 255, 230), anchor="mm")
        fwd_p = q if name.startswith("fwd") else (1.0 if name in ("loss", "bwd", "upd") else 0)
        bwd_p = q if name == "bwd" else (1.0 if name == "upd" else 0)
        for i, px in enumerate(self.NX):
            for j, ph_ in enumerate(self.NH):
                w = Wn.W1[i, j]
                pul = None
                if name.startswith("fwd") and x[i] > 0:
                    pul = clamp(q * 1.6)
                elif name == "bwd":
                    pul = clamp((q * 1.6) - 0.6)
                self.edge(d, px, ph_, w, pulse=pul if (name.startswith("fwd") and x[i] > 0) or name == "bwd" else None, rev=(name == "bwd"))
                mx, my = (px[0] + ph_[0]) / 2 - 10 + (i * 30 - 15), (px[1] + ph_[1]) / 2 + (-10 if i == 0 else 10)
                txt = f"{w:+.2f}"
                if name == "bwd" and bwd_p > 0.5:
                    txt = f"g{g[5][i, j]:+.2f}"
                d.text((mx, my), txt, font=font(F_MONO, 25), fill=(255, 255, 255, 255), anchor="mm", stroke_width=4, stroke_fill=INK)
        for j, ph_ in enumerate(self.NH):
            w = Wn.W2[j, 0]
            pul = clamp(q * 1.6 - 0.6) if name.startswith("fwd") else (clamp(q * 1.6) if name == "bwd" else None)
            self.edge(d, ph_, self.NO, w, pulse=pul if (name.startswith("fwd") or name == "bwd") else None, rev=(name == "bwd"))
            mx, my = (ph_[0] + self.NO[0]) / 2, (ph_[1] + self.NO[1]) / 2 - 18
            txt = f"{w:+.2f}"
            if name == "bwd" and bwd_p > 0.3:
                txt = f"g{g[7][j, 0]:+.2f}"
            d.text((mx, my), txt, font=font(F_MONO, 25), fill=(255, 255, 255, 255), anchor="mm", stroke_width=4, stroke_fill=INK)
        for i, p in enumerate(self.NX):
            self.node(d, p, f"{int(x[i])}")
        for j, p in enumerate(self.NH):
            self.node(d, p, f"{h[j]:+.1f}" if fwd_p > 0.45 or fast else "?")
        ocol = (30, 20, 70)
        self.node(d, self.NO, f"{o:.2f}" if fwd_p > 0.9 or fast else "?", col=ocol, r=52)
        # captions per phase
        tag = {"intro": "A NETWORK THAT KNOWS NOTHING", "fwd1": "1. FORWARD: NUMBERS FLOW RIGHT", "loss": f"2. HOW WRONG? LOSS = {bce(o, y):.2f}", "bwd": "3. BACKWARD: BLAME FLOWS LEFT",
               "upd": "4. NUDGE EVERY WEIGHT A LITTLE", "fwd2": "AGAIN: THE ANSWER MOVED CLOSER", "fast": "REPEAT... AND REPEAT"}.get(name, "REPEAT")
        if name == "end":
            tag = "IT LEARNED XOR"
        d.rounded_rectangle([30, 850, 930, 920], radius=24, fill=(26, 16, 62, 255), outline=YEL + (200,), width=3)
        d.text((VS / 2, 885), tag, font=font(F_BOLD, 36), fill=YEL + (255,), anchor="mm")
        if fast:
            e = int(c["epoch_at"](t))
            ep = c["hist"]
            e = min(e, len(ep["loss"]) - 1)
            # mini table of the 4 predictions
            for k in range(4):
                oy_ = ep["pred"][e][k]
                yy = 60 + k * 0
            bx0 = 30
            for k in range(4):
                xx = bx0 + k * 225
                ok = abs(ep["pred"][e][k] - YS[k]) < 0.1
                d.rounded_rectangle([xx, 760, xx + 210, 830], radius=16, fill=(18, 10, 48, 255), outline=(GREEN if ok else PINK) + (255,), width=3)
                d.text((xx + 105, 782), f"{int(XS[k][0])},{int(XS[k][1])} = {int(YS[k])}", font=font(F_MONO, 22), fill=(255, 255, 255, 220), anchor="mm")
                d.text((xx + 105, 812), f"{ep['pred'][e][k]:.2f}", font=font(F_BOLD, 34), fill=(GREEN if ok else PINK) + (255,), anchor="mm")
            d.text((VS - 30, 100), f"EPOCH {e}", font=font(F_MONO, 36), fill=(255, 255, 255, 255), anchor="rm", stroke_width=5, stroke_fill=INK)
            d.text((VS - 30, 146), f"loss {ep['loss'][e]:.3f}", font=font(F_MONO, 30), fill=(255, 255, 255, 200), anchor="rm", stroke_width=5, stroke_fill=INK)
        return im

    def stats(self, t):
        name, q = self.phase(t)
        c = self.cfg
        if name in ("fast", "end"):
            e = min(int(c["epoch_at"](t)), len(c["hist"]["loss"]) - 1)
            L = c["hist"]["loss"][e]
            return [(f"EPOCH {e}", YEL), (f"LOSS {L:.3f}", PINK)], 1 - min(1, L / c["hist"]["loss"][0]), "HOW LOW THE ERROR GOT", GREEN
        w = c["net_at"](t)
        o = float(w.fwd(c["x_ex"])[1][0])
        return [(f"ANSWER {o:.2f}", YEL), ("TARGET 1", CYAN)], o, "ANSWER (1.00 = PERFECT)", PINK


def build_backprop(model):
    net = Net(seed=7)
    x, y = XS[2], 1.0
    nets = [net.copy()]
    lr = 1.0
    cur = net.copy()
    for _ in range(3):
        g = cur.grads(x, y)
        cur.W1 -= lr * g[5]
        cur.b1 -= lr * g[6]
        cur.W2 -= lr * g[7]
        cur.b2 -= lr * g[8]
        nets.append(cur.copy())
    o_list = [float(n.fwd(x)[1][0]) for n in nets]
    print("single-example answers", [round(o, 3) for o in o_list], "loss", [round(bce(o, 1.0), 3) for o in o_list])
    # full-batch training from nets[3]
    tr = nets[3].copy()
    hist = dict(loss=[], pred=[])
    for e in range(1200):
        hs = np.tanh(XS @ tr.W1 + tr.b1)
        o = (1 / (1 + np.exp(-(hs @ tr.W2 + tr.b2)))).ravel()
        hist["loss"].append(float(np.mean([bce(o[k], YS[k]) for k in range(4)])))
        hist["pred"].append(o.copy())
        d2 = (o - YS)[:, None] / 4
        gW2 = hs.T @ d2
        gb2 = d2.sum(0)
        dh = d2 @ tr.W2.T * (1 - hs ** 2)
        tr.W1 -= 0.8 * (XS.T @ dh)
        tr.b1 -= 0.8 * dh.sum(0)
        tr.W2 -= 0.8 * gW2
        tr.b2 -= 0.8 * gb2
    L = np.array(hist["loss"])
    e_good = int(np.argmax(np.all(np.abs(np.array(hist["pred"]) - YS) < 0.1, axis=1)))
    print("all four within 0.1 at epoch", e_good, "loss0", L[0], "final", L[-1], "preds", np.round(hist["pred"][-1], 3))
    phases = [("intro", 0, 4.0), ("fwd1", 4.0, 9.0), ("loss", 9.0, 14.0), ("bwd", 14.0, 21.0), ("upd", 21.0, 24.5), ("fwd2", 24.5, 28.5), ("fast", 28.5, 37.0), ("end", 37.0, 99)]

    def net_at(t):
        if t < 21.0:
            return nets[0]
        if t < 24.5:
            a = (t - 21.0) / 3.5
            n = nets[0].copy()
            n.W1 = nets[0].W1 * (1 - a) + nets[1].W1 * a
            n.b1 = nets[0].b1 * (1 - a) + nets[1].b1 * a
            n.W2 = nets[0].W2 * (1 - a) + nets[1].W2 * a
            n.b2 = nets[0].b2 * (1 - a) + nets[1].b2 * a
            return n
        if t < 28.5:
            return nets[1]
        return nets[3]
    epoch_at = lambda t: pw([(28.5, 0), (31.0, 30), (33.5, 150), (36.5, min(e_good + 60, 1199)), (99, min(e_good + 60, 1199))], t)  # noqa: E731
    cfg = dict(T=45.0, c1=VIOLET, c2=PINK, phases=phases, net_at=net_at, cyc_at=lambda t: 0, epoch_at=epoch_at, hist=hist, x_ex=x, y_ex=1.0, payoff=None, seed=91,
               still_times=[0.5, 6, 11, 17, 22, 26, 32, 40],
               header=lambda t: (([("HOW AI LEARNS:", (255, 255, 255)), ("BACKPROP.", YEL)], "ONE STEP AT A TIME") if t < 37.0 else
                                 ([("GUESS. BLAME.", (255, 255, 255)), ("NUDGE.", GREEN)], "REPEAT THOUSANDS OF TIMES")))
    lines = [(0.2, "This tiny network knows nothing. Its weights are random."),
             (4.4, "Step one. Feed in an input, and the numbers flow forward."),
             (9.4, f"Step two. How wrong? It answers {o_list[0]:.2f}, but the right answer is one."),
             (15.1, "Step three. Send the blame backward, so every weight learns how much it contributed to the mistake."),
             (22.4, "Step four. Nudge every weight a little, to shrink the error."),
             (27.0, f"Run it again, and the answer moves from {o_list[0]:.2f} to {o_list[1]:.2f}."),
             (32.2, f"Now repeat on all four examples. After about {e_good} rounds, all four answers are right, within a tenth."),
             (39.0, "That loop is how neural networks learn. Follow for more.")]
    buf, vo = make_audio(cfg, lines, model)
    chime(buf, 37.0)
    reel = BackReel(cfg, vo)
    post = dict(title="INFO-3  How AI learns: backpropagation, step by step",
                caption=f"A tiny neural network learning XOR. Forward pass, measure the error, send the blame backward, nudge every weight. Repeat. On one example the answer moves from {o_list[0]:.2f} toward 1 in a single step, and after about {e_good} full rounds on all four examples it's right to within 0.1.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment BACKPROP and I'll send the prompt so you can build it.",
                tags=["ai", "neuralnetworks", "machinelearning", "backpropagation", "deeplearning", "shipwithpip"],
                pinned="Build me a single-file HTML page that animates one step of backpropagation on a 2-3-1 network (tanh hidden, sigmoid output). Show the weights on the edges, animate the forward pass, show the loss, animate the backward pass with the real gradients on each edge, then update the weights with learning rate 1.0. After 3 steps, fast-forward training on the 4 XOR examples and show the 4 predictions and the loss.",
                notes=f"Network 2-3-1 (tanh hidden, sigmoid output), binary cross-entropy loss, seed 7. Single example (1,0)->1, lr 1.0: answers {[round(o, 3) for o in o_list]}. Then full-batch training (lr 0.8) on XOR from that state: all four outputs within 0.1 of target at epoch {e_good}; final loss {L[-1]:.4f}. Gradient shown on edges is dLoss/dWeight for the single example.\nBackprop popularized by Rumelhart, Hinton & Williams (Nature, 1986).")
    return reel, buf, cfg, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = {"temp": build_temp, "bpe": build_bpe, "backprop": build_backprop}[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
