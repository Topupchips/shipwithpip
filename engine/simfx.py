"""Real-simulation retention reels: (A) evolution of 120 'AIs' through a maze, (B) a neural net learning a spiral.

usage: python simfx.py evo|nn out.mp4 --model <kokoro dir> [--stills out.png]
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sim"))
import evo  # noqa: E402
import nn  # noqa: E402
import voice  # noqa: E402
from make_reel import music_bed  # noqa: E402

W, H, FPS, SR = 1080, 1920, 30, 24000
A = os.path.join(HERE, "assets")
F_BOLD = os.path.join(A, "InterDisplay-Bold.otf")
F_SEMI = os.path.join(A, "InterDisplay-SemiBold.otf")
F_MONO = os.path.join(A, "JetBrainsMono-700.ttf")
INK = (10, 6, 26)
PINK, CYAN, GREEN, YEL = (255, 76, 170), (60, 222, 255), (70, 255, 170), (255, 226, 96)
VX, VY, VS = 60, 430, 960
_f = {}


def font(p, s):
    k = (p, int(s))
    if k not in _f:
        _f[k] = ImageFont.truetype(p, int(s))
    return _f[k]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    return 1 - (1 - clamp(x)) ** 3


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def make_bg():
    y = np.linspace(0, 1, H)[:, None, None]
    arr = np.array([30, 14, 72])[None, None, :] * (1 - y) + np.array([8, 5, 26])[None, None, :] * y
    arr = np.broadcast_to(arr, (H, W, 3)).astype(np.uint8).copy()
    im = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(im, "RGBA")
    for yy in range(0, H, 64):
        for xx in range((yy // 64) % 2 * 32, W, 64):
            d.ellipse([xx - 2, yy - 2, xx + 2, yy + 2], fill=(255, 255, 255, 22))
    return im


def tw(d, text, f):
    return d.textlength(text, font=f)


def wrap(d, text, f, maxw):
    ls, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if tw(d, t, f) <= maxw:
            cur = t
        else:
            ls.append(cur)
            cur = w
    if cur:
        ls.append(cur)
    return ls


class Reel:
    """Shared compositor: headline, viewport card, pills, progress bar, karaoke caption, CTA."""

    def __init__(self, cfg, vo):
        self.cfg, self.vo = cfg, vo
        self.bg = make_bg()
        self.T = cfg["T"]

    def header(self, d, t):
        lines, tag = self.cfg["header"](t)
        pulse = 0.6 + 0.4 * math.sin(t * 6)
        f = font(F_SEMI, 34)
        w = tw(d, tag, f) + 90
        d.rounded_rectangle([(W - w) / 2, 70, (W + w) / 2, 136], radius=33, fill=YEL + (255,))
        d.ellipse([(W - w) / 2 + 22, 94, (W - w) / 2 + 44, 116], fill=(255, 60, 90, int(255 * pulse)))
        d.text((W / 2 + 14, 103), tag, font=f, fill=INK, anchor="mm")
        for i, (txt, col) in enumerate(lines):
            d.text((W / 2, 235 + i * 120), txt, font=font(F_BOLD, 108), fill=col, anchor="mm", stroke_width=9, stroke_fill=INK)

    def card(self, d):
        d.rounded_rectangle([VX - 6, VY - 6, VX + VS + 6, VY + VS + 6], radius=46, fill=self.cfg["accent"] + (255,))
        d.rounded_rectangle([VX, VY, VX + VS, VY + VS], radius=40, fill=(12, 8, 34, 255))

    def pills(self, d, items):
        x = VX
        f = font(F_MONO, 40)
        for txt, col in items:
            w = tw(d, txt, f) + 56
            d.rounded_rectangle([x, 1412, x + w, 1488], radius=38, fill=(255, 255, 255, 22), outline=col + (255,), width=4)
            d.text((x + w / 2, 1450), txt, font=f, fill=(255, 255, 255, 255), anchor="mm")
            x += w + 18

    def bar(self, d, frac, label, col):
        d.rounded_rectangle([VX, 1520, VX + VS, 1560], radius=20, fill=(255, 255, 255, 30))
        if frac > 0.005:
            d.rounded_rectangle([VX, 1520, VX + max(40, VS * frac), 1560], radius=20, fill=col + (255,))
        d.text((VX + 20, 1540), label, font=font(F_MONO, 26), fill=INK if frac > 0.12 else (255, 255, 255, 255), anchor="lm")

    def caption(self, d, t):
        for st, en, text in self.vo:
            if st <= t <= en + 0.15:
                words = text.split()
                wts = np.array([len(w) + 2 for w in words], float)
                cum = np.cumsum(wts) / wts.sum() * (en - st)
                k = min(int(np.searchsorted(cum, t - st)), len(words) - 1)
                g0 = (k // 4) * 4
                chunk = words[g0:g0 + 4]
                size = 68
                while True:
                    f = font(F_BOLD, size)
                    ws = [tw(d, w, f) + 26 for w in chunk]
                    if sum(ws) <= W - 90 or size <= 44:
                        break
                    size -= 4
                x = (W - sum(ws)) / 2
                for i, w in enumerate(chunk):
                    col = YEL if g0 + i == k else (255, 255, 255)
                    d.text((x, 1650), w, font=f, fill=col + (255,), stroke_width=9, stroke_fill=INK + (255,), anchor="lm")
                    x += ws[i]
                return

    def cta(self, d, t):
        if t > self.T - 4.2:
            q = ease_out((t - (self.T - 4.2)) / 0.4)
            txt = "@shipwithpip"
            f = font(F_BOLD, 52)
            w = tw(d, txt, f) + 90
            d.rounded_rectangle([(W - w) / 2, 1740 - (1 - q) * 30, (W + w) / 2, 1830 - (1 - q) * 30], radius=45, fill=PINK + (int(255 * q),))
            d.text((W / 2, 1785 - (1 - q) * 30), txt, font=f, fill=(255, 255, 255, int(255 * q)), anchor="mm")

    def progress(self, d, t):
        d.rectangle([0, 0, W * clamp(t / self.T), 12], fill=YEL + (255,))


# ----------------------------------------------------------------- scene A: evolution
class EvoReel(Reel):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        runs = evo.evolve(2, maxg=40, target=0.6)
        self.runs = runs
        self.sel = cfg["gens"]
        self.sched = []
        t = 1.0
        for gi, dur in zip(self.sel, cfg["durs"]):
            self.sched.append((t, t + dur, gi))
            t += dur + 0.45
        self.end_play = t - 0.45
        for r in runs:
            st = r["st"]
            first = np.where((st == 2).any(0), (st == 2).argmax(0), -1)
            r["first"] = first
        self.final = runs[self.sel[-1]]["reached"]

    def cur(self, t):
        for i, (a, b, gi) in enumerate(self.sched):
            if a <= t <= b:
                return i, gi, (t - a) / (b - a)
            if i + 1 < len(self.sched) and b < t < self.sched[i + 1][0]:
                return i, gi, 1.0
        if t < self.sched[0][0]:
            return -1, self.sel[0], 0.0
        return len(self.sched) - 1, self.sel[-1], 1.0

    def px(self, x, y):
        return VX + x * VS / 100.0, VY + y * VS / 100.0

    def frame(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img, "RGBA")
        self.header(d, t)
        self.card(d)
        for g in range(1, 10):
            d.line([(VX + g * 96, VY + 8), (VX + g * 96, VY + VS - 8)], fill=(255, 255, 255, 10), width=2)
            d.line([(VX + 8, VY + g * 96), (VX + VS - 8, VY + g * 96)], fill=(255, 255, 255, 10), width=2)
        for x0, y0, x1, y1 in evo.WALLS:
            a, b = self.px(x0, y0)
            c, e = self.px(x1, y1)
            d.rounded_rectangle([a, b - 4, c, e + 4], radius=14, fill=(92, 70, 170, 255), outline=(190, 160, 255, 255), width=4)
        gx, gy = self.px(*evo.GOAL)
        pulse = 0.5 + 0.5 * math.sin(t * 5)
        for k, (r, al) in enumerate(((78, 40), (56, 70), (34, 120))):
            rr = r + pulse * 6
            d.ellipse([gx - rr, gy - rr, gx + rr, gy + rr], fill=GREEN + (al,))
        d.ellipse([gx - 16, gy - 16, gx + 16, gy + 16], fill=GREEN + (255,))
        d.text((gx, gy - 96), "GOAL", font=font(F_MONO, 30), fill=GREEN + (255,), anchor="mm")
        sx, sy = self.px(*evo.START)
        d.ellipse([sx - 22, sy - 22, sx + 22, sy + 22], outline=(255, 255, 255, 120), width=3)
        i, gi, prog = self.cur(t)
        run = self.runs[gi]
        hist, st = run["hist"], run["st"]
        s_f = prog * evo.STEPS
        s = int(min(s_f, evo.STEPS - 1))
        fr = s_f - s
        pos = hist[s] * (1 - fr) + hist[s + 1] * fr
        # ghost of previous best path
        if i >= 1:
            pr = self.runs[self.sel[i - 1]]
            pts = [self.px(*pr["hist"][k, pr["bi"]]) for k in range(0, evo.STEPS + 1, 4)]
            d.line(pts, fill=(255, 255, 255, 38), width=3)
        n_reached = 0
        for a in range(evo.N):
            state = st[s + 1] if s + 1 <= evo.STEPS else st[-1]
            x, y = self.px(*pos[a])
            if t < self.sched[0][0]:
                x += 26 * math.cos(a * 2.4 + t * 2.0) * (0.4 + (a % 5) / 5.0)
                y += 20 * math.sin(a * 1.7 + t * 2.6) * (0.4 + (a % 4) / 4.0) - 10
            sa = int(state[a])
            if sa == 2:
                n_reached += 1
                ring = (s - run["first"][a]) / 18.0
                if 0 <= ring < 1:
                    rr = 10 + 40 * ring
                    d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=GREEN + (int(220 * (1 - ring)),), width=3)
                d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=GREEN + (255,))
            elif sa == 1 or s == 0:
                for k in range(1, 7):
                    p0 = hist[max(0, s - k)][a]
                    px_, py_ = self.px(*p0)
                    d.ellipse([px_ - 5 + k * .4, py_ - 5 + k * .4, px_ + 5 - k * .4, py_ + 5 - k * .4], fill=PINK + (int(110 - k * 15),))
                d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=PINK + (255,), outline=(255, 255, 255, 255), width=2)
            else:
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(120, 104, 170, 150))
        if t < self.sched[0][0]:
            n_reached = 0
        gen_no = gi + 1 if t >= self.sched[0][0] else 1
        # summary flash in the gap
        if i >= 0 and prog >= 1.0 and i + 1 < len(self.sched) and t < self.sched[i + 1][0]:
            txt = f"GEN {gi + 1}: {run['reached']} reached"
            f = font(F_BOLD, 70)
            w = tw(d, txt, f) + 70
            d.rounded_rectangle([VX + (VS - w) / 2, VY + 380, VX + (VS + w) / 2, VY + 480], radius=50, fill=(10, 6, 26, 225), outline=YEL + (255,), width=4)
            d.text((VX + VS / 2, VY + 430), txt, font=f, fill=(255, 255, 255, 255), anchor="mm")
        self.pills(d, [(f"GEN {gen_no}", YEL), (f"REACHED {n_reached}/{evo.N}", GREEN)])
        self.bar(d, n_reached / evo.N, "SUCCESS RATE", GREEN)
        self.caption(d, t)
        self.cta(d, t)
        self.progress(d, t)
        return img.convert("RGB")


# ----------------------------------------------------------------- scene B: neural net
class NNReel(Reel):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.X, self.y, snaps = nn.train(cfg["epochs"], lr=0.005)
        self.snaps = snaps
        self.cmap_cache = {}

    def epoch_at(self, t):
        tau = t - 0.8
        pts = [(0, 0), (4, 40), (14, 175), (20, 262), (23, self.cfg["epochs"])]
        if tau <= 0:
            return 0.0
        for (t0, e0), (t1, e1) in zip(pts, pts[1:]):
            if tau <= t1:
                return e0 + (e1 - e0) * (tau - t0) / (t1 - t0)
        return float(self.cfg["epochs"])

    def heat(self, e):
        i = int(min(e, len(self.snaps) - 2))
        fr = e - i
        p = self.snaps[i][3] * (1 - fr) + self.snaps[i + 1][3] * fr
        a = np.clip(np.abs(p - 0.5) * 2, 0, 1) ** 0.8 * 0.9
        base = np.array([16, 10, 44], float)
        mag, cy = np.array(PINK, float), np.array(CYAN, float)
        col = np.where((p > 0.5)[..., None], mag, cy)
        arr = base * (1 - a[..., None]) + col * a[..., None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").resize((VS, VS), Image.BICUBIC)

    def frame(self, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img, "RGBA")
        self.header(d, t)
        self.card(d)
        e = self.epoch_at(t)
        hm = self.heat(e)
        mask = Image.new("L", (VS, VS), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, VS, VS], radius=40, fill=255)
        img.paste(hm, (VX, VY), mask)
        d = ImageDraw.Draw(img, "RGBA")
        sc = VS / 2.3
        for (px_, py_), c in zip(self.X, self.y):
            x = VX + (px_ + 1.15) * sc
            yv = VY + (1.15 - py_) * sc
            col = PINK if c > 0.5 else CYAN
            d.ellipse([x - 9, yv - 9, x + 9, yv + 9], fill=col + (255,), outline=(255, 255, 255, 255), width=3)
        i = int(min(e, len(self.snaps) - 1))
        _, loss, acc, _ = self.snaps[i]
        d.text((VX + 30, VY + 40), "ACCURACY", font=font(F_MONO, 30), fill=(255, 255, 255, 230), anchor="lm", stroke_width=5, stroke_fill=INK)
        d.text((VX + 30, VY + 100), f"{acc * 100:0.0f}%", font=font(F_BOLD, 104), fill=YEL + (255,), anchor="lm", stroke_width=8, stroke_fill=INK)
        d.text((VX + VS - 30, VY + 40), f"EPOCH {i}", font=font(F_MONO, 40), fill=(255, 255, 255, 255), anchor="rm", stroke_width=5, stroke_fill=INK)
        d.text((VX + VS - 30, VY + 92), f"loss {loss:0.2f}", font=font(F_MONO, 34), fill=(255, 255, 255, 200), anchor="rm", stroke_width=5, stroke_fill=INK)
        self.pills(d, [(f"EPOCH {i}", YEL), (f"LOSS {loss:0.2f}", PINK)])
        self.bar(d, acc, "HOW MANY DOTS IT GETS RIGHT", YEL)
        self.caption(d, t)
        self.cta(d, t)
        self.progress(d, t)
        return img.convert("RGB")


# ----------------------------------------------------------------- audio
def tone(buf, t0, dur, f0, f1, amp, kind="sine"):
    i0, n = int(t0 * SR), int(dur * SR)
    if i0 >= len(buf):
        return
    n = min(n, len(buf) - i0)
    tt = np.arange(n) / SR
    f = f0 + (f1 - f0) * (tt / max(dur, 1e-6))
    ph = 2 * np.pi * np.cumsum(f) / SR
    w = np.sin(ph) + (0.35 * np.sin(2 * ph) if kind == "rich" else 0)
    env = np.minimum(1, tt / 0.01) * np.exp(-tt / (dur * 0.45))
    buf[i0:i0 + n] += (w * env * amp).astype(np.float32)


def make_audio(cfg, vo_lines, model_dir):
    T = cfg["T"]
    n = int(T * SR)
    buf = np.zeros(n, np.float32)
    bed = music_bed(T, cfg.get("seed", 5))[:n]
    buf[:len(bed)] += bed * 0.5
    vo = []
    for st, text in vo_lines:
        x, sr = voice.say(cfg["model"], text, cfg.get("voice", "af_bella"), 1.0)
        x = x / (np.max(np.abs(x)) or 1) * 0.95
        i = int(st * SR)
        buf[i:i + len(x)] += x[: n - i]
        vo.append((st, st + len(x) / SR, text))
    for st, en, text in vo:
        pass
    return buf, vo


def finish_audio(buf, cfg, sonify):
    T = cfg["T"]
    n = len(buf)
    sonify(buf)
    buf = buf / (np.max(np.abs(buf)) or 1) * 0.92
    return buf


def render(reel, out, audio, T, stills=None):
    if stills:
        ts = reel.cfg["still_times"]
        fr = [reel.frame(t).resize((300, 533)) for t in ts]
        sheet = Image.new("RGB", (300 * len(fr), 533))
        for i, f_ in enumerate(fr):
            sheet.paste(f_, (i * 300, 0))
        sheet.save(stills)
        return
    wav = os.path.splitext(out)[0] + "_a.wav"
    sf.write(wav, audio, SR)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-15:TP=-1.2:LRA=11",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for i in range(int(T * FPS)):
        ff.stdin.write(reel.frame(i / FPS).tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    if a.which == "evo":
        cfg = dict(T=31.0, model=a.model, seed=5, gens=[0, 1, 3, 5, 7, 9, 19], durs=[4.2, 2.6, 2.2, 2.0, 1.6, 1.4, 4.0], accent=PINK,
                   still_times=[0.3, 3, 7, 10, 14, 19, 22, 27],
                   header=lambda t: ((([("120 DUMB AIs.", (255, 255, 255)), ("1 GOAL.", YEL)], "WATCH TILL THE END") if t < 21.7 else
                                      ([("73 OF 120", GREEN), ("MADE IT.", (255, 255, 255))], "NATURAL SELECTION"))))
        lines = [(0.2, "A hundred and twenty dumb AIs. One goal. Watch."), (3.7, "Generation one. They know nothing, so they crash."),
                 (7.3, "Every round, the best ones survive, and their moves get copied with small mutations."),
                 (12.9, "Generation by generation, they get smarter."), (16.4, "Now watch generation twenty."),
                 (21.9, "Nobody told them the route. They found it by trial, selection, and repeat."), (27.3, "Follow for more AI, explained in motion.")]
        buf, vo = make_audio(cfg, lines, a.model)
        reel = EvoReel(cfg, vo)

        def sonify(b):
            for k, (t0, t1, gi) in enumerate(reel.sched):
                tone(b, t0, 0.35, 330 + 60 * k, 440 + 90 * k, 0.16, "rich")
            tone(b, 17.0, 0.9, 200, 900, 0.12)
            for k, f in enumerate((523, 659, 784, 1047)):
                tone(b, 21.75 + k * 0.07, 1.3, f, f, 0.14, "rich")
        audio = finish_audio(buf, cfg, sonify)
        render(reel, a.out, audio, cfg["T"], a.stills)
        print(a.out, "reached", reel.final)
    else:
        cfg = dict(T=31.0, model=a.model, seed=9, epochs=320, accent=CYAN, still_times=[0.3, 3, 8, 13, 17, 19, 21, 26],
                   header=lambda t: ((([("THIS AI KNOWS", (255, 255, 255)), ("NOTHING.", YEL)], "WATCH IT LEARN") if t < 19.6 else
                                      ([("IT LEARNED", (255, 255, 255)), ("A SPIRAL.", GREEN)], "NOBODY SHOWED IT")))
                   )
        lines = [(0.2, "This AI knows absolutely nothing. Watch it learn."), (3.6, "It sees two colors of dots. Its job is to draw the border between them."),
                 (8.4, "Each round, it guesses, measures its mistake, and nudges its weights."), (13.4, "Guess. Measure. Nudge. Again, and again."),
                 (17.2, "Almost there. Wait for it."), (19.8, "There. It found a spiral nobody showed it."),
                 (23.8, "That's what training is. Guess, measure, nudge, repeat."), (27.6, "Follow for more AI, explained in motion.")]
        buf, vo = make_audio(cfg, lines, a.model)
        reel = NNReel(cfg, vo)

        def sonify(b):
            T = cfg["T"]
            tt = np.arange(len(b)) / SR
            ep = np.array([reel.epoch_at(x) for x in np.arange(0, T, 0.05)])
            acc = np.array([reel.snaps[int(min(e, len(reel.snaps) - 1))][2] for e in ep])
            f = np.interp(tt, np.arange(0, T, 0.05), 160 + 700 * np.clip((acc - 0.4) / 0.6, 0, 1) ** 1.3)
            ph = 2 * np.pi * np.cumsum(f) / SR
            lvl = np.interp(tt, [0, 0.8, 2, 19.6, 21, 23, 24], [0, 0, 0.05, 0.08, 0.02, 0.0, 0.0])
            b += (np.sin(ph) * lvl).astype(np.float32)
            for k, fq in enumerate((523, 659, 784, 1047)):
                tone(b, 19.6 + k * 0.07, 1.4, fq, fq, 0.15, "rich")
        audio = finish_audio(buf, cfg, sonify)
        render(reel, a.out, audio, cfg["T"], a.stills)
        print(a.out, "final acc", reel.snaps[-1][2])


if __name__ == "__main__":
    main()
