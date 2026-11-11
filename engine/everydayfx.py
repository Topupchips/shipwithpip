"""Everyday-science reels: phantom traffic jam, noise cancelling headphones.
usage: python everydayfx.py jam|anc out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trafficsim as TS  # noqa: E402
from learnfx import write_post  # noqa: E402
from scenes import CYAN, GREEN, PINK, VIOLET, YEL, ORANGE, Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, INK, VS, clamp, ease_out, font, tone  # noqa: E402

BG = (10, 6, 28)


def speed_col(f):
    f = clamp(f)
    if f < 0.5:
        a, b, q = (255, 60, 80), (255, 210, 60), f / 0.5
    else:
        a, b, q = (255, 210, 60), (60, 235, 140), (f - 0.5) / 0.5
    return tuple(int(a[i] + (b[i] - a[i]) * q) for i in range(3))


# ------------------------------------------------------------------ traffic jam
class JamReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.X, self.V, self.dt, self.L, self.k = cfg["X"], cfg["V"], cfg["dt"], cfg["L"], cfg["k"]
        self.vref = cfg["vref"]
        yy, xx = np.mgrid[0:VS, 0:VS]
        base = np.zeros((VS, VS, 3), float) + np.array(BG)
        base += (np.clip(1 - np.hypot(xx - 480, yy - 480) / 700, 0, 1) ** 2 * 45)[..., None] * np.array([0.3, 0.5, 1.0])
        self.base = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(self.base, "RGBA")
        d.ellipse([80, 80, 880, 880], outline=(90, 90, 140, 255), width=46)
        d.ellipse([103, 103, 857, 857], outline=(40, 38, 70, 255), width=2)
        d.ellipse([57, 57, 903, 903], outline=(40, 38, 70, 255), width=2)
        # spacetime image
        step = int(0.25 / self.dt)
        nb = 130
        edges = np.linspace(0, self.L, nb, endpoint=False) + self.L / nb / 2
        rows = []
        for kk in range(0, len(self.X), step):
            order = np.argsort(self.X[kk])
            xs, vs = self.X[kk][order], self.V[kk][order]
            idx = np.searchsorted(xs, edges) - 1
            rows.append(vs[idx % len(xs)])
        self.ST = np.array(rows)  # (time, pos)
        self.stepst = step
        self.lut = np.array([speed_col(i / 255) for i in range(256)], np.uint8)

    def sim_t(self, t):
        return max(0.0, (t - 0.4) * self.k)

    def content(self, t):
        st = self.sim_t(t)
        kk = int(min(len(self.X) - 1, st / self.dt))
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        R = 400
        # spacetime plot (last 40 sim seconds)
        n_cols = 160
        j = int(st / 0.25)
        a = max(0, j - n_cols + 1)
        sl = self.ST[a:j + 1]
        img = np.zeros((self.ST.shape[1], n_cols, 3), np.uint8)
        img[:] = (24, 20, 46)
        if len(sl):
            cols = self.lut[np.clip((sl / self.vref * 255).astype(int), 0, 255)]  # (t,pos,3)
            img[:, n_cols - len(sl):, :] = np.transpose(cols, (1, 0, 2))
        pim = Image.fromarray(img, "RGB").resize((430, 430), Image.NEAREST)
        px, py = 265, 265
        d.rounded_rectangle([px - 8, py - 8, px + 438, py + 438], radius=18, fill=(14, 10, 34, 255), outline=(150, 140, 220, 200), width=3)
        im.paste(pim, (px, py))
        d = ImageDraw.Draw(im, "RGBA")
        d.text((480, 232), "ROAD OVER TIME", font=font(F_MONO, 22), fill=(200, 195, 240, 255), anchor="mm")
        d.text((480, 725), "RED BANDS = STOPPED. WATCH THEM DRIFT.", font=font(F_MONO, 19), fill=(200, 195, 240, 255), anchor="mm")
        # cars
        N = self.X.shape[1]
        for i in range(N):
            th = 2 * math.pi * self.X[kk][i] / self.L - math.pi / 2
            cx, cy = 480 + R * math.cos(th), 480 + R * math.sin(th)
            tx, ty = -math.sin(th), math.cos(th)   # tangent (clockwise)
            nx, ny = math.cos(th), math.sin(th)
            hl, hw = 24, 12
            pts = [(cx + tx * s1 * hl + nx * s2 * hw, cy + ty * s1 * hl + ny * s2 * hw) for s1, s2 in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            col = speed_col(self.V[kk][i] / self.vref)
            d.polygon(pts, fill=col + (255,), outline=INK + (255,))
            if self.V[kk][i] < 0.5:
                bx, by = cx - tx * hl, cy - ty * hl
                d.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(255, 40, 60, 255))
        d.text((930, 60), "SIM \u00b7 3X SPEED", font=font(F_MONO, 20), fill=(200, 195, 240, 255), anchor="rm")
        # brake tap label
        bt = self.cfg["brake_t"]
        if bt <= st < bt + 2.0:
            d.text((480, 150), "BRAKE TAP: 2 SECONDS", font=font(F_BOLD, 40), fill=PINK + (255,), anchor="mm", stroke_width=5, stroke_fill=INK)
        return im

    def stats(self, t):
        st = self.sim_t(t)
        kk = int(min(len(self.X) - 1, st / self.dt))
        v = self.V[kk]
        stopped = int((v < 0.5).sum())
        slow = v.min() * 3.6
        frac = float((v > 0.5 * self.vref).mean())
        return [(f"SLOWEST {slow:0.0f} KM/H", YEL if slow > 5 else PINK), (f"{stopped} STOPPED", PINK if stopped else GREEN)], frac, "CARS MOVING FREELY", GREEN


def build_jam(model):
    k = 3.0
    brake_sim = 21.0
    tmax = 125.0
    X, V, dt = TS.idm_ring(brake_t=brake_sim, tmax=tmax)
    X0, V0, _ = TS.idm_ring(brake_t=999.0, tmax=60.0)
    calm = float(V0.min() * 3.6)
    calm_max = float(V0.max() * 3.6)
    L, N = 260.0, X.shape[1]
    vref = float(V[0].mean())
    # jam speed
    ts, ang = [], []
    for kk in range(int(35 / dt), int(105 / dt), int(0.25 / dt)):
        m = V[kk] < 1.0
        if m.sum() >= 2:
            ang.append(np.angle(np.exp(1j * 2 * np.pi * X[kk][m] / L).mean()))
            ts.append(kk * dt)
    jam_kmh = float(np.polyfit(ts, np.unwrap(ang) * L / (2 * np.pi), 1)[0] * 3.6)
    stop_first = next(i * dt for i in range(len(V)) if (V[i] < 0.5).sum() >= 3)
    n_at = lambda s: int((V[min(len(V) - 1, int(s / dt))] < 0.5).sum())  # noqa: E731
    r = lambda s: s / k + 0.4  # noqa: E731
    print("calm min/max kmh", calm, calm_max, "vref", vref * 3.6, "jam km/h", jam_kmh, "first >=3 stopped at sim", stop_first)
    tj = r(stop_first)
    gap = stop_first - brake_sim
    cfg = dict(T=60.0, c1=PINK, c2=CYAN, X=X, V=V, dt=dt, L=L, k=k, vref=vref, brake_t=brake_sim, payoff=round(tj, 1), seed=131,
               still_times=[0.8, r(brake_sim) + 0.8, r(brake_sim) + 4, tj - 1, tj + 1.5, tj + 8, tj + 14],
               header=lambda t: (([("22 CARS, NO CRASH", (255, 255, 255)), ("NO ROADWORKS.", YEL)], "ONE TINY MISTAKE") if t < tj else
                                 ([("A JAM FROM", (255, 255, 255)), ("NOTHING.", PINK)], "AND IT MOVES BACKWARD")))
    lines = [(0.2, "Twenty two cars on a ring road. No crash. No roadworks. Nothing in the way."),
             (r(brake_sim) - 0.6, "Now one driver taps the brake. Just two seconds."),
             (r(brake_sim) + 2.7, "The car behind brakes a little harder. The next one, harder again."),
             (tj - 1.2, f"{gap:0.0f} seconds later, cars are completely stopped. For no reason."),
             (tj + 5.0, f"And watch the jam. It drifts backward, against the traffic, at about {abs(jam_kmh):0.0f} kilometers an hour."),
             (tj + 11.0, "Japanese researchers saw the same thing in 2008 with real cars on a circular track."),
             (tj + 16.8, "Stuck in a jam with no cause? That is why. Comment JAM for the prompt.")]
    buf, vo = make_audio(cfg, lines, model)
    chime(buf, cfg["payoff"])
    cfg["T"] = vo[-1][1] + 2.4
    reel = JamReel(cfg, vo)
    post = dict(title="EVERY-1  one brake tap makes a traffic jam from nothing",
                caption=f"22 cars on a ring road, everyone driving smoothly. One driver taps the brake for 2 seconds, and a stop-and-go jam appears out of nothing, then drifts backward against the traffic. This is a car-following simulation (Intelligent Driver Model). Without the brake tap, the same cars stay smooth ({calm:.1f} to {calm_max:.1f} km/h). In 2008, a Japanese team saw the same effect with 22 real cars on a 230 m circular track (Sugiyama et al., New Journal of Physics).\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment JAM and I'll send the prompt so you can build it.",
                tags=["physics", "traffic", "science", "simulation", "coding", "shipwithpip"],
                pinned="Build me a single-file HTML canvas page that simulates 22 cars on a circular road using the Intelligent Driver Model (v0 30 km/h scaled, T 1.0 s, s0 2 m, a 1 m/s2, b 2 m/s2, car length 4.5 m, road 260 m). Start everyone at equilibrium speed, add a button that makes one car brake hard for 2 seconds, color cars by speed, and draw a space-time plot so the jam wave is visible.",
                notes=f"Simulation: Intelligent Driver Model (Treiber, Hennecke, Helbing 2000), N={N}, road length {L:.0f} m, T=1.0 s, s0=2 m, a=1, b=2, car length 4.5 m, v0=30 km/h free speed, 0.05 s steps. Start: equilibrium spacing/speed ({vref * 3.6:.1f} km/h). Control run with NO brake tap: speed stays {calm:.1f} to {calm_max:.1f} km/h for 60 s. Brake tap: car 1 brakes at -4 m/s2 for 2 s. First time 3+ cars fully stopped: {gap:.0f} sim seconds after the tap. Jam drift measured from the sim: {jam_kmh:.1f} km/h (negative = backward). Video plays the sim at 3x speed (labelled 3X SPEED on screen).\nReal study: Sugiyama et al. (2008), 'Traffic jams without bottlenecks', New Journal of Physics 10, 033001: 22 cars on a 230 m circular track, drivers asked to hold ~30 km/h; a stop-and-go wave formed spontaneously (reported by ScienceDaily/IOP). The sim is a model, not footage, and its numbers are not the study's.")
    return reel, buf, cfg, post


# ------------------------------------------------------------------ noise cancelling
TAU = 1e-4  # assumed 0.1 ms total delay (illustrative)


def resid(f):
    return 2 * abs(math.sin(math.pi * f * TAU))


def freq_at(t, t0, t1, f0=100.0, f1=5000.0):
    q = clamp((t - t0) / (t1 - t0))
    return f0 * (f1 / f0) ** q


class AncReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        yy, xx = np.mgrid[0:VS, 0:VS]
        base = np.zeros((VS, VS, 3), float) + np.array(BG)
        base += (np.clip(1 - np.hypot(xx - 480, yy - 480) / 700, 0, 1) ** 2 * 40)[..., None] * np.array([0.4, 0.3, 1.0])
        self.base = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
        self.ph = 0.0
        self.lastt = -1.0

    def f(self, t):
        return freq_at(t, *self.cfg["sweep"])

    def content(self, t):
        c = self.cfg
        f = self.f(t)
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        show_anti = clamp((t - c["anti_t"]) / 0.8)
        show_sum = clamp((t - c["sum_t"]) / 0.8)
        # scroll phase (visual): independent of f so the picture stays readable
        ph = t * 2 * math.pi * 0.9
        delta = 2 * math.pi * f * TAU      # real phase lag of the anti-noise
        cyc = 2.5
        panels = [("NOISE", PINK, 120), ("ANTI-NOISE (LATE BY 0.1 MS)", CYAN, 410), ("WHAT YOU HEAR", GREEN, 700)]
        for name, col, cy in panels:
            d.rounded_rectangle([40, cy - 120, 920, cy + 120], radius=22, fill=(14, 10, 34, 255), outline=col + (150,), width=3)
            d.text((60, cy - 100), name, font=font(F_MONO, 24), fill=col + (255,), anchor="lm")
            d.line([(60, cy), (900, cy)], fill=(255, 255, 255, 40), width=2)
        A = 80
        xs = np.linspace(60, 900, 420)
        th = (xs - 60) / 840 * cyc * 2 * math.pi - ph
        noise = np.sin(th)
        anti = -np.sin(th - delta)
        tot = noise + anti
        def poly(y, cy, col, w, a=255):
            pts = [(float(x), float(cy - v * A)) for x, v in zip(xs, y)]
            d.line(pts, fill=col + (int(a * 0.25),), width=w + 10, joint="curve")
            d.line(pts, fill=col + (a,), width=w, joint="curve")
        poly(noise, 120, PINK, 6)
        if show_anti > 0:
            poly(anti, 410, CYAN, 6, int(255 * show_anti))
        if show_sum > 0:
            lv = (tot) * (A * 1.0)
            # drawn to true scale (no exaggeration)
            pts = [(float(x), float(700 - v / A * A * 1.0)) for x, v in zip(xs, tot)]
            d.line(pts, fill=GREEN + (int(60 * show_sum),), width=16, joint="curve")
            d.line(pts, fill=GREEN + (int(255 * show_sum),), width=6, joint="curve")
        # frequency readout
        if t > c["sweep"][0] - 1.0:
            d.text((900, 265), f"{f:,.0f} HZ", font=font(F_BOLD, 40), fill=YEL + (255,), anchor="rm", stroke_width=5, stroke_fill=INK)
        else:
            d.text((900, 265), "100 HZ (LOW HUM)", font=font(F_BOLD, 34), fill=YEL + (255,), anchor="rm", stroke_width=5, stroke_fill=INK)
        return im

    def stats(self, t):
        f = self.f(t)
        rv = resid(f) if t > self.cfg["sum_t"] else 1.0
        pct = rv * 100
        col = GREEN if rv < 0.25 else (YEL if rv < 1.0 else PINK)
        tag = "NOISE LEFT" if rv <= 1.0 else "NOISE LEFT (WORSE!)"
        return [(f"{pct:0.0f}% LEFT", col), (f"{f:,.0f} HZ", YEL)], min(1.0, rv), "NOISE REACHING YOU", col


def build_anc(model):
    c0, c1 = 18.0, 31.0
    cfg = dict(T=60.0, c1=CYAN, c2=PINK, anti_t=5.0, sum_t=8.0, sweep=(c0, c1), payoff=None, seed=141,
               still_times=[1.5, 6.0, 10.5, 20, 24, 28, 33])
    p100, p1k, p2k, p5k = resid(100) * 100, resid(1000) * 100, resid(2000) * 100, resid(5000) * 100
    f_at = lambda t: freq_at(t, c0, c1)  # noqa: E731
    f23 = f_at(23.0)
    p23 = resid(f23) * 100
    print("residual % at 100/1000/2000/5000 Hz", p100, p1k, p2k, p5k, "at 23s", f23, p23, "db100", 20 * math.log10(resid(100)))
    cfg["header"] = lambda t: (([("HEADPHONES", (255, 255, 255)), ("ADD SOUND", YEL)], "TO MAKE SILENCE") if t < 17.5 else
                               ([("HIGH SOUNDS", (255, 255, 255)), ("SLIP THROUGH", PINK)], "TIMING IS EVERYTHING"))
    lines = [(0.2, "Noise cancelling headphones do not block sound. They add more sound."),
             (4.2, "A mic hears the noise, and the headphones play its exact opposite. Peak meets valley."),
             (9.6, f"They cancel. With a tenth of a millisecond of delay, a low hum is down to {p100:0.0f} percent."),
             (15.6, "But raise the pitch, and that same tiny delay matters more."),
             (20.4, f"At {round(f23, -1):,.0f} hertz, {resid(round(f23, -1)) * 100:0.0f} percent gets through."),
             (25.6, "Past that, in this simple model, the anti sound lands so late it makes noise worse."),
             (31.4, "That is why steady hums vanish but sharp sounds slip through. Comment SILENCE for the prompt.")]
    buf, vo = make_audio(cfg, lines, model)
    cfg["T"] = vo[-1][1] + 2.4
    reel = AncReel(cfg, vo)
    post = dict(title="EVERY-2  noise cancelling headphones add sound to make silence",
                caption=f"Noise cancelling headphones do not block sound, they add more. A mic hears the noise and the headphones play its opposite, so peak meets valley and the two cancel. But timing matters. In this simplified model with a fixed 0.1 ms delay, a 100 Hz hum is down to about {p100:.0f}% of its size, while at 5,000 Hz the anti-sound lands so late it makes things worse. Real headphones use faster tricks plus passive isolation, but steady low hums really are the easiest to cancel and sharp high sounds the hardest.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment SILENCE and I'll send the prompt so you can build it.",
                tags=["physics", "science", "sound", "headphones", "coding", "shipwithpip"],
                pinned="Build me a single-file HTML page that draws a noise sine wave, an anti-noise wave that is the inverse delayed by 0.1 ms, and their sum, with a frequency slider from 100 Hz to 5 kHz. Show the remaining noise as 2*abs(sin(pi*f*0.0001)) in percent and color it green when under 25% and red when over 100%.",
                notes=f"Math: noise sin(2*pi*f*t); anti-noise = -sin(2*pi*f*(t - tau)) with tau = 0.1 ms ASSUMED (illustrative, not a measured headphone spec). Sum amplitude = 2*|sin(pi*f*tau)|. Results: 100 Hz {p100:.1f}% ({20 * math.log10(resid(100)):.1f} dB), 1000 Hz {p1k:.0f}%, 2000 Hz {p2k:.0f}%, 5000 Hz {p5k:.0f}% (over 100% means worse than no cancelling, in this simplified model). Waves drawn to true scale; scroll speed is visual only, the phase lag is the real value.\nSources for the qualitative claims: ANC uses mics to capture noise and plays inverted-phase anti-noise; steady low frequencies cancel best and sharp high frequencies are hardest (https://soundly.com/blog/how-noise-canceling-headphones-work). Real products use feedforward + feedback mics, adaptive filters and passive isolation, so this model exaggerates the high-frequency failure.")
    return reel, buf, cfg, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = {"jam": build_jam, "anc": build_anc}[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
