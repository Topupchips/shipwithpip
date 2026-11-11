"""Drop test reel: penny, basketball, bowling ball and a no-air ghost from the Burj Khalifa (828 m), simulated with air drag.
usage: python fallfx.py out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa: E402
from scenes import CYAN, GREEN, PINK, VIOLET, YEL, ORANGE, Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, INK, VS, clamp, font, tone  # noqa: E402

BG = (10, 6, 28)
H0 = 828.0
G = 9.81
RHO = 1.2
OBJ = [  # name, mass kg, diameter m, Cd, area override
    ("BOWLING BALL", 6.35, 0.218, 0.47, None),
    ("BASKETBALL", 0.62, 0.24, 0.47, None),
    ("PENNY", 0.0025, 0.01905, 0.55, math.pi * 0.009525 ** 2),
]


def simulate(m, d, cd, area=None, dt=0.002, tmax=90.0):
    A = area if area else math.pi * (d / 2) ** 2
    k = 0.5 * RHO * cd * A / m
    y, v, t = 0.0, 0.0, 0.0
    ts, ys, vs = [0.0], [0.0], [0.0]
    while y < H0 and t < tmax:
        a = G - k * v * v
        v += a * dt
        y += v * dt
        t += dt
        ts.append(t); ys.append(min(y, H0)); vs.append(v)
    vt = math.sqrt(G / k)
    return np.array(ts), np.array(ys), np.array(vs), t, vt


class FallReel(Reel2):
    LANES = (190, 390, 590, 790)

    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.sims = cfg["sims"]
        self.t0, self.W = cfg["t0"], cfg["W"]
        yy, xx = np.mgrid[0:VS, 0:VS]
        base = np.zeros((VS, VS, 3), float) + np.array(BG)
        base += (np.clip(1 - np.hypot(xx - 480, yy - 500) / 640, 0, 1) ** 2 * 40)[..., None] * np.array([0.3, 0.6, 1.0])
        self.base = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(self.base, "RGBA")
        self.top, self.bot = 120, 860
        for x in self.LANES:
            d.line([(x, self.top), (x, self.bot)], fill=(160, 150, 230, 60), width=3)
        for m in range(0, 829, 100):
            y = self.top + (self.bot - self.top) * m / H0
            d.line([(40, y), (70, y)], fill=(200, 195, 240, 120), width=2)
            d.text((76, y), f"{int(H0 - m) if m else 828}m", font=font(F_MONO, 16), fill=(200, 195, 240, 170), anchor="lm")
        d.line([(30, self.bot), (930, self.bot)], fill=(255, 255, 255, 140), width=4)
        d.text((480, self.bot + 26), "GROUND", font=font(F_MONO, 20), fill=(200, 195, 240, 200), anchor="mm")
        d.text((480, 70), "FROM THE TOP OF THE BURJ KHALIFA (828 M)", font=font(F_MONO, 22), fill=(220, 215, 250, 255), anchor="mm")

    def sim_t(self, t):
        return max(0.0, (t - self.t0) * self.W)

    def state(self, i, st):
        """i: 0..2 objects, 3 ghost. returns y (m), v (m/s), landed_time or None"""
        if i == 3:
            tl = math.sqrt(2 * H0 / G)
            if st >= tl:
                return H0, G * tl, tl
            return 0.5 * G * st * st, G * st, None
        ts, ys, vs, tl, vt = self.sims[i]
        if st >= tl:
            return H0, vs[-1], tl
        j = int(st / 0.002)
        return ys[j], vs[j], None

    def content(self, t):
        st = self.sim_t(t)
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        names = [o[0] for o in OBJ] + ["NO AIR"]
        cols = [(150, 120, 255), ORANGE, YEL, (255, 255, 255)]
        for i, x in enumerate(self.LANES):
            y_m, v, tl = self.state(i, st)
            y = self.top + (self.bot - self.top) * y_m / H0
            r = (22, 21, 12, 16)[i]
            col = cols[i]
            if i == 3:
                d.ellipse([x - r, y - r, x + r, y + r], outline=col + (255,), width=4)
            else:
                d.ellipse([x - r - 6, y - r - 6, x + r + 6, y + r + 6], fill=col + (60,))
                d.ellipse([x - r, y - r, x + r, y + r], fill=col + (255,), outline=INK + (255,), width=2)
            d.text((x, 100), names[i], font=font(F_MONO, 19), fill=col + (255,), anchor="mm")
            if tl is None:
                d.text((x, self.bot - 28), f"{v * 3.6:0.0f} km/h", font=font(F_MONO, 22), fill=(235, 230, 255, 255), anchor="mm")
            else:
                d.rounded_rectangle([x - 84, self.bot - 66, x + 84, self.bot - 16], radius=14, fill=(14, 10, 34, 235), outline=col + (255,), width=3)
                d.text((x, self.bot - 41), f"LANDED {tl:0.1f}s", font=font(F_BOLD, 21), fill=col + (255,), anchor="mm")
        return im

    def stats(self, t):
        st = self.sim_t(t)
        _, vp, tl = self.state(2, st)
        _, vb, tb = self.state(0, st)
        frac = clamp(self.state(2, st)[0] / H0)
        shown = min(st, self.sims[2][3])
        return [(f"CLOCK {shown:0.1f} s", YEL), (f"PENNY {vp * 3.6:0.0f} KM/H", ORANGE)], frac, "PENNY FALLING", ORANGE


def build(model):
    sims = [simulate(*o[1:]) for o in OBJ]
    names = [o[0] for o in OBJ]
    tl = [s[3] for s in sims]
    vt = [s[4] for s in sims]
    vac = math.sqrt(2 * H0 / G)
    print("land times", dict(zip(names, [round(x, 2) for x in tl])), "vacuum", round(vac, 2), "terminal km/h", [round(v * 3.6, 1) for v in vt], "mph", [round(v * 2.237, 1) for v in vt])
    t0, W = 1.0, 2.2
    pay = t0 + tl[2] / W
    r = lambda s: t0 + s / W  # noqa: E731
    cfg = dict(T=60.0, c1=ORANGE, c2=CYAN, sims=sims, t0=t0, W=W, payoff=round(pay, 1), seed=191,
               still_times=[0.5, 3, r(vac) + 0.5, r(tl[0]) + 0.5, r(tl[1]) + 0.5, pay - 1.5, pay + 1.5],
               header=lambda tt: (([("PENNY OFF THE", (255, 255, 255)), ("TALLEST BUILDING", YEL)], "HOW LONG TO LAND?") if tt < pay else
                                  ([("THE PENNY TAKES", (255, 255, 255)), (f"{tl[2]:0.0f} SECONDS.", ORANGE)], "AIR DOES THE REST")))
    lines = [(0.2, f"If you dropped a penny off the world's tallest building, it would take {tl[2]:0.0f} seconds to land."),
             (5.3, "Four drops from 828 meters. The white ring has no air."),
             (9.5, f"With no air, it lands in {vac:0.1f} seconds. The bowling ball takes {tl[0]:0.0f}."),
             (r(tl[0]) + 6.1, f"A basketball takes {tl[1]:0.0f}. Air stops speeding it up at about {vt[1] * 3.6:0.0f} kilometers an hour."),
             (r(tl[1]) + 1.2, f"The penny tops out around {vt[2] * 3.6:0.0f}. Slow and harmless."),
             (pay + 0.6, "Want to build your own app instead? Use code AT100 on Base44. Comment DROP for the prompt.")]
    buf, vo = make_audio(cfg, lines, model)
    chime(buf, cfg["payoff"])
    cfg["T"] = vo[-1][1] + 2.4
    reel = FallReel(cfg, vo)
    post = dict(title="DROP-1  drop a penny off the tallest building (real physics sim)",
                caption=f"If you dropped a penny off the Burj Khalifa (828 m), it would take about {tl[2]:.0f} seconds to land, because air drag stops it speeding up at roughly {vt[2] * 3.6:.0f} km/h. In my physics simulation a bowling ball lands in {tl[0]:.0f} s, a basketball in {tl[1]:.0f} s, and with no air at all everything lands in {vac:.1f} s.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment DROP and I'll send the prompt so you can build it.",
                tags=["physics", "science", "simulation", "howitworks", "coding", "shipwithpip"],
                pinned="Build me a single-file HTML canvas page that simulates dropping a bowling ball, basketball and penny from 828 m with quadratic air drag (a = g - 0.5*rho*Cd*A/m * v^2, rho 1.2, g 9.81), plus a 'no air' ghost ball. Show a live clock, each speed in km/h, and mark the landing time for each.",
                notes=f"Physics sim (quadratic drag, constant air density 1.2 kg/m3, g 9.81, dt 0.002 s). Drop height 828 m (Burj Khalifa, tallest building as of Oct 2026; Jeddah Tower is not expected until about 2028).\nObjects: bowling ball 6.35 kg, d 0.218 m, Cd 0.47; basketball 0.62 kg, d 0.24 m, Cd 0.47; penny 2.5 g, d 19.05 mm, Cd 0.55 (tumbling, chosen so terminal speed lands in the sourced 30-50 mph range).\nResults: land times {dict(zip(names, [round(x, 1) for x in tl]))}, no-air {vac:.1f} s. Terminal speeds: {dict(zip(names, [f'{v * 3.6:.0f} km/h ({v * 2.237:.0f} mph)' for v in vt]))}.\nSimplifications: air density is constant (it really thins with height), no wind, no tumbling randomness, objects don't collide with the building. The video plays at 2.2x speed with the real clock on screen.\nSources: penny terminal velocity 30 to 50 mph and not lethal: https://mechse.illinois.edu/blogs/lethal-pennies-and-other-misconceptions . Burj Khalifa 828 m: https://www.enr.com/articles/431-world-s-tallest-building-is-828-meters-says-developer . Jeddah Tower expected by 2028: https://www.theweek.in/news/middle-east/2026/08/16/jeddah-tower-to-dethrone-burj-khalifa-as-worlds-tallest-building-by-2028.html .\nFormat inspired by 'what if you dropped X from a skyscraper' physics hooks; script and visuals are original.")
    return reel, buf, cfg, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = build(a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
