"""Minimal dark physically-accurate fun-fact reels: venus | sunlight.
usage: python minfx.py venus|sunlight out.mp4 --model <kokoro dir> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa: E402
from scenes import Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, F_SEMI, H, INK, W, clamp, font, tw  # noqa: E402

WHITE = (255, 255, 255)
GOLD = (255, 214, 130)
ICE = (140, 215, 255)
SOFT = (170, 178, 200)
DIM = (80, 88, 112)


class MinReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        yy = np.linspace(0, 1, H)[:, None, None]
        base = np.array([9, 11, 22.0]) * (1 - yy) + np.array([3, 4, 9.0]) * yy
        base = np.broadcast_to(base, (H, W, 3)).copy()
        Y, X = np.mgrid[0:H, 0:W]
        v = np.clip(1 - np.hypot((X - W / 2) / (W * 0.9), (Y - H * 0.47) / (H * 0.7)), 0, 1)
        base *= (0.55 + 0.45 * v)[..., None]
        self.sky = Image.fromarray(base.astype(np.uint8), "RGB")
        rs = np.random.default_rng(8)
        self.stars = [(int(rs.integers(0, W)), int(rs.integers(0, H)), float(rs.uniform(0.2, 0.8)), float(rs.uniform(0, 6.28))) for _ in range(140)]

    def new_frame(self, t):
        img = self.sky.copy()
        d = ImageDraw.Draw(img, "RGBA")
        for x, y, b, ph in self.stars:
            a = int(255 * b * (0.55 + 0.45 * math.sin(t * 0.7 + ph)))
            d.rectangle([x, y, x + 1, y + 1], fill=(200, 215, 255, a // 3))
        return img

    def title(self, d, lines, tag):
        f = font(F_MONO, 30)
        d.text((W / 2, 96), tag, font=f, fill=SOFT + (230,), anchor="mm")
        for i, (txt, col) in enumerate(lines):
            d.text((W / 2, 205 + i * 112), txt, font=font(F_BOLD, 92), fill=col + (255,), anchor="mm")

    def glow_paste(self, img, layer_fn, blur=24, boost=1.0):
        g = Image.new("RGB", (W // 4, H // 4), (0, 0, 0))
        layer_fn(ImageDraw.Draw(g), 0.25)
        g = g.filter(ImageFilter.GaussianBlur(blur / 4)).resize((W, H), Image.BICUBIC)
        from PIL import ImageChops
        return ImageChops.add(img, g.point(lambda v: int(min(255, v * boost))))

    def bar(self, d, y, frac, label, col, locked=False):
        x0, x1 = 140, W - 140
        d.text((x0, y - 34), label, font=font(F_MONO, 28), fill=SOFT + (255,), anchor="lm")
        d.rounded_rectangle([x0, y, x1, y + 10], radius=5, fill=(255, 255, 255, 30))
        if frac > 0.002:
            d.rounded_rectangle([x0, y, x0 + (x1 - x0) * clamp(frac), y + 10], radius=5, fill=col + (255,))
        if locked:
            d.ellipse([x1 - 7, y - 2, x1 + 7, y + 12], fill=col + (255,))

    def handle(self, img, t):
        d = ImageDraw.Draw(img, "RGBA")
        self.caption(d, t)
        self.cta(d, t)
        d.rectangle([0, 0, W * clamp(t / self.T), 5], fill=GOLD + (200,))
        return img


# ====================================================================== VENUS
ORB, SPIN = 224.701, 243.0185  # sidereal orbital period and rotation period, Earth days


class VenusReel(MinReel):
    CX, CY, R = W // 2, 930, 360
    T0, TD = 0.8, 26.0  # animation start and length (seconds) for SPIN days

    def days(self, t):
        return clamp((t - self.T0) / self.TD) * SPIN

    def content_t(self, t):
        d_ = self.days(t)
        img = self.new_frame(t)
        dr = ImageDraw.Draw(img, "RGBA")
        cx, cy, R = self.CX, self.CY, self.R
        th = 2 * math.pi * d_ / ORB
        px, py = cx + R * math.cos(th), cy - R * math.sin(th)
        # orbit ring (full faint, travelled part brighter)
        dr.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(255, 255, 255, 26), width=2)
        n = 180
        arc = [(cx + R * math.cos(th * i / n), cy - R * math.sin(th * i / n)) for i in range(n + 1)]
        if len(arc) > 1 and th > 0.01:
            dr.line(arc, fill=GOLD + (150,), width=3, joint="curve")
        dr.line([cx + R - 18, cy, cx + R + 18, cy], fill=(255, 255, 255, 90), width=3)  # start line
        # sun
        def sun_layer(g, k):
            g.ellipse([(cx - 34) * k, (cy - 34) * k, (cx + 34) * k, (cy + 34) * k], fill=(255, 190, 90))
        img = self.glow_paste(img, sun_layer, blur=60, boost=1.4)
        dr = ImageDraw.Draw(img, "RGBA")
        dr.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=(255, 238, 200, 255))
        # venus disk shaded from sun
        r = 74
        S = r * 2 + 40
        yy, xx = np.mgrid[0:S, 0:S].astype(float)
        nx, ny = (xx - S / 2) / r, (yy - S / 2) / r
        rr2 = nx ** 2 + ny ** 2
        inside = rr2 <= 1
        nz = np.sqrt(np.clip(1 - rr2, 0, 1))
        lx, ly = (cx - px), (cy - py)
        ln = math.hypot(lx, ly)
        lx, ly = lx / ln, ly / ln
        lam = np.clip(nx * lx + ny * ly + 0.0 * nz, 0, 1) ** 0.8
        lam = 0.06 + 0.94 * lam
        col = np.array([236, 208, 150.0])
        rgb = col[None, None, :] * lam[..., None]
        halo = np.clip(1.25 - np.sqrt(rr2), 0, 1)
        alpha = np.where(inside, 255, (np.clip((1.0 + 0.16 - np.sqrt(rr2)) / 0.16, 0, 1) * 70)).astype(np.uint8)
        rgb = np.where(inside[..., None], rgb, np.array([255, 224, 160.0])[None, None, :] * 0.8 * np.clip(lam.mean(), 0.1, 1))
        tile = Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha]), "RGBA")
        img.paste(tile, (int(px - S / 2), int(py - S / 2)), tile)
        dr = ImageDraw.Draw(img, "RGBA")
        # fixed reference direction + spin marker (clockwise = retrograde)
        dr.line([px + r + 6, py, px + r + 90, py], fill=(255, 255, 255, 90), width=2)
        ph = -2 * math.pi * d_ / SPIN
        mx, my = px + (r + 4) * math.cos(ph), py - (r + 4) * math.sin(ph)
        ex, ey = px + (r + 46) * math.cos(ph), py - (r + 46) * math.sin(ph)
        dr.line([mx, my, ex, ey], fill=ICE + (255,), width=5)
        dr.ellipse([ex - 9, ey - 9, ex + 9, ey + 9], fill=ICE + (255,))
        # labels
        dr.text((cx, cy + 58), "SUN", font=font(F_MONO, 24), fill=SOFT + (200,), anchor="mm")
        return img, d_

    def frame(self, t):
        img, d_ = self.content_t(t)
        d = ImageDraw.Draw(img, "RGBA")
        yr_done = d_ >= ORB
        if t < self.T0 + self.TD * ORB / SPIN + 0.4:
            lines, tag = [("A DAY ON VENUS", WHITE), ("IS LONGER", GOLD), ("THAN ITS YEAR", WHITE)], "REAL ORBIT. REAL SPIN."
        else:
            lines, tag = [("YEAR OVER.", GOLD), ("DAY ISN'T.", WHITE)], f"{SPIN - ORB:0.0f} MORE DAYS TO GO"
        self.title(d, lines, tag)
        d.text((W / 2, 1420), f"{int(d_)}", font=font(F_BOLD, 130), fill=WHITE + (255,), anchor="mm")
        d.text((W / 2, 1495), "EARTH DAYS", font=font(F_MONO, 28), fill=SOFT + (255,), anchor="mm")
        self.bar(d, 1560, d_ / ORB, f"ONE VENUS YEAR  {ORB:0.1f} days", GOLD, yr_done)
        self.bar(d, 1640, d_ / SPIN, f"ONE VENUS DAY   {SPIN:0.0f} days", ICE, d_ >= SPIN - 0.01)
        return self.handle(img, t)


def build_venus(model):
    t_year = 0.8 + 26.0 * ORB / SPIN
    t_day = 0.8 + 26.0
    cfg = dict(T=60, c1=GOLD, c2=ICE, payoff=None, seed=211, still_times=[0.4, 6, 13, t_year - 0.5, t_year + 1.5, t_day + 0.5], header=lambda t: ([], ""))
    lines = [(0.2, "On Venus, a day is longer than a year."),
             (3.4, "Venus goes around the Sun once every 225 Earth days. That's its year."),
             (9.0, "The blue marker shows it spinning. One full turn is one day."),
             (14.0, "And it spins backwards. On Venus, the Sun rises in the west."),
             (19.0, "Watch the year counter."),
             (t_year + 0.2, "The year is over. The marker hasn't finished its turn."),
             (t_day + 0.1, f"It takes {SPIN:0.0f} days to spin once. Its year ends {SPIN - ORB:0.0f} days sooner."),
             (t_day + 6.0, "Follow for more.")]
    lines = [list(x) for x in lines]
    for _ in range(8):
        buf, vo = make_audio(cfg, [tuple(x) for x in lines], model)
        bad = False
        for i in range(len(vo) - 1):
            if vo[i][1] + 0.3 > lines[i + 1][0]:
                lines[i + 1][0] = vo[i][1] + 0.3
                bad = True
        if not bad:
            break
    chime(buf, t_day)
    cfg["T"] = max(vo[-1][1] + 2.4, t_day + 3)
    reel = VenusReel(cfg, vo)
    post = dict(title="FACT-1  A day on Venus is longer than its year",
                caption=f"Venus orbits the Sun in about {ORB:0.1f} Earth days, but takes about {SPIN:0.0f} Earth days to spin once on its axis. So its day is longer than its year. It also spins backwards (retrograde), so the Sun rises in the west. The animation uses the real periods; sizes and distances are not to scale.",
                tags=["space", "venus", "didyouknow", "science", "facts", "shipwithpip"],
                pinned="A fact most people don't know: because Venus spins so slowly and backwards, one sunrise to the next takes about 117 Earth days.",
                notes=f"Venus sidereal orbital period {ORB} days; sidereal rotation period {SPIN} days (retrograde). Source: NASA Planetary Fact Sheet, nssdc.gsfc.nasa.gov/planetary/factsheet/venusfact.html. Solar day (sunrise to sunrise) is about 116.75 Earth days. Animation runs {SPIN:.0f} Earth days in 26 seconds (about {SPIN / 26:.1f} days/s). Orbit radius, planet size and Sun size are not to scale. Verify the NASA values before posting.")
    return reel, buf, cfg, post


# ====================================================================== SUNLIGHT
AU_KM, C_KMS = 149597870.7, 299792.458
LT = AU_KM / C_KMS  # 499.0 s


def mmss(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


class LightReel(MinReel):
    SX, EX, Y = 150, 930, 930
    TV, TA = 2.0, 16.0  # vanish time (s), compressed travel duration (s)

    def lsec(self, t):
        return clamp((t - self.TV) / self.TA) * LT

    def frame(self, t):
        img = self.new_frame(t)
        s = self.lsec(t)
        k = s / LT
        sx, ex, y = self.SX, self.EX, self.Y
        gone = t >= self.TV
        arrived = k >= 1.0
        tail = sx + (ex - sx) * k

        def layer(g, sc):
            if not gone:
                g.ellipse([(sx - 50) * sc, (y - 50) * sc, (sx + 50) * sc, (y + 50) * sc], fill=(255, 190, 90))
            if not arrived:
                g.rectangle([tail * sc, (y - 5) * sc, (ex - 40) * sc, (y + 5) * sc], fill=(255, 220, 150))
                g.ellipse([(ex - 80) * sc, (y - 80) * sc, (ex + 80) * sc, (y + 80) * sc], fill=(60, 90, 140))
        img = self.glow_paste(img, layer, blur=40, boost=1.2)
        d = ImageDraw.Draw(img, "RGBA")
        d.line([sx, y, ex, y], fill=(255, 255, 255, 28), width=2)
        # sun
        if not gone:
            d.ellipse([sx - 30, y - 30, sx + 30, y + 30], fill=(255, 240, 205, 255))
        else:
            d.ellipse([sx - 30, y - 30, sx + 30, y + 30], fill=(8, 9, 18, 255), outline=(255, 255, 255, 60), width=2)
        d.text((sx, y + 70), "SUN", font=font(F_MONO, 24), fill=SOFT + (220,), anchor="mm")
        # beam
        if not arrived:
            d.line([tail, y, ex - 40, y], fill=(255, 236, 190, 255), width=4)
        # ticks (real light-times)
        for name, au in (("MERCURY", 0.3871), ("VENUS", 0.7233)):
            xx = sx + (ex - sx) * au  # scaled to AU: Earth = 1.0
            hit = tail >= xx
            d.line([xx, y - 16, xx, y + 16], fill=(255, 255, 255, 200 if not hit else 70), width=3)
            d.text((xx, y - 52), name, font=font(F_MONO, 22), fill=SOFT + (230 if not hit else 90,), anchor="mm")
            d.text((xx, y + 56), mmss(au * LT), font=font(F_MONO, 24), fill=GOLD + (230 if not hit else 90,), anchor="mm")
        # earth shaded: lit side faces the sun
        r = 40
        S = r * 2 + 8
        yy, xx2 = np.mgrid[0:S, 0:S].astype(float)
        nx, ny = (xx2 - S / 2) / r, (yy - S / 2) / r
        inside = nx ** 2 + ny ** 2 <= 1
        lam = np.clip(-nx, 0, 1) ** 0.7
        lit = 0.04 if arrived else 1.0
        col = np.array([70, 130, 230.0])
        rgb = col[None, None, :] * (0.07 + 0.93 * lam[..., None] * lit)
        tile = Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), (inside * 255).astype(np.uint8)]), "RGBA")
        img.paste(tile, (int(ex - S / 2), int(y - S / 2)), tile)
        d = ImageDraw.Draw(img, "RGBA")
        d.text((ex, y + 78), "EARTH", font=font(F_MONO, 24), fill=SOFT + (220,), anchor="mm")
        # titles
        if t < self.TV + 0.2:
            lines, tag = [("THE SUN JUST", WHITE), ("VANISHED.", GOLD)], "WOULD YOU NOTICE?"
        elif not arrived:
            lines, tag = [("EARTH IS STILL", WHITE), ("LIT.", GOLD)], "THE LAST LIGHT IS STILL TRAVELLING"
        else:
            lines, tag = [("8 MIN 19 SEC", GOLD), ("LATER...", WHITE)], "LIGHTS OUT"
        self.title(d, lines, tag)
        d.text((W / 2, 1380), mmss(s), font=font(F_BOLD, 170), fill=WHITE + (255,), anchor="mm")
        d.text((W / 2, 1480), "SINCE THE SUN VANISHED", font=font(F_MONO, 28), fill=SOFT + (255,), anchor="mm")
        if t >= self.TV + self.TA + 0.1 and t < self.TV + self.TA + 3.0:
            a = int(255 * clamp(1 - (t - self.TV - self.TA) / 3.0))
            d.text((W / 2, 1180), "NOW IT KNOWS", font=font(F_BOLD, 70), fill=GOLD + (a,), anchor="mm")
        return self.handle(img, t)


def build_sunlight(model):
    TV, TA = LightReel.TV, LightReel.TA
    tend = TV + TA
    t_at = lambda sec: TV + TA * sec / LT  # noqa: E731
    moon = 384400 / C_KMS
    cfg = dict(T=60, c1=GOLD, c2=ICE, payoff=None, seed=212, still_times=[0.5, TV + 1, t_at(193), t_at(361), tend + 0.3, tend + 2.5], header=lambda t: ([], ""))
    lines = [(0.2, "If the Sun vanished right now, you wouldn't know for eight minutes."),
             (TV + 0.4, "Sunlight takes eight minutes and nineteen seconds to reach us."),
             (t_at(193) - 0.6, f"The last light passes Mercury at {mmss(0.3871 * LT)}."),
             (t_at(361) - 0.6, f"Venus at {mmss(0.7233 * LT)}."),
             (tend - 2.4, "Earth would still be lit, still orbiting. Gravity travels at light speed too."),
             (tend + 0.3, "Then, lights out."),
             (tend + 3.4, f"For comparison, light reaches the Moon in just {moon:0.1f} seconds."),
             (tend + 8.4, "Follow for more.")]
    lines = [list(x) for x in lines]
    for _ in range(8):
        buf, vo = make_audio(cfg, [tuple(x) for x in lines], model)
        bad = False
        for i in range(len(vo) - 1):
            if vo[i][1] + 0.3 > lines[i + 1][0]:
                lines[i + 1][0] = vo[i][1] + 0.3
                bad = True
        if not bad:
            break
    chime(buf, tend, 262)
    cfg["T"] = max(vo[-1][1] + 2.4, tend + 3)
    reel = LightReel(cfg, vo)
    post = dict(title="FACT-2  The Sun could vanish and you wouldn't know for 8 minutes",
                caption=f"Light takes about {LT:0.0f} seconds ({mmss(LT)}) to travel the 1 AU from the Sun to Earth. If the Sun disappeared, Earth would stay lit for that long. According to general relativity, gravity's changes also propagate at light speed, so Earth would keep orbiting until then too. Light reaches Mercury in about {mmss(0.3871 * LT)}, Venus in about {mmss(0.7233 * LT)}, and the Moon in {moon:0.1f} seconds. Distances in the animation are scaled by orbit size (Mercury 0.387 AU, Venus 0.723 AU); the Sun and planets are drawn bigger than real.",
                tags=["space", "science", "didyouknow", "sun", "physics", "shipwithpip"],
                pinned="Fun follow-up: the sunlight you feel today left the Sun about 8 minutes ago, but the energy in it took tens of thousands of years or more to get out of the Sun's core.",
                notes=f"1 AU = {AU_KM:,.1f} km (IAU 2012). Speed of light = {C_KMS:,.3f} km/s. 1 AU / c = {LT:.1f} s = {mmss(LT)}. Mercury 0.3871 AU -> {0.3871 * LT:.0f} s; Venus 0.7233 AU -> {0.7233 * LT:.0f} s; Earth-Moon 384,400 km -> {moon:.2f} s. Gravity-speed claim: GW170817 constrains gravity's speed to equal c within about 1e-15 (Abbott et al. 2017). Pinned-comment claim (core-to-surface time, 'tens of thousands of years or more') is a commonly cited range with real uncertainty (estimates vary widely); verify or delete before posting. Animation compresses {LT:.0f} s into {TA:.0f} s.")
    return reel, buf, cfg, post



# ====================================================================== CAFFEINE
HL, MG = 5.0, 95.0  # average half-life (h), mg in a typical 8 oz brewed coffee (USDA ~95 mg)


def clock(h0, hours):
    h = (h0 + hours) % 24
    ap = "AM" if h < 12 else "PM"
    hh = int(h) % 12 or 12
    return f"{hh}:{int((h % 1) * 60):02d} {ap}"


class CaffeineReel(MinReel):
    T0, TD = 1.0, 24.0  # 12 hours in TD seconds
    GX0, GX1, GY0, GY1 = 120, 960, 1360, 600

    def hrs(self, t):
        return clamp((t - self.T0) / self.TD) * 12

    def frame(self, t):
        h = self.hrs(t)
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        gx = lambda x: self.GX0 + (self.GX1 - self.GX0) * x / 12  # noqa: E731
        gy = lambda p: self.GY0 + (self.GY1 - self.GY0) * p  # noqa: E731
        d.line([self.GX0, self.GY0, self.GX1, self.GY0], fill=(255, 255, 255, 60), width=2)
        for hh in (0, 3, 6, 9, 12):
            d.line([gx(hh), self.GY0, gx(hh), self.GY0 + 12], fill=(255, 255, 255, 90), width=2)
            d.text((gx(hh), self.GY0 + 40), clock(15, hh).replace(":00 ", " "), font=font(F_MONO, 24), fill=SOFT + (230,), anchor="mm")
        for p in (0.5, 0.25):
            d.line([self.GX0, gy(p), self.GX1, gy(p)], fill=(255, 255, 255, 22), width=2)
            d.text((self.GX0 - 14, gy(p)), f"{int(p * 100)}%", font=font(F_MONO, 22), fill=DIM + (255,), anchor="rm")
        d.text((self.GX0 - 14, gy(1)), "100%", font=font(F_MONO, 22), fill=DIM + (255,), anchor="rm")
        # bedtime marker 11 PM = 8 h
        bx = gx(8)
        d.line([bx, gy(1) - 20, bx, self.GY0], fill=ICE + (110,), width=2)
        d.text((bx, gy(1) - 40), "BEDTIME", font=font(F_MONO, 24), fill=ICE + (230,), anchor="mm")
        n = 160
        pts = [(gx(h * i / n), gy(0.5 ** (h * i / n / HL))) for i in range(n + 1)]
        def layer(g, k):
            if len(pts) > 1:
                g.line([(x * k, y * k) for x, y in pts], fill=(255, 170, 70), width=int(14 * k))
        img = self.glow_paste(img, layer, blur=18, boost=1.0)
        d = ImageDraw.Draw(img, "RGBA")
        if len(pts) > 1:
            d.line(pts, fill=(255, 214, 140, 255), width=6, joint="curve")
        d.ellipse([pts[-1][0] - 12, pts[-1][1] - 12, pts[-1][0] + 12, pts[-1][1] + 12], fill=WHITE + (255,))
        left = 0.5 ** (h / HL)
        if t < self.T0 + self.TD + 0.2:
            lines, tag = [("YOUR 3PM COFFEE", WHITE), ("IS STILL IN YOU", GOLD)], "AT BEDTIME"
        else:
            lines, tag = [("A THIRD OF IT", GOLD), ("AT BEDTIME", WHITE)], "SAVE THIS"
        self.title(d, lines, tag)
        d.text((W / 2, 1520), clock(15, h), font=font(F_BOLD, 84), fill=WHITE + (255,), anchor="mm")
        d.text((W / 2, 1596), f"{left * 100:0.0f}% LEFT   -   {left * MG:0.0f} mg of {MG:0.0f}", font=font(F_MONO, 30), fill=GOLD + (255,), anchor="mm")
        if h >= 8 and h < 9.6:
            d.text((bx, gy(0.5 ** (8 / HL)) - 56), f"{0.5 ** (8 / HL) * 100:0.0f}%", font=font(F_BOLD, 64), fill=ICE + (255,), anchor="mm")
        if t > self.T0 + self.TD + 1.0:
            a = int(255 * clamp((t - self.T0 - self.TD - 1.0) / 0.6))
            d.rounded_rectangle([110, 430, W - 110, 525], radius=30, fill=(10, 14, 30, int(a * 0.9)), outline=GOLD + (a,), width=3)
            d.text((W / 2, 478), "LAST COFFEE: 8+ HOURS BEFORE BED", font=font(F_SEMI, 36), fill=WHITE + (a,), anchor="mm")
        return self.handle(img, t)


def seq_audio(cfg, lines, model):
    lines = [list(x) for x in lines]
    cfg["T"] = max(cfg["T"], 90)
    for _ in range(8):
        buf, vo = make_audio(cfg, [tuple(x) for x in lines], model)
        bad = False
        for i in range(len(vo) - 1):
            if vo[i][1] + 0.3 > lines[i + 1][0]:
                lines[i + 1][0] = vo[i][1] + 0.3
                bad = True
        if not bad:
            break
    return buf, vo


def build_caffeine(model):
    T0, TD = CaffeineReel.T0, CaffeineReel.TD
    tb = T0 + TD * 8 / 12
    tend = T0 + TD
    at8 = 0.5 ** (8 / HL)
    at9 = 0.5 ** (9 / HL)
    cfg = dict(T=60, c1=GOLD, c2=ICE, payoff=None, seed=213, still_times=[0.4, T0 + 6, T0 + 12, tb + 0.4, tend + 0.5, tend + 3], header=lambda t: ([], ""))
    lines = [(0.2, "The coffee you drink at three p m is still in your body at midnight."),
             (T0 + 3.2, f"Caffeine has a half life of about five hours. Half of it is gone every five hours."),
             (T0 + 8.5, f"By eight p m, half is left. By one in the morning, a quarter."),
             (tb - 0.8, f"At bedtime, eleven p m, about {at8 * 100:0.0f} percent is still there."),
             (tend + 0.3, "That's about thirty milligrams from one cup, still keeping you wired."),
             (tend + 4.8, "One study found caffeine six hours before bed cost people over an hour of sleep, and they didn't feel it."),
             (tend + 12.2, "So the last coffee: eight hours before bed. Save this.")]
    buf, vo = seq_audio(cfg, lines, model)
    chime(buf, tend, 392)
    cfg["T"] = max(vo[-1][1] + 2.4, tend + 3)
    reel = CaffeineReel(cfg, vo)
    post = dict(title="SAVE-1  Your 3pm coffee at midnight",
                caption=f"Caffeine's average half-life is about 5 hours (it varies a lot between people, roughly 1.5 to 9.5 hours). Start with a typical 95 mg cup at 3 PM: about {at8 * MG:0.0f} mg ({at8 * 100:0.0f}%) is still in your system at 11 PM, and about {at9 * 100:0.0f}% at midnight. In a 2013 sleep study, 400 mg of caffeine taken 6 hours before bed cost people more than an hour of sleep, even though they didn't notice. Rule of thumb: last coffee 8+ hours before bed. Save this one.",
                tags=["sleep", "coffee", "caffeine", "health", "didyouknow", "shipwithpip"],
                pinned="Quick math you can screenshot: caffeine left = dose x 0.5^(hours / 5). Coffee at 3 PM, bed at 11 PM: 95 mg x 0.5^(8/5) = about 31 mg.",
                notes=f"Half-life ~5 h is the commonly cited average; individual range ~1.5-9.5 h (smoking, pregnancy, liver function, medications and genetics change it). Model: remaining = 0.5^(h/5), starting at 100% (ignores absorption time, which peaks ~30-60 min after drinking). 8 h -> {at8:.3f}; 9 h -> {at9:.3f}. mg: 95 mg is an approximate USDA figure for 8 fl oz brewed coffee; real cups range widely. Study: Drake C, Roehrs T, Shambroom J, Roth T. 'Caffeine effects on sleep taken 0, 3, or 6 hours before going to bed.' J Clin Sleep Med 2013;9(11):1195-1200 - 400 mg, 6 h before bed reduced total sleep by more than 1 hour (measured), subjects did not perceive it. I could not open the paper this session (PMC blocked my fetch), so verify the numbers before posting. Not medical advice. Do not post without checking.")
    return reel, buf, cfg, post


# ====================================================================== THUNDER
VS_SND = 343.0  # m/s at 20 C


class ThunderReel(MinReel):
    TF, DIST = 2.0, 3090.0  # flash time (s); distance (m) -> thunder after 9.0 s
    X0, X1, Y = 130, 960, 900

    def frame(self, t):
        img = self.new_frame(t)
        s = max(0.0, t - self.TF)
        xs = lambda m: self.X0 + (self.X1 - self.X0) * m / 4000  # noqa: E731
        stormx = xs(self.DIST)
        front = min(self.DIST, VS_SND * s)
        heard = s * VS_SND >= self.DIST and t >= self.TF
        flash = clamp(1 - (t - self.TF) / 0.5) if t >= self.TF else 0

        def layer(g, k):
            if flash > 0:
                f = int(255 * flash)
                g.ellipse([(stormx - 110) * k, (self.Y - 300) * k, (stormx + 110) * k, (self.Y - 80) * k], fill=(f // 2, f // 2, f))
        img = self.glow_paste(img, layer, blur=50, boost=1.3)
        d = ImageDraw.Draw(img, "RGBA")
        d.line([self.X0 - 40, self.Y, self.X1 + 40, self.Y], fill=(255, 255, 255, 40), width=2)
        # ground ticks every km with real seconds
        for km in range(0, 4):
            x = xs(km * 1000)
            d.line([x, self.Y, x, self.Y + 16], fill=(255, 255, 255, 110), width=2)
            d.text((x, self.Y + 46), f"{km} km", font=font(F_MONO, 24), fill=SOFT + (230,), anchor="mm")
            d.text((x, self.Y + 82), f"{km * 1000 / VS_SND:0.0f}s", font=font(F_MONO, 22), fill=GOLD + (200,), anchor="mm")
        # storm
        d.ellipse([stormx - 90, self.Y - 230, stormx + 90, self.Y - 150], fill=(34, 40, 62, 255))
        d.ellipse([stormx - 130, self.Y - 200, stormx + 20, self.Y - 130], fill=(30, 36, 56, 255))
        if flash > 0.02:
            pts = [(stormx, self.Y - 150), (stormx - 22, self.Y - 100), (stormx + 12, self.Y - 80), (stormx - 16, self.Y - 28), (stormx + 4, self.Y)]
            d.line(pts, fill=(235, 240, 255, int(255 * flash)), width=10, joint="curve")
        # you
        d.ellipse([self.X0 - 10, self.Y - 34, self.X0 + 10, self.Y - 14], fill=WHITE + (255,))
        d.line([self.X0, self.Y - 14, self.X0, self.Y], fill=WHITE + (255,), width=6)
        d.text((self.X0, self.Y - 64), "YOU", font=font(F_MONO, 22), fill=SOFT + (230,), anchor="mm")
        # sound wavefront
        if t >= self.TF:
            fx = stormx - (stormx - self.X0) * front / self.DIST
            d.line([fx, self.Y - 90, fx, self.Y + 6], fill=ICE + (255,), width=4)
            d.arc([fx - 80, self.Y - 80, fx + 80, self.Y + 80], 120, 240, fill=ICE + (140,), width=3)
        if t < self.TF + 0.4:
            lines, tag = [("COUNT FROM THE", WHITE), ("FLASH", GOLD)], "STORM SAFETY"
        elif not heard:
            lines, tag = [("HOW FAR IS", WHITE), ("THE STORM?", GOLD)], "REAL TIME. SOUND TRAVELS 343 M/S"
        else:
            lines, tag = [("9 SECONDS", GOLD), ("= 3 KM AWAY", WHITE)], "SAVE THIS"
        self.title(d, lines, tag)
        shown = min(s, self.DIST / VS_SND)
        d.text((W / 2, 1380), f"{shown:0.1f}" if not heard else f"{self.DIST / VS_SND:0.1f}", font=font(F_BOLD, 190), fill=WHITE + (255,), anchor="mm")
        d.text((W / 2, 1490), "SECONDS SINCE THE FLASH", font=font(F_MONO, 28), fill=SOFT + (255,), anchor="mm")
        if heard:
            a = int(255 * clamp((t - self.TF - self.DIST / VS_SND) / 0.8))
            d.rounded_rectangle([90, 1010, W - 90, 1150], radius=30, fill=(10, 14, 30, int(a * 0.92)), outline=GOLD + (a,), width=3)
            d.text((W / 2, 1050), "SECONDS / 3 = KM", font=font(F_SEMI, 44), fill=WHITE + (a,), anchor="mm")
            d.text((W / 2, 1112), "SECONDS / 5 = MILES", font=font(F_SEMI, 44), fill=WHITE + (a,), anchor="mm")
        return self.handle(img, t)


def build_thunder(model):
    TF, dist = ThunderReel.TF, ThunderReel.DIST
    sec = dist / VS_SND
    tth = TF + sec
    cfg = dict(T=60, c1=GOLD, c2=ICE, payoff=None, seed=214, still_times=[0.4, TF + 0.2, TF + 5, tth + 0.2, tth + 2.5, tth + 4], header=lambda t: ([], ""))
    lines = [(0.2, "Do this the next time you see lightning."),
             (TF + 0.9, "Start counting the second you see the flash."),
             (TF + 4.2, "Light arrives instantly. Sound crawls at about three hundred forty meters a second."),
             (tth - 0.3, f"Thunder. {sec:0.0f} seconds. Divide by three for kilometers, so about {dist / 1000:0.0f} km."),
             (tth + 4.0, "Or divide by five for miles. About two."),
             (tth + 8.4, "And if you can hear thunder at all, you're close enough to be struck. Go indoors. Save this.")]
    buf, vo = seq_audio(cfg, lines, model)
    chime(buf, tth, 330)
    cfg["T"] = max(vo[-1][1] + 2.4, tth + 4)
    reel = ThunderReel(cfg, vo)
    mi = dist / 1609.344
    post = dict(title="SAVE-2  The lightning counting trick",
                caption=f"Light reaches you almost instantly; sound travels about 343 m/s (roughly 1 km every 3 seconds, 1 mile every 5). Count the seconds from flash to thunder: seconds / 3 = km, seconds / 5 = miles. Here, {sec:0.0f} seconds is about {dist / 1000:0.1f} km ({mi:0.1f} miles). Safety: if you can hear thunder you are close enough to be struck, so head indoors and wait 30 minutes after the last thunder (US National Weather Service guidance). Save this one for your next storm.",
                tags=["weather", "lightning", "safety", "didyouknow", "science", "shipwithpip"],
                pinned="Cheat sheet: seconds / 3 = km. Seconds / 5 = miles. Any thunder at all = go indoors.",
                notes=f"Speed of sound ~343 m/s at 20 C (varies with temperature, about 331 m/s at 0 C). {dist:.0f} m / 343 = {sec:.1f} s. 5 s per mile: 1609.34 / 343 = 4.7 s (rule rounds to 5); 3 s per km: 1000 / 343 = 2.9 s. Animation is real-time. Safety line ('when thunder roars, go indoors', wait 30 minutes after last thunder) is the US National Weather Service lightning-safety guidance; I could not open the NWS page this session, only search results referencing it, so verify the wording at weather.gov/safety/lightning before posting. Lightning can strike miles from the storm centre.")
    return reel, buf, cfg, post


BUILDERS = {"venus": build_venus, "sunlight": build_sunlight, "caffeine": build_caffeine, "thunder": build_thunder}


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
