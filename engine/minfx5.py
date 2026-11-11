"""3 short viral-format reels: weeks | seconds | onepct.  usage: python minfx5.py NAME out.mp4 --model <kokoro dir> [--stills png]"""
import argparse
import math
import os
import sys

from PIL import ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa: E402
from minfx import DIM, GOLD, ICE, SOFT, WHITE, MinReel  # noqa: E402
from minfx2 import GREEN, RED, big, card, finish_reel, mkcfg  # noqa: E402
from scenes import render  # noqa: E402
from simfx import F_BOLD, F_MONO, F_SEMI, W, clamp, font  # noqa: E402


# ============================================================== 1. 4,000 weeks
class WeeksReel(MinReel):
    T0, RATE = 2.0, 3.2  # years per second

    def age(self, t):
        return clamp((t - self.T0) * self.RATE / 80) * 80

    def frame(self, t):
        a = self.age(t)
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        cs, cols, rows = 11, 52, 80
        x0, y0 = (W - cols * cs) // 2, 430
        used = int(a * 52)
        for i in range(cols * rows):
            r, c = divmod(i, cols)
            x, y = x0 + c * cs, y0 + r * cs
            if i < used:
                col = (30, 36, 58, 255)
            else:
                col = (120, 138, 196, 255)
            d.rectangle([x, y, x + cs - 3, y + cs - 3], fill=col)
        if used < cols * rows:
            r, c = divmod(used, cols)
            d.rectangle([x0 - 8, y0 + r * cs - 1, x0 + cols * cs + 8, y0 + r * cs + cs - 2], outline=GOLD + (255,), width=2)
        for yr in (20, 40, 60, 80):
            y = y0 + yr * cs
            d.text((x0 - 20, y - cs / 2), f"{yr}", font=font(F_MONO, 22), fill=(SOFT if a >= yr else DIM) + (255,), anchor="rm")
        done = a >= 80
        if not done:
            lines, tag = [("YOU GET ABOUT", WHITE), ("4,000 WEEKS", GOLD)], "EACH SQUARE IS ONE WEEK"
        else:
            lines, tag = [("THAT'S IT.", GOLD), ("4,160 WEEKS", WHITE)], "IF YOU LIVE TO 80"
        self.title(d, lines, tag)
        left = 4160 - used
        big(d, 1460, f"{left:,}", f"WEEKS LEFT AT AGE {int(a)}")
        return self.handle(img, t)


def build_weeks(model):
    T0, R = WeeksReel.T0, WeeksReel.RATE
    t = lambda age: T0 + age / R  # noqa: E731
    tend = t(80)
    cfg = mkcfg([0.4, t(10), t(20) + 0.3, t(40) + 0.3, t(65), tend + 1], 251)
    lines = [(0.2, "You only get about four thousand weeks. Watch yours disappear."),
             (T0 + 3.0, "Each square is a week, over an eighty year life."),
             (t(20) + 0.2, "Twenty years old? A quarter is already gone."),
             (t(40) + 0.2, "Forty, and half is gone."),
             (tend + 0.3, "That's all of it. About four thousand Saturdays."),
             (tend + 3.8, "What are you doing with this one?")]
    reel, buf = finish_reel(cfg, lines, model, WeeksReel, tend, 330)
    post = dict(title="VIRAL-1  You get about 4,000 weeks",
                caption="80 years x 52 weeks = 4,160 weeks. Each square is one week. How many are left at your age? Do the math: (80 - your age) x 52. Not a prediction, just a way to see the size of a life. Send this to someone who needs it.",
                tags=["perspective", "motivation", "time", "life", "mindset", "shipwithpip"],
                pinned="Your number: (80 minus your age) x 52 = weeks left, if you live to 80. Life expectancy varies, so treat 80 as a round number, not a promise.",
                notes="Pure arithmetic: 80 x 52 = 4,160 weeks (a year is really 52.18 weeks, so about 4,174). 'About four thousand weeks' is also the title of Oliver Burkeman's book 'Four Thousand Weeks' (2021); the idea is not original to this reel, mention the book if you want to credit it. 80 is an assumed lifespan, not a personal prediction. Tone is reflective; avoid implying anything about any viewer's health or age.")
    return reel, buf, cfg, post


# ============================================================== 2. million vs billion seconds
YEAR_S = 365.25 * 86400


def human(s):
    if s < 60:
        return f"{s:0.0f} sec"
    if s < 3600:
        return f"{s / 60:0.0f} min"
    if s < 86400:
        return f"{s / 3600:0.1f} hours"
    if s < YEAR_S:
        return f"{s / 86400:0.1f} days"
    return f"{s / YEAR_S:0.1f} years"


class SecondsReel(MinReel):
    T0, TD = 1.5, 16.0

    def frame(self, t):
        p = clamp((t - self.T0) / self.TD)
        s = 10 ** (9 * p)
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        x0, x1, y = 110, W - 110, 800
        f = lambda sec: x0 + (x1 - x0) * math.log10(max(sec, 1)) / 9  # noqa: E731
        d.rounded_rectangle([x0, y, x1, y + 20], radius=10, fill=(255, 255, 255, 25))
        d.rounded_rectangle([x0, y, max(x0 + 20, f(s)), y + 20], radius=10, fill=GOLD + (255,))
        marks = [(60, "1 MIN"), (3600, "1 HR"), (86400, "1 DAY"), (1e6, "1 MILLION"), (YEAR_S, "1 YEAR"), (1e9, "1 BILLION")]
        for i, (sec, lab) in enumerate(marks):
            lit = s >= sec
            yy = y + 54 + (i % 2) * 44
            d.line([f(sec), y + 22, f(sec), y + 40], fill=(255, 255, 255, 140), width=2)
            d.text((f(sec), yy + 10), lab, font=font(F_MONO, 24), fill=(WHITE if lit else DIM) + (255,), anchor="mm")
        reveal = t > self.T0 + self.TD + 0.8
        if not reveal:
            lines, tag = [("1 MILLION SECONDS", WHITE), ("vs 1 BILLION", GOLD)], "HOW BIG IS THE GAP?"
            big(d, 1180, f"{s:,.0f}", "SECONDS")
            d.text((W / 2, 1330), human(s), font=font(F_BOLD, 80), fill=GOLD + (255,), anchor="mm")
        else:
            lines, tag = [("11 DAYS", ICE), ("vs 32 YEARS", RED)], "TO SCALE"
            k = clamp((t - self.T0 - self.TD - 0.8) / 2.5)
            bx0, bx1 = 110, W - 110
            d.text((bx0, 990), "1 MILLION SECONDS  =  11.6 DAYS", font=font(F_SEMI, 34), fill=ICE + (255,), anchor="lm")
            d.rounded_rectangle([bx0, 1020, bx1, 1070], radius=12, fill=(255, 255, 255, 25))
            d.rectangle([bx0, 1020, bx0 + max(3, (bx1 - bx0) * (11.574 / 11574) * k), 1070], fill=ICE + (255,))
            d.text((bx0, 1150), "1 BILLION SECONDS  =  31.7 YEARS", font=font(F_SEMI, 34), fill=RED + (255,), anchor="lm")
            d.rounded_rectangle([bx0, 1180, bx1, 1230], radius=12, fill=(255, 255, 255, 25))
            d.rounded_rectangle([bx0, 1180, bx0 + (bx1 - bx0) * k, 1230], radius=12, fill=RED + (255,))
            d.text((W / 2, 1360), "1,000 TIMES BIGGER", font=font(F_BOLD, 70), fill=GOLD + (int(255 * k),), anchor="mm")
        self.title(d, lines, tag)
        return self.handle(img, t)


def build_seconds(model):
    T0, TD = SecondsReel.T0, SecondsReel.TD
    tr = T0 + TD + 0.8
    cfg = mkcfg([0.4, T0 + 5, T0 + 11, T0 + 15.5, tr + 3, tr + 5], 252)
    lines = [(0.2, "A million seconds is eleven days. A billion seconds is thirty two years."),
             (T0 + 5.8, "Watch the counter. Each tick of this bar is ten times more."),
             (T0 + 12.5, "One million is about a week and a half. One billion is a whole life stage."),
             (tr, "To scale, a million is a sliver. A billion is a thousand times bigger."),
             (tr + 5.4, "Our brains can't feel the difference. Your bank account can.")]
    reel, buf = finish_reel(cfg, lines, model, SecondsReel, tr + 0.5, 392)
    post = dict(title="VIRAL-2  A million seconds vs a billion seconds",
                caption="1,000,000 seconds = 11.6 days. 1,000,000,000 seconds = 31.7 years. A billion is exactly 1,000 times a million, but our brains treat them as 'both really big'. Share with the friend who says 'it's just a few zeros.'",
                tags=["math", "perspective", "didyouknow", "numbers", "science", "shipwithpip"],
                pinned="1 million seconds = 11.6 days. 1 billion seconds = 31.7 years. 1 trillion seconds = about 31,700 years.",
                notes=f"1e6 s / 86,400 = 11.57 days. 1e9 s / 31,557,600 (Julian year) = 31.69 years (31.71 using 365-day years). 1e12 s = 31,688 years. 1,000x ratio is exact. The to-scale bar compares 11.57 days to 11,574 days. The 'bank account' line is a loose rhetorical remark, not a claim. Nothing here needs outside sources.")
    return reel, buf, cfg, post


# ============================================================== 3. 1% a day
class OnePctReel(MinReel):
    T0, TD = 2.5, 16.0

    def frame(self, t):
        day = clamp((t - self.T0) / self.TD) * 365
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        gx0, gx1, gy0, gy1 = 110, 970, 1260, 560  # gy0 bottom (log -1.7) gy1 top (+1.7)
        gx = lambda dd: gx0 + (gx1 - gx0) * dd / 365  # noqa: E731
        gy = lambda v: gy0 - (gy0 - gy1) * v / 40  # noqa: E731
        d.line([gx0, gy0, gx1, gy0], fill=(255, 255, 255, 80), width=2)
        for v in (1, 10, 20, 30, 40):
            d.line([gx0, gy(v), gx1, gy(v)], fill=(255, 255, 255, 22), width=2)
            d.text((gx0 - 10, gy(v)), f"{v}x", font=font(F_MONO, 24), fill=SOFT + (255,), anchor="rm")
        n = 120
        up = [(gx(day * i / n), gy(1.01 ** (day * i / n))) for i in range(n + 1)]
        dn = [(gx(day * i / n), gy(0.99 ** (day * i / n))) for i in range(n + 1)]
        if day > 1:
            d.line(up, fill=GREEN + (255,), width=7, joint="curve")
            d.line(dn, fill=RED + (255,), width=7, joint="curve")
        for pts, col in ((up, GREEN), (dn, RED)):
            d.ellipse([pts[-1][0] - 11, pts[-1][1] - 11, pts[-1][0] + 11, pts[-1][1] + 11], fill=col + (255,))
        for dd in (0, 100, 200, 300):
            d.text((gx(dd), gy0 + 40), f"day {dd}", font=font(F_MONO, 22), fill=SOFT + (230,), anchor="mm")
        u, w = 1.01 ** day, 0.99 ** day
        done = t > self.T0 + self.TD
        if not done:
            lines, tag = [("GET 1% BETTER", GREEN), ("EVERY DAY", WHITE)], "OR 1% WORSE"
        else:
            lines, tag = [("37X BETTER", GREEN), ("OR 97% GONE", RED)], "IN ONE YEAR"
        self.title(d, lines, tag)
        d.text((W / 2 - 230, 1400), f"{u:0.1f}x", font=font(F_BOLD, 110), fill=GREEN + (255,), anchor="mm")
        d.text((W / 2 + 230, 1400), f"{w:0.2f}x", font=font(F_BOLD, 110), fill=RED + (255,), anchor="mm")
        d.text((W / 2, 1490), f"DAY {int(day)}", font=font(F_MONO, 30), fill=SOFT + (255,), anchor="mm")
        return self.handle(img, t)


def build_onepct(model):
    T0, TD = OnePctReel.T0, OnePctReel.TD
    tend = T0 + TD
    cfg = mkcfg([0.4, T0 + 4, T0 + 9, T0 + 14, tend + 0.5, tend + 3], 253)
    lines = [(0.2, "Get one percent better every day, and in a year you're thirty seven times better."),
             (T0 + 1.5, "Green is one percent better each day. Red is one percent worse."),
             (T0 + 8.5, "The curves barely move for months. Then they take off."),
             (tend + 0.2, "Day three sixty five. Thirty seven times better, or ninety seven percent gone."),
             (tend + 5.8, "It's a metaphor, not a measurement. But small daily choices compound.")]
    reel, buf = finish_reel(cfg, lines, model, OnePctReel, tend, 392)
    post = dict(title="VIRAL-3  1% a day = 37x in a year",
                caption="1.01^365 = 37.8. 0.99^365 = 0.026. Improving 1% a day compounds to about 37.8x over a year, and declining 1% a day leaves you at about 2.6% (97% gone). It's a metaphor about compounding, not a literal measure of human skill. Which side are you on?",
                tags=["mindset", "habits", "compoundinterest", "motivation", "math", "shipwithpip"],
                pinned="The math: 1.01^365 = 37.78. 0.99^365 = 0.0255. Small daily changes compound in both directions.",
                notes="1.01^365 = 37.783; 0.99^365 = 0.02551. Chart is linear scale (0-40x). The '1% better every day' idea is popular in self-improvement writing (e.g. James Clear, 'Atomic Habits' discusses 1.01^365 = 37.78); skills do not literally improve 1% per day forever, and the reel says so. Credit the book if you mention it. Nothing else needs verification.")
    return reel, buf, cfg, post


BUILDERS = {"weeks": build_weeks, "seconds": build_seconds, "onepct": build_onepct}


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
