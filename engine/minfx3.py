"""Five more minimal dark save-worthy reels: ice | eggs | fridge | phone | dv.
usage: python minfx3.py NAME out.mp4 --model <kokoro dir> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa: E402
from minfx import DIM, GOLD, ICE, SOFT, WHITE, MinReel  # noqa: E402
from minfx2 import GREEN, RED, big, card, finish_reel, mkcfg  # noqa: E402
from scenes import render  # noqa: E402
from simfx import F_BOLD, F_MONO, F_SEMI, W, clamp, font  # noqa: E402


def lerp(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * clamp(k)) for i in range(3))


# ============================================================== 1. ice thickness
class IceReel(MinReel):
    T0, TD = 1.5, 27.0
    CM = 30

    def cm(self, t):
        return clamp((t - self.T0) / self.TD) * self.CM

    def frame(self, t):
        c = self.cm(t)
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        sy, px = 1060, 26.0  # water surface y (top of ice at sy - c*px?) ice grows downward from surface line
        top = 800
        # water body
        d.rectangle([0, top, W, 1330], fill=(18, 50, 96, 255))
        # ice slab hangs from the top line, thickness c cm at 26 px/cm (scaled 20 px/cm)
        s = 16.0
        d.rectangle([0, top, W, top + c * s], fill=(70, 130, 180, 215))
        d.line([0, top, W, top], fill=WHITE + (255,), width=3)
        d.line([0, top + c * s, W, top + c * s], fill=ICE + (255,), width=3)
        # ruler on left
        for k in range(0, 31, 5):
            y = top + k * s
            d.line([40, y, 80, y], fill=WHITE + (255,), width=3)
            d.text((100, y), f"{k} cm", font=font(F_MONO, 26), fill=WHITE + (230,), anchor="lm")
        # thresholds
        zones = [(15, "WALK / SKATE", "person"), (20, "GROUP / HOCKEY", "group"), (25, "SNOWMOBILE", "sled")]
        for th, lab, kind in zones:
            y = top + th * s
            d.line([160, y, W - 60, y], fill=(255, 255, 255, 60), width=2)
            on = c >= th
            a = 255 if on else 90
            x = 640
            d.text((W - 60, y - 22), f"{th} cm  {lab}", font=font(F_SEMI, 30), fill=(GREEN if on else SOFT) + (a,), anchor="rm")
            gy = top - 6
        # objects on the surface, appear when safe
        if c >= 15:
            d.ellipse([300, top - 60, 330, top - 30], fill=WHITE + (255,))
            d.line([315, top - 30, 315, top], fill=WHITE + (255,), width=8)
        if c >= 20:
            for x in (420, 470, 520):
                d.ellipse([x - 12, top - 54, x + 12, top - 30], fill=WHITE + (255,))
                d.line([x, top - 30, x, top], fill=WHITE + (255,), width=7)
        if c >= 25:
            d.rounded_rectangle([640, top - 50, 800, top], radius=12, fill=GOLD + (255,))
            d.ellipse([660, top - 16, 690, top + 4], fill=(30, 30, 40, 255))
            d.ellipse([750, top - 16, 780, top + 4], fill=(30, 30, 40, 255))
        done = t > self.T0 + self.TD
        verdict = ("STAY OFF" if c < 15 else ("OK FOR ONE PERSON" if c < 20 else ("OK FOR A GROUP" if c < 25 else "OK FOR A SNOWMOBILE")))
        vcol = RED if c < 15 else GREEN
        if not done:
            lines, tag = [("HOW THICK MUST", WHITE), ("THE ICE BE?", GOLD)], "BEFORE YOU STEP ON IT"
        else:
            lines, tag = [("15 - 20 - 25", GOLD), ("CENTIMETRES", WHITE)], "SAVE THIS"
        self.title(d, lines, tag)
        big(d, 1480, f"{c:0.0f} cm", verdict, vcol)
        if done:
            card(d, 400, 500, int(255 * clamp((t - self.T0 - self.TD) / 0.6)), ["BLUE = STRONG.  WHITE = HALF.  GREY = NO."], 34)
        return self.handle(img, t)


def build_ice(model):
    T0, TD = IceReel.T0, IceReel.TD
    t = lambda cm: T0 + TD * cm / IceReel.CM  # noqa: E731
    tend = T0 + TD
    cfg = mkcfg([0.4, t(8), t(16), t(21), t(26), tend + 1], 231)
    lines = [(0.2, "How thick does ice have to be before you step on it?"),
             (T0 + 0.8, "Watch it grow. Under fifteen centimeters, stay off."),
             (t(15) + 0.2, "Fifteen centimeters, about six inches: one person walking or skating."),
             (t(20) + 0.2, "Twenty: a group, or a hockey game."),
             (t(25) + 0.2, "Twenty five: a snowmobile or ATV."),
             (tend + 0.4, "And check the color. Clear blue ice is strongest. White ice is about half as strong. Grey ice means water, stay off."),
             (tend + 10.5, "Save this before the first freeze.")]
    reel, buf = finish_reel(cfg, lines, model, IceReel, t(25), 330)
    post = dict(title="SAVE-8  How thick must ice be?",
                caption="Minimum ice thickness guidance used by Canadian safety organisations: 15 cm (6 in) for walking or skating, 20 cm (8 in) for a group or a hockey game, 25 cm (10 in) for snowmobiles and ATVs. Colour matters too: clear blue ice is strongest, white or opaque ice is about half as strong, and grey ice is unsafe because it has water in it. Ice thickness can change from one spot to the next, so check local advisories and never go alone. Save this before the first freeze.",
                tags=["icesafety", "winter", "canada", "safety", "didyouknow", "shipwithpip"],
                pinned="Bookmark: 15 cm walk, 20 cm group, 25 cm snowmobile. Clear blue strong, white half, grey no. When in doubt, stay off.",
                notes="Thicknesses 15/20/25 cm and the colour guidance (clear blue strongest, white/opaque about half, grey unsafe) appear in the Canadian Red Cross ice safety tips PDF (Manitoba, 2014) and the Government of Nunavut 'Ice Safety' page, both read this session. The Red Cross PDF phrases 15 cm for small groups; Nunavut says walking or skating alone. Ice strength varies by location, currents, snow cover and temperature history, so these are minimums for new clear ice, not guarantees. Check your local authority.")
    return reel, buf, cfg, post


# ============================================================== 2. boil an egg
EGG_STAGES = [(6, "RUNNY"), (7, "JAMMY"), (9, "SOFT SET"), (12, "HARD")]


class EggReel(MinReel):
    T0, TD = 1.5, 27.0
    MAXM = 12.5

    def mins(self, t):
        return clamp((t - self.T0) / self.TD) * self.MAXM

    def frame(self, t):
        m = self.mins(t)
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        cx, cy, R = W // 2, 860, 300
        # white sets from outside by ~3 min; yolk goes from liquid to firm across 6..12
        white = lerp((190, 200, 225), (250, 250, 252), (m - 1) / 3)
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=white + (255,), outline=(255, 255, 255, 80), width=3)
        yr = 150
        ycol = lerp((255, 160, 30), (255, 196, 70), (m - 5) / 3) if m < 8 else lerp((255, 196, 70), (244, 222, 120), (m - 8) / 4)
        d.ellipse([cx - yr, cy - yr, cx + yr, cy + yr], fill=ycol + (255,))
        if m < 8:
            d.ellipse([cx - 70, cy - 90, cx - 20, cy - 60], fill=(255, 255, 255, 90))  # liquid shine
        # stage label
        name = "RAW" if m < 5.6 else next((n for tm, n in EGG_STAGES if m <= tm + 0.4), "HARD")
        if m < 3:
            name = "COOKING"
        stage = (name, GOLD)
        # timeline
        x0, x1, y = 140, W - 140, 1220
        f = lambda mm: x0 + (x1 - x0) * mm / 12.5  # noqa: E731
        d.rounded_rectangle([x0, y, x1, y + 12], radius=6, fill=(255, 255, 255, 30))
        d.rounded_rectangle([x0, y, f(m), y + 12], radius=6, fill=GOLD + (255,))
        for tm, n in EGG_STAGES:
            d.line([f(tm), y - 10, f(tm), y + 22], fill=(255, 255, 255, 140), width=2)
            d.text((f(tm), y + 52), f"{tm}", font=font(F_MONO, 26), fill=WHITE + (255,), anchor="mm")
            d.text((f(tm), y - 34), n, font=font(F_MONO, 20), fill=SOFT + (255,), anchor="mm")
        done = t > self.T0 + self.TD
        if not done:
            lines, tag = [("THE PERFECT EGG", WHITE), ("MINUTE BY MINUTE", GOLD)], "LARGE EGG, BOILING WATER"
        else:
            lines, tag = [("6  7  9  12", GOLD), ("MINUTES", WHITE)], "SAVE THIS"
        self.title(d, lines, tag)
        big(d, 1440, f"{int(m)}:{int((m % 1) * 60):02d}", name)
        if done:
            card(d, 400, 500, int(255 * clamp((t - self.T0 - self.TD) / 0.6)), ["THEN COLD WATER. IT STOPS THE COOKING."], 34)
        return self.handle(img, t)


def build_eggs(model):
    T0, TD = EggReel.T0, EggReel.TD
    t = lambda mm: T0 + TD * mm / EggReel.MAXM  # noqa: E731
    tend = T0 + TD
    cfg = mkcfg([0.4, t(3), t(6.2), t(7.2), t(9.2), tend + 1], 232)
    lines = [(0.2, "Stop guessing how long to boil an egg."),
             (T0 + 0.8, "Drop it in boiling water and start the clock. This is the inside of the egg."),
             (t(6) - 0.4, "Six minutes: runny yolk."),
             (t(7) - 0.2, "Seven: jammy."),
             (t(9) - 0.2, "Nine: soft but set."),
             (t(12) - 0.4, "Twelve: fully hard."),
             (tend + 0.4, "Move it to cold water the second it's done, so it doesn't keep cooking."),
             (tend + 5.4, "Save this.")]
    reel, buf = finish_reel(cfg, lines, model, EggReel, t(12), 392)
    post = dict(title="SAVE-9  Boil an egg: 6, 7, 9, 12",
                caption="A rule of thumb for large eggs from the fridge, dropped into boiling water: 6 minutes runny yolk, 7 jammy, 9 soft but set, 12 fully hard. Put them in cold water right after so they stop cooking. Times shift with egg size, starting temperature, altitude and pot size, so treat these as a starting point and adjust to taste. Save this one.",
                tags=["cooking", "eggs", "kitchenhacks", "food", "didyouknow", "shipwithpip"],
                pinned="Cheat sheet (large eggs, boiling water): 6 runny, 7 jammy, 9 soft-set, 12 hard. Ice bath after.",
                notes="Times are a widely used rule of thumb, not a measured result; the yolk colours and cross-section are my stylised illustration, not photographs or lab data. Results vary with egg size, fridge vs room temperature, altitude, and how many eggs are in the pot. I did not verify these against a published source this session. Food safety: fully cooked eggs are the safe option for pregnant people, young children and older adults.")
    return reel, buf, cfg, post


# ============================================================== 3. how long food lasts in the fridge
FRIDGE = [("RAW GROUND MEAT", 1, 2), ("RAW CHICKEN", 1, 2), ("COOKED LEFTOVERS", 3, 4), ("OPENED DELI MEAT", 3, 5), ("HARD-BOILED EGGS", 7, 7), ("RAW EGGS IN SHELL", 21, 35)]


class FridgeReel(MinReel):
    T0, TD = 1.5, 30.0
    DMAX = 35

    def day(self, t):
        return clamp((t - self.T0) / self.TD) * self.DMAX

    def frame(self, t):
        dday = self.day(t)
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        x0, x1 = 90, W - 90
        f = lambda dd: x0 + (x1 - x0) * dd / self.DMAX  # noqa: E731
        y = 640
        for name, lo, hi in FRIDGE:
            d.text((x0, y - 36), name, font=font(F_SEMI, 32), fill=WHITE + (255,), anchor="lm")
            d.rounded_rectangle([x0, y, x1, y + 22], radius=11, fill=(255, 255, 255, 22))
            end = min(dday, hi)
            if end > 0:
                col = GREEN if dday <= lo else (GOLD if dday <= hi else RED)
                d.rounded_rectangle([x0, y, max(x0 + 22, f(end)), y + 22], radius=11, fill=col + (255,))
            if dday > hi:
                d.rounded_rectangle([f(hi), y, f(min(dday, self.DMAX)), y + 22], radius=11, fill=RED + (140,))
            lab = f"{lo}-{hi} days" if lo != hi else f"{lo} days"
            if hi >= 21:
                lab = "3-5 weeks"
            d.text((x1, y - 36), lab, font=font(F_MONO, 28), fill=GOLD + (255,), anchor="rm")
            y += 110
        done = t > self.T0 + self.TD
        if not done:
            lines, tag = [("FOOD LIFE", WHITE), ("IN YOUR FRIDGE", GOLD)], "AT 4 C (40 F) OR COLDER"
        else:
            lines, tag = [("SAVE THIS", GOLD), ("FOR YOUR FRIDGE", WHITE)], "GREEN SAFE. RED TOSS IT."
        self.title(d, lines, tag)
        big(d, 1440, f"DAY {int(dday)}", "SINCE YOU BOUGHT OR COOKED IT")
        return self.handle(img, t)


def build_fridge(model):
    T0, TD = FridgeReel.T0, FridgeReel.TD
    t = lambda dd: T0 + TD * dd / FridgeReel.DMAX  # noqa: E731
    tend = T0 + TD
    cfg = mkcfg([0.4, t(1.2), t(3.5), t(6), t(8), tend + 1], 233)
    lines = [(0.2, "How long does food really last in your fridge?"),
             (T0 + 0.8, "Watch the days go by. Green is fine, red means toss it."),
             (t(2) + 0.2, "Raw ground meat and raw chicken: one to two days."),
             (t(4) + 0.4, "Cooked leftovers: three to four days."),
             (t(7) + 0.4, "Hard boiled eggs: one week."),
             (t(21) + 0.2, "Raw eggs in the shell last three to five weeks."),
             (tend + 0.4, "Colder than four degrees. And when in doubt, throw it out. Save this.")]
    reel, buf = finish_reel(cfg, lines, model, FridgeReel, t(35) - 2, 330)
    post = dict(title="SAVE-10  How long food lasts in the fridge",
                caption="Fridge storage times (at 4 C / 40 F or colder), based on US government food-safety guidance: raw ground meat and raw poultry 1-2 days, cooked leftovers 3-4 days, opened deli meat 3-5 days, hard-boiled eggs 1 week, raw eggs in the shell 3-5 weeks. Smell and look aren't reliable tests for the bacteria that make you sick. Save this and stick it on your fridge.",
                tags=["foodsafety", "kitchen", "fridge", "mealprep", "didyouknow", "shipwithpip"],
                pinned="Fridge cheat sheet: ground meat/chicken 1-2 days. Leftovers 3-4. Deli meat (opened) 3-5. Hard-boiled eggs 7. Raw eggs 3-5 weeks. Keep the fridge at 4 C or colder.",
                notes="Numbers are from my memory of the USDA / FSIS refrigerator storage chart ('Cold Food Storage Chart' and the FoodKeeper app) and match Health Canada's general guidance on leftovers (2-3 days is sometimes quoted in Canada: Health Canada says to eat leftovers within 2 days for some foods; confirm the Canadian number if posting to a Canadian audience). My fetch of the USDA page was blocked this session, so these are NOT verified here: check fsis.usda.gov and canada.ca/foodsafety before posting. Canada's own figures can be shorter than the US ones.")
    return reel, buf, cfg, post


# ============================================================== 4. phone days per year
class PhoneReel(MinReel):
    T0, TD = 1.5, 22.0
    HRS = 3

    def frame(self, t):
        p = clamp((t - self.T0) / self.TD)
        hours = self.HRS * 365 * p
        days = hours / 24
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        cols, rows = 20, 19
        cw = 40
        x0 = (W - cols * cw) // 2
        y0 = 570
        for i in range(365):
            r, c = divmod(i, cols)
            x, y = x0 + c * cw, y0 + r * cw
            passed = i < p * 365
            gone = i < int(days)
            col = (RED if gone else ((70, 80, 110) if passed else (36, 42, 66)))
            d.rounded_rectangle([x + 4, y + 4, x + cw - 4, y + cw - 4], radius=8, fill=col + (255,))
        done = t > self.T0 + self.TD
        if not done:
            lines, tag = [("3 HOURS A DAY", WHITE), ("ON YOUR PHONE", GOLD)], "EACH SQUARE IS ONE DAY OF THE YEAR"
        else:
            lines, tag = [("46 FULL DAYS", GOLD), ("EVERY YEAR", WHITE)], "SAVE THIS"
        self.title(d, lines, tag)
        big(d, 1480, f"{days:0.0f} DAYS", f"{hours:,.0f} HOURS ON THE SCREEN")
        if done:
            card(d, 395, 535, int(255 * clamp((t - self.T0 - self.TD) / 0.6)), ["1 HR = 15 DAYS    2 HR = 30 DAYS", "4 HR = 61 DAYS    5 HR = 76 DAYS"], 34)
        return self.handle(img, t)


def build_phone(model):
    T0, TD = PhoneReel.T0, PhoneReel.TD
    tend = T0 + TD
    cfg = mkcfg([0.4, T0 + 5, T0 + 11, T0 + 17, tend + 0.3, tend + 3], 234)
    hrs = PhoneReel.HRS * 365
    lines = [(0.2, "Three hours a day on your phone sounds small. Watch this."),
             (T0 + 1.0, "Each square is a day of the year. Red means a full day gone to the screen."),
             (T0 + 9.6, "It adds up quietly, three hours at a time."),
             (tend + 0.2, f"{hrs:,} hours. That's forty six full days, around one and a half months."),
             (tend + 6.2, "One hour a day is about fifteen days. Five hours is seventy six. Save this.")]
    reel, buf = finish_reel(cfg, lines, model, PhoneReel, tend, 330)
    post = dict(title="SAVE-11  3 hours a day = 46 days a year",
                caption=f"Math, not a judgement: 3 hours a day x 365 days = {hrs:,} hours = {hrs / 24:0.1f} full days a year. 1 hour a day is {365 / 24:0.1f} days a year, 2 hours is {730 / 24:0.1f}, 4 hours is {1460 / 24:0.1f}, 5 hours is {1825 / 24:0.1f}. Check your own screen-time number and do the math. Save this.",
                tags=["screentime", "productivity", "phone", "habits", "didyouknow", "shipwithpip"],
                pinned="Your number: hours per day x 365 / 24 = days per year.",
                notes=f"Pure arithmetic: hours/day x 365 / 24. 3h -> {hrs / 24:.2f} d; 1h -> {365 / 24:.2f}; 2h -> {730 / 24:.2f}; 4h -> {1460 / 24:.2f}; 5h -> {1825 / 24:.2f}. The reel does not claim 3 hours is anyone's average. 'Full days' means 24-hour days, not waking days (which would be ~ 3 hours / 16 waking hours = 68 waking days). Pick your own number.")
    return reel, buf, cfg, post


# ============================================================== 5. %DV on food labels
DV_EX = [("SODIUM", 24, "LIMIT THIS"), ("FIBRE", 22, "GET MORE"), ("IRON", 3, ""), ("SATURATED FAT", 3, "")]


class DvReel(MinReel):
    T0, TD = 1.5, 26.0

    def frame(self, t):
        p = clamp((t - self.T0) / self.TD)
        v = p * 30
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        x0, x1, y = 120, W - 120, 800
        f = lambda dv: x0 + (x1 - x0) * dv / 30  # noqa: E731
        d.rounded_rectangle([x0, y, x1, y + 70], radius=20, fill=(255, 255, 255, 22))
        d.rounded_rectangle([x0, y, f(5), y + 70], radius=20, fill=ICE + (150,))
        d.rectangle([f(15), y, x1, y + 70], fill=GOLD + (150,))
        d.text(((x0 + f(5)) / 2, y - 36), "A LITTLE", font=font(F_SEMI, 32), fill=ICE + (255,), anchor="mm")
        d.text(((f(15) + x1) / 2, y - 36), "A LOT", font=font(F_SEMI, 32), fill=GOLD + (255,), anchor="mm")
        for dv in (0, 5, 15, 30):
            d.line([f(dv), y + 70, f(dv), y + 96], fill=WHITE + (200,), width=3)
            d.text((f(dv), y + 128), f"{dv}%", font=font(F_MONO, 30), fill=WHITE + (255,), anchor="mm")
        d.ellipse([f(v) - 18, y + 26, f(v) + 18, y + 62], fill=WHITE + (255,))
        # example label rows
        ty = 1080
        d.text((x0, ty - 60), "EXAMPLE LABEL (MADE UP)", font=font(F_MONO, 24), fill=SOFT + (255,), anchor="lm")
        for name, dv, note in DV_EX:
            lit = v >= dv
            tag = "a little" if dv <= 5 else ("a lot" if dv >= 15 else "")
            col = ICE if dv <= 5 else GOLD
            d.text((x0, ty), f"{name}", font=font(F_SEMI, 34), fill=(WHITE if lit else DIM) + (255,), anchor="lm")
            d.text((x1, ty), f"{dv}% DV" + (f"   {tag}" if lit else ""), font=font(F_MONO, 30), fill=(col if lit else DIM) + (255,), anchor="rm")
            ty += 64
        done = t > self.T0 + self.TD
        if not done:
            lines, tag = [("DECODE ANY", WHITE), ("FOOD LABEL", GOLD)], "IN 5 SECONDS"
        else:
            lines, tag = [("5% = A LITTLE", ICE), ("15% = A LOT", GOLD)], "SAVE THIS"
        self.title(d, lines, tag)
        big(d, 1480, f"{v:0.0f}%", "% DAILY VALUE")
        if done:
            card(d, 400, 500, int(255 * clamp((t - self.T0 - self.TD) / 0.6)), ["WANT LESS: SODIUM, SAT. FAT.  MORE: FIBRE."], 30)
        return self.handle(img, t)


def build_dv(model):
    T0, TD = DvReel.T0, DvReel.TD
    t = lambda dv: T0 + TD * dv / 30  # noqa: E731
    tend = T0 + TD
    cfg = mkcfg([0.4, t(3), t(10), t(20), t(26), tend + 1], 235)
    lines = [(0.2, "Here's how to read any food label in five seconds."),
             (T0 + 0.8, "Ignore the grams. Look at the percent daily value."),
             (t(5) - 0.4, "Five percent or less is a little."),
             (t(15) - 0.6, "Fifteen percent or more is a lot."),
             (tend + 0.3, "Want less of it, like sodium and saturated fat? Pick the low number."),
             (tend + 5.2, "Want more, like fibre? Pick the high one. Save this.")]
    reel, buf = finish_reel(cfg, lines, model, DvReel, t(15), 392)
    post = dict(title="SAVE-12  Read a food label in 5 seconds",
                caption="Health Canada's quick rule for the % Daily Value on a nutrition label: 5% or less is a little, 15% or more is a lot. Look for a low % of sodium and saturated fat, and a high % of nutrients like fibre, calcium and iron. The label example in the video is made up. Save this for your next grocery run.",
                tags=["nutrition", "groceryshopping", "health", "foodlabels", "didyouknow", "shipwithpip"],
                pinned="5% or less = a little. 15% or more = a lot. Low % for sodium and sat fat, high % for fibre.",
                notes="The 5% / 15% rule is Health Canada's 'Percent Daily Value' guidance (also used by the US FDA). My fetch of the Canada.ca page was blocked this session, so this is from memory and NOT verified here: check canada.ca (search 'percent daily value Health Canada') before posting. Example label numbers in the video are invented for illustration. Not personalised dietary advice.")
    return reel, buf, cfg, post


BUILDERS = {"ice": build_ice, "eggs": build_eggs, "fridge": build_fridge, "phone": build_phone, "dv": build_dv}


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
