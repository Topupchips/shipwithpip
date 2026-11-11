"""Five more minimal dark save-worthy fact reels: stopping | bacteria | cpr | compound | coldwater.
usage: python minfx2.py NAME out.mp4 --model <kokoro dir> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa: E402
from minfx import DIM, GOLD, ICE, SOFT, WHITE, MinReel, seq_audio  # noqa: E402
from scenes import chime, render  # noqa: E402
from simfx import F_BOLD, F_MONO, F_SEMI, H, W, clamp, font, tone  # noqa: E402

RED = (255, 96, 110)
GREEN = (110, 235, 170)


def card(d, y0, y1, a, rows, size=40):
    d.rounded_rectangle([90, y0, W - 90, y1], radius=30, fill=(10, 14, 30, int(a * 0.92)), outline=GOLD + (a,), width=3)
    step = (y1 - y0) / (len(rows) + 0.0)
    for i, r in enumerate(rows):
        d.text((W / 2, y0 + step * (i + 0.5)), r, font=font(F_SEMI, size), fill=WHITE + (a,), anchor="mm")


def big(d, y, txt, sub, col=WHITE):
    d.text((W / 2, y), txt, font=font(F_BOLD, 150), fill=col + (255,), anchor="mm")
    d.text((W / 2, y + 90), sub, font=font(F_MONO, 28), fill=SOFT + (255,), anchor="mm")


def finish_reel(cfg, lines, model, cls, chime_t, chime_base=392, extra_audio=None):
    buf, vo = seq_audio(cfg, lines, model)
    if extra_audio:
        extra_audio(buf)
    chime(buf, chime_t, chime_base)
    cfg["T"] = max(vo[-1][1] + 2.4, chime_t + 3)
    return cls(cfg, vo), buf


def mkcfg(stills, seed):
    return dict(T=60, c1=GOLD, c2=ICE, payoff=None, seed=seed, still_times=stills, header=lambda t: ([], ""))


# ============================================================== 1. stopping distance
MU, G, RT = 0.7, 9.81, 1.5
A_DEC = MU * G


def stop_pos(v, t):
    """position (m) at time t after the hazard appears; reaction RT then constant braking"""
    if t <= RT:
        return v * t
    tb = min(t - RT, v / A_DEC)
    return v * RT + v * tb - 0.5 * A_DEC * tb * tb


class StopReel(MinReel):
    T0, SLOW = 2.0, 0.5
    SPEEDS = (50 / 3.6, 100 / 3.6)

    def frame(self, t):
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        tt = max(0.0, (t - self.T0) * self.SLOW)
        ppm = 8.4  # px per metre
        x0 = 90
        done = tt > 7.5
        if not done:
            lines, tag = [("DOUBLE THE SPEED", WHITE), ("4X THE BRAKING", GOLD)], "REAL PHYSICS, DRY ROAD"
        else:
            lines, tag = [("35 M  VS  98 M", GOLD), ("TO STOP", WHITE)], "SAVE THIS"
        self.title(d, lines, tag)
        for i, (v, name, y) in enumerate(zip(self.SPEEDS, ("50 km/h", "100 km/h"), (760, 1010))):
            d.line([x0, y + 38, x0 + 100 * ppm, y + 38], fill=(255, 255, 255, 40), width=2)
            for m in range(0, 101, 20):
                d.line([x0 + m * ppm, y + 38, x0 + m * ppm, y + 52], fill=(255, 255, 255, 90), width=2)
                d.text((x0 + m * ppm, y + 78), f"{m}m", font=font(F_MONO, 22), fill=DIM + (255,), anchor="mm")
            d.text((x0, y - 70), name, font=font(F_BOLD, 52), fill=(GOLD if i else ICE) + (255,), anchor="lm")
            # hazard line
            pos = stop_pos(v, tt)
            front = x0 + pos * ppm
            d.line([x0, y - 20, x0, y + 38], fill=RED + (255,), width=4)
            # reaction segment
            rx = x0 + v * RT * ppm
            d.rectangle([x0, y + 20, min(front, rx), y + 30], fill=(255, 214, 130, 140))
            if tt > RT:
                d.rectangle([rx, y + 20, front, y + 30], fill=RED + (200,))
            braking = RT < tt < RT + v / A_DEC
            car = [front - 4.5 * ppm, y - 14, front, y + 26]
            d.rounded_rectangle(car, radius=10, fill=(235, 240, 255, 255))
            if braking:
                d.rectangle([car[0] - 4, y - 8, car[0], y + 20], fill=RED + (255,))
            if tt >= RT + v / A_DEC:
                d.text((front + 18, y + 6), f"{pos:0.0f} m", font=font(F_BOLD, 44), fill=(GOLD if i else ICE) + (255,), anchor="lm")
        d.text((x0, 1190), "GOLD: you haven't reacted yet (1.5 s).  RED: brakes on.", font=font(F_MONO, 24), fill=SOFT + (255,), anchor="lm")
        big(d, 1380, f"{tt:0.1f}s", "SINCE THE HAZARD APPEARED (SHOWN AT HALF SPEED)")
        if done:
            card(d, 1215, 1320, int(255 * clamp((tt - 7.5) / 0.6)), ["BRAKING DISTANCE GOES WITH SPEED SQUARED"], 30)
        return self.handle(img, t)


def build_stopping(model):
    v1, v2 = StopReel.SPEEDS
    b1, b2 = v1 ** 2 / (2 * A_DEC), v2 ** 2 / (2 * A_DEC)
    s1, s2 = v1 * RT + b1, v2 * RT + b2
    T0, SL = StopReel.T0, StopReel.SLOW
    tdone = T0 + 7.5 / SL
    cfg = mkcfg([0.4, T0 + 3, T0 + 8, T0 + 14, tdone + 0.5, tdone + 3], 221)
    lines = [(0.2, "Going twice as fast doesn't double your stopping distance."),
             (T0 + 0.2, "Two cars, same road. One at fifty, one at a hundred. A hazard appears now."),
             (T0 + 3.4, "First, a second and a half to react. The faster car already covers more ground."),
             (T0 + 8.0, f"Then the brakes. Fifty needs {b1:0.0f} meters. A hundred needs {b2:0.0f}. That's four times."),
             (tdone + 0.3, f"All in, {s1:0.0f} meters versus {s2:0.0f}."),
             (tdone + 5.6, "Braking distance grows with speed squared. Save this.")]
    reel, buf = finish_reel(cfg, lines, model, StopReel, tdone, 392)
    post = dict(title="SAVE-3  Double the speed, 4x the braking",
                caption=f"On a dry road, a car at 50 km/h needs about {b1:0.0f} m to brake to a stop; at 100 km/h it needs about {b2:0.0f} m, four times as much, because braking distance grows with the square of speed. Add 1.5 s of reaction time and the totals are about {s1:0.0f} m and {s2:0.0f} m. Wet or icy roads make it much longer. Shown at half speed. Save this one.",
                tags=["driving", "safety", "physics", "didyouknow", "cars", "shipwithpip"],
                pinned=f"The math: braking distance = v squared / (2 x 0.7 x 9.81). 50 km/h = {b1:0.1f} m. 100 km/h = {b2:0.1f} m. Wet roads (mu about 0.4) make it roughly 1.75x longer.",
                notes=f"Model: reaction time {RT} s at constant speed, then constant deceleration mu*g with mu = {MU} (typical dry asphalt; real values vary 0.5-0.8; wet about 0.4; snow/ice much lower). 50 km/h = {v1:.2f} m/s: reaction {v1 * RT:.1f} m + braking {b1:.1f} m = {s1:.1f} m. 100 km/h = {v2:.2f} m/s: reaction {v2 * RT:.1f} m + braking {b2:.1f} m = {s2:.1f} m. 1.5 s reaction is a commonly used design figure (AASHTO uses 2.5 s). Real cars with ABS and good tires can brake harder or softer. This is an idealised illustration.")
    return reel, buf, cfg, post


# ============================================================== 2. bacteria on the counter
class BacReel(MinReel):
    T0, TD = 1.2, 24.0  # 4 hours in TD seconds
    N = 4096

    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        r = np.random.default_rng(5)
        pts = []
        while len(pts) < self.N:
            x, y = r.uniform(-1, 1, 2)
            if x * x + y * y <= 0.96:
                pts.append((x, y))
        self.pts = pts

    def mins(self, t):
        return clamp((t - self.T0) / self.TD) * 240

    def frame(self, t):
        m = self.mins(t)
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        n = 2 ** (m / 20)
        cx, cy, R = W // 2, 920, 360
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(18, 22, 40, 255), outline=(255, 255, 255, 60), width=3)
        k = int(n)
        col = GREEN if m < 120 else (GOLD if m < 180 else RED)
        for i in range(min(k, self.N)):
            x, y = self.pts[i]
            r = 9 if k < 80 else (6 if k < 600 else 4)
            d.ellipse([cx + x * (R - 12) - r, cy + y * (R - 12) - r, cx + x * (R - 12) + r, cy + y * (R - 12) + r], fill=col + (235,))
        if m < 120:
            lines, tag = [("FOOD LEFT OUT", WHITE), ("ON THE COUNTER", GOLD)], "BETWEEN 4 AND 60 C"
        else:
            lines, tag = [("2 HOURS.", GOLD), ("THEN BIN IT.", WHITE)], "SAVE THIS"
        self.title(d, lines, tag)
        hh, mm = int(m // 60), int(m % 60)
        big(d, 1380, f"{hh}:{mm:02d}", "ROOM TEMPERATURE CLOCK")
        d.text((W / 2, 1520), f"{int(n):,} x  (best case growth)", font=font(F_SEMI, 40), fill=col + (255,), anchor="mm")
        if m >= 120:
            a = int(255 * clamp((m - 120) / 10))
            card(d, 395, 485, a, ["DANGER ZONE 4-60 C  (40-140 F)"], 38)
        return self.handle(img, t)


def build_bacteria(model):
    T0, TD = BacReel.T0, BacReel.TD
    t2h = T0 + TD / 2
    tend = T0 + TD
    cfg = mkcfg([0.4, T0 + 4, T0 + 10, t2h, t2h + 6, tend + 1], 222)
    lines = [(0.2, "This is what happens to food you leave out on the counter."),
             (T0 + 0.4, "Each dot is a bacterium. In ideal conditions, some can double every twenty minutes."),
             (T0 + 7.0, "Stay on the counter, between four and sixty degrees, and it snowballs."),
             (t2h - 1.4, "Two hours in, that's about sixty times more."),
             (t2h + 3.2, "This is why the rule is two hours. One hour if it's hot out."),
             (tend + 0.2, f"Four hours, and in the best case it's about four thousand times more."),
             (tend + 5.6, "Reheating doesn't fix every kind of toxin. Save this.")]
    reel, buf = finish_reel(cfg, lines, model, BacReel, t2h, 330)
    post = dict(title="SAVE-4  The 2 hour rule",
                caption="Food left in the 'danger zone' (4 to 60 C, or 40 to 140 F) can let bacteria multiply fast; in ideal conditions some double in as little as 20 minutes. At that best-case pace, 2 hours is 64x and 4 hours is 4,096x. The standard advice from USDA: refrigerate or discard perishable food after 2 hours, or 1 hour if it's above 32 C (90 F). Real growth is usually slower, and it depends on the food and the bacteria. Save this.",
                tags=["foodsafety", "cooking", "didyouknow", "health", "kitchen", "shipwithpip"],
                pinned="Cheat sheet: 2 hours on the counter max (1 hour if above 32 C / 90 F). Fridge 4 C or colder. Cook leftovers to 74 C / 165 F when reheating.",
                notes="Shown growth is an idealised best case: 2^(minutes/20). 120 min -> 64x, 240 min -> 4,096x. Real doubling times depend on species, food and temperature; many are slower than 20 min. The 20-minute figure and 2 hour / 1 hour (above 90 F) rule are USDA FSIS guidance, danger zone 40-140 F (4-60 C). I could only confirm the sources via search listings (fsis.usda.gov, 'Danger Zone'), not read the page this session, so double check the wording before posting. Heating does not destroy every toxin (e.g. Staph. aureus enterotoxin, B. cereus emetic toxin). Reheat temperature 74 C / 165 F is USDA guidance.")
    return reel, buf, cfg, post


# ============================================================== 3. CPR rhythm
BPM = 110


class CprReel(MinReel):
    T0 = 4.0

    def frame(self, t):
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        s = max(0.0, t - self.T0)
        beat = (s * BPM / 60.0)
        ph = beat % 1.0
        on = t >= self.T0
        push = (1 - ph) ** 3 if on else 0
        cx, cy = W // 2, 880
        # heart pulse
        for k in range(3):
            q = (ph + k * 0.33) % 1.0 if on else 0
            if on:
                r = 150 + q * 260
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=RED + (int(150 * (1 - q)),), width=4)
        r0 = 150 - 22 * push if on else 150
        d.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], fill=(255, 88, 104, 255))
        d.text((cx, cy), "PUSH", font=font(F_BOLD, 64), fill=WHITE + (255,), anchor="mm")
        # target tempo band 100-120
        bx0, bx1, by = 140, W - 140, 1230
        f = lambda b: bx0 + (bx1 - bx0) * (b - 80) / 60  # noqa: E731
        d.rounded_rectangle([bx0, by, bx1, by + 12], radius=6, fill=(255, 255, 255, 30))
        d.rounded_rectangle([f(100), by - 6, f(120), by + 18], radius=9, fill=GREEN + (255,))
        for b in (80, 100, 110, 120, 140):
            d.text((f(b), by + 52), str(b), font=font(F_MONO, 24), fill=SOFT + (255,), anchor="mm")
        d.ellipse([f(BPM) - 12, by - 6, f(BPM) + 12, by + 18], fill=WHITE + (255,))
        if t < 7.5:
            lines, tag = [("SOMEONE COLLAPSED", WHITE), ("NOT BREATHING", GOLD)], "30 SECONDS THAT MATTER"
        else:
            lines, tag = [("PUSH HARD. FAST.", GOLD), ("100-120 A MINUTE", WHITE)], "SAVE THIS"
        self.title(d, lines, tag)
        cnt = int(beat) if on else 0
        big(d, 1420, f"{cnt}", f"PUSHES AT {BPM} PER MINUTE")
        if t > 22:
            card(d, 1010, 1140, int(255 * clamp((t - 22) / 0.6)), ["1. CALL EMERGENCY (911)", "2. CENTRE OF CHEST, 5-6 CM DEEP"], 34)
        return self.handle(img, t)


def build_cpr(model):
    cfg = mkcfg([0.4, 5, 9, 14, 24, 30], 223)
    lines = [(0.2, "If someone collapses and isn't breathing, this is what you do."),
             (3.6, "First, call emergency services. Then push hard in the centre of the chest."),
             (10.0, "Fast. One hundred to one hundred twenty pushes a minute. That's this pulse."),
             (16.0, "Push about five to six centimeters deep, two inches, and let the chest come back up."),
             (23.5, "Don't stop until help or a defibrillator arrives."),
             (28.5, "The song Stayin' Alive has about this tempo. Take a class if you can. Save this.")]

    def clicks(buf):
        for k in range(int((60 - CprReel.T0) * BPM / 60)):
            tone(buf, CprReel.T0 + k * 60.0 / BPM, 0.07, 900, 600, 0.10, "sine")
    reel, buf = finish_reel(cfg, lines, model, CprReel, 26, 440, clicks)
    post = dict(title="SAVE-5  The tempo that can save a life",
                caption="Hands-only CPR basics for an adult who collapses and isn't breathing normally: call emergency services (911 in Canada and the US), then push hard and fast in the centre of the chest at 100 to 120 compressions per minute, about 5 to 6 cm (2 to 2.4 in) deep, letting the chest rise fully between pushes. This reel is not training. Please take a certified CPR course (Red Cross, St. John Ambulance, Heart & Stroke).",
                tags=["cpr", "firstaid", "safety", "lifesaving", "didyouknow", "shipwithpip"],
                pinned="Save this and share it. Then book a real CPR class (Red Cross, St. John Ambulance, Heart and Stroke).",
                notes=f"Tempo shown: {BPM} per minute (guideline range 100-120). Source: American Heart Association CPR guidelines (hands-only CPR: 100-120/min, depth at least 2 in / 5 cm, not more than 2.4 in / 6 cm for adults). I could only confirm via search listings (heart.org) this session, not read the page, so verify before posting. Stayin' Alive is commonly cited at about 103 bpm. Guidelines differ for children, infants and trained responders. Not a substitute for training; do not describe as medical advice.")
    return reel, buf, cfg, post


# ============================================================== 4. $5 a day
RATE, ANNUAL = 0.07, 5 * 365


def fv_series(years=40):
    bal, out = 0.0, [(0, 0.0, 0.0)]
    for y in range(1, years + 1):
        bal = bal * (1 + RATE) + ANNUAL
        out.append((y, bal, ANNUAL * y))
    return out


class CompoundReel(MinReel):
    T0, TD = 1.2, 24.0

    def frame(self, t):
        ser = fv_series()
        yf = clamp((t - self.T0) / self.TD) * 40
        y = int(yf)
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        gx0, gx1, gy0, gy1 = 110, 970, 1270, 620
        top = ser[-1][1] * 1.05
        d.line([gx0, gy0, gx1, gy0], fill=(255, 255, 255, 60), width=2)
        for yy in (0, 10, 20, 30, 40):
            x = gx0 + (gx1 - gx0) * yy / 40
            d.line([x, gy0, x, gy0 + 12], fill=(255, 255, 255, 90), width=2)
            d.text((x, gy0 + 40), f"{yy}y", font=font(F_MONO, 24), fill=SOFT + (230,), anchor="mm")
        bw = (gx1 - gx0) / 40 * 0.78
        for i in range(0, y + 1):
            yr, bal, con = ser[i]
            x = gx0 + (gx1 - gx0) * (yr - 0.5) / 40 if yr else gx0
            if yr == 0:
                continue
            hb = (gy0 - gy1) * bal / top
            hc = (gy0 - gy1) * con / top
            d.rectangle([x - bw / 2, gy0 - hc, x + bw / 2, gy0], fill=(140, 150, 180, 255))
            d.rectangle([x - bw / 2, gy0 - hb, x + bw / 2, gy0 - hc], fill=GOLD + (255,))
        bal, con = ser[y][1], ser[y][2]
        if t < self.T0 + self.TD + 0.2:
            lines, tag = [("$5 A DAY", WHITE), ("FOR 40 YEARS", GOLD)], "7% ASSUMED. NOT GUARANTEED"
        else:
            lines, tag = [("$73K IN.", WHITE), ("$364K OUT.", GOLD)], "SAVE THIS"
        self.title(d, lines, tag)
        big(d, 1420, f"${bal:,.0f}", f"YEAR {y}   YOU PUT IN ${con:,.0f}", GOLD if y >= 40 else WHITE)
        d.rectangle([110, 1585, 130, 1605], fill=(140, 150, 180, 255))
        d.text((142, 1596), "what you put in", font=font(F_MONO, 24), fill=SOFT + (255,), anchor="lm")
        d.rectangle([450, 1585, 470, 1605], fill=GOLD + (255,))
        d.text((482, 1596), "growth", font=font(F_MONO, 24), fill=SOFT + (255,), anchor="lm")
        return self.handle(img, t)


def build_compound(model):
    ser = fv_series()
    fv, con = ser[-1][1], ser[-1][2]
    half_year = next(y for y, b, c in ser if b - c > c)
    T0, TD = CompoundReel.T0, CompoundReel.TD
    tend = T0 + TD
    t_half = T0 + TD * half_year / 40
    cfg = mkcfg([0.4, T0 + 6, t_half, T0 + 18, tend + 0.4, tend + 3], 224)
    lines = [(0.2, "Five dollars a day. Here's what that turns into."),
             (T0 + 1.2, "That's about eighteen hundred dollars a year, invested, assuming seven percent growth."),
             (t_half - 0.8, f"By year {half_year}, the growth is bigger than everything you put in."),
             (tend - 3.0, "Most of the money shows up at the very end."),
             (tend + 0.3, f"After forty years, you put in seventy three thousand. It's worth about {fv / 1000:0.0f} thousand."),
             (tend + 6.4, "Not guaranteed, and before inflation. But time is the whole trick. Save this.")]
    reel, buf = finish_reel(cfg, lines, model, CompoundReel, tend, 392)
    post = dict(title="SAVE-6  What $5 a day becomes",
                caption=f"$5 a day is $1,825 a year. Invested at an assumed 7% annual return for 40 years, that grows to about ${fv:,.0f}, even though you only put in ${con:,.0f}. By year {half_year}, growth passes your own contributions. This is a hypothetical illustration, not a prediction or advice: returns vary year to year, and the figure is before inflation, fees and taxes.",
                tags=["investing", "money", "compoundinterest", "personalfinance", "didyouknow", "shipwithpip"],
                pinned="The formula: balance = yearly deposit x ((1.07^years - 1) / 0.07). $1,825 x 199.6 = about $364K. Change the 7% and the years to see how much both matter.",
                notes=f"Deposit ${ANNUAL}/yr added at the end of each year, annual compounding at {RATE:.0%}, 40 years: FV = {ANNUAL} * ((1.07^40 - 1)/0.07) = ${fv:,.2f}; contributions ${con:,.0f}. Growth first exceeds contributions in year {half_year}. 7% is an assumption (often cited as an approximate long-run real return of broad US stocks, but not guaranteed and not valid for every market/period). Not financial advice; I am not a licensed advisor. Check this before posting if you want a source line.")
    return reel, buf, cfg, post


# ============================================================== 5. cold water 1-10-1
def cw_minutes(t, T0=1.5):
    s = max(0.0, t - T0)
    if s < 5:
        return s / 5.0
    if s < 12:
        return 1 + (s - 5) / 7.0 * 9
    if s < 19:
        return 10 + (s - 12) / 7.0 * 50
    return 60.0


class ColdReel(MinReel):
    def frame(self, t):
        m = cw_minutes(t)
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        # water
        wy = 760
        for i in range(0, 520, 4):
            a = int(110 * (1 - i / 520))
            d.rectangle([0, wy + i, W, wy + i + 4], fill=(30, 80, 140, a))
        d.line([0, wy, W, wy], fill=ICE + (180,), width=3)
        ph = 1 if m < 1 else (2 if m < 10 else 3)
        # person (head) bobbing, panics in minute 1
        shake = (math.sin(t * 25) * 8 if m < 1 and t > 1.5 else 0)
        sink = clamp((m - 10) / 50) * 70
        d.ellipse([W / 2 - 30 + shake, wy - 40 + sink, W / 2 + 30 + shake, wy + 20 + sink], fill=(235, 240, 255, 255))
        if ph == 1:
            for k in range(3):
                q = (t * 2 + k / 3) % 1
                d.ellipse([W / 2 - 40 - q * 150, wy - 12 - q * 12, W / 2 + 40 + q * 150, wy + 12 + q * 12], outline=ICE + (int(160 * (1 - q)),), width=2)
        # timeline
        x0, x1, y = 130, W - 130, 1230
        d.rounded_rectangle([x0, y, x1, y + 12], radius=6, fill=(255, 255, 255, 30))
        marks = [(0, "0"), (1, "1 MIN"), (10, "10 MIN"), (60, "1 HOUR")]
        pos = lambda mm: x0 + (x1 - x0) * (0 if mm <= 0 else (mm / 1 * 0.2 if mm <= 1 else (0.2 + (mm - 1) / 9 * 0.35 if mm <= 10 else 0.55 + (mm - 10) / 50 * 0.45)))  # noqa: E731
        cols = {1: RED, 2: GOLD, 3: ICE}
        px = pos(m)
        d.rounded_rectangle([x0, y, px, y + 12], radius=6, fill=cols[ph] + (255,))
        for mm, lab in marks:
            d.line([pos(mm), y - 8, pos(mm), y + 20], fill=(255, 255, 255, 120), width=2)
            d.text((pos(mm), y + 52), lab, font=font(F_MONO, 24), fill=SOFT + (255,), anchor="mm")
        d.ellipse([px - 14, y - 8, px + 14, y + 20], fill=WHITE + (255,))
        names = {1: ("COLD SHOCK", "GASPING. CONTROL YOUR BREATHING"), 2: ("LOSING STRENGTH", "SWIM OR CLIMB OUT NOW"), 3: ("HYPOTHERMIA", "TIME IS RUNNING OUT")}
        if t < 8:
            lines, tag = [("YOU FALL INTO", WHITE), ("COLD WATER", GOLD)], "THE 1-10-1 RULE"
        else:
            lines, tag = [("1 - 10 - 1", GOLD), ("SAVE THIS", WHITE)], "COLD WATER SURVIVAL GUIDELINE"
        self.title(d, lines, tag)
        nm, sb = names[ph]
        d.text((W / 2, 1380), nm, font=font(F_BOLD, 80), fill=cols[ph] + (255,), anchor="mm")
        d.text((W / 2, 1460), sb, font=font(F_MONO, 28), fill=SOFT + (255,), anchor="mm")
        if m >= 60:
            card(d, 1010, 1130, int(255 * clamp((t - 21) / 0.6)), ["WEAR A LIFEJACKET. IT BUYS YOU ALL 3."], 32)
        return self.handle(img, t)


def build_coldwater(model):
    cfg = mkcfg([0.4, 3, 8, 14, 22, 28], 225)
    t1, t10, t60 = 1.5 + 5, 1.5 + 12, 1.5 + 19
    lines = [(0.2, "If you fall into cold water, you have less time than you think."),
             (1.8, "First minute: cold shock. You gasp and can't control your breathing. Don't panic, it passes."),
             (t1 + 0.2, "Ten minutes: your arms and legs lose strength. Get out, or get to something that floats."),
             (t10 + 0.4, "One hour: hypothermia can make you pass out."),
             (t60 + 0.3, "A lifejacket keeps your head up through all three."),
             (t60 + 4.2, "Numbers change with water temperature, so treat it as a guide. Save this.")]
    reel, buf = finish_reel(cfg, lines, model, ColdReel, t60, 330)
    post = dict(title="SAVE-7  The 1-10-1 rule for cold water",
                caption="Cold water survival guideline from Canadian researcher Dr. Gordon Giesbrecht: 1 minute to get your breathing under control (cold shock), 10 minutes of useful movement before your arms and legs weaken, and about 1 hour before hypothermia can make you pass out. Real times vary a lot with water temperature, clothing and fitness, so treat it as a guide, not a promise. A lifejacket helps with all three. Especially worth knowing on our rivers in the spring and fall.",
                tags=["watersafety", "safety", "canada", "survival", "didyouknow", "shipwithpip"],
                pinned="Save this for boating, paddling, ice season. Wear a lifejacket. It's the one thing that works in all three phases.",
                notes="The 1-10-1 guideline is widely taught in Canadian water-safety materials, attributed to Dr. Gordon Giesbrecht (University of Manitoba). Water-safety organisations note the numbers are approximate and vary with water temperature (the guideline is for roughly 0-15 C water); some sources call parts of it a simplification (see coldwatersafety.org, '1-10-1 myth'). I could only confirm via search listings, not read the pages, so verify wording and attribution before posting. Animation uses a nonlinear time scale (1 min = 5 s, 10 min = 7 s, 1 hour = 7 s).")
    return reel, buf, cfg, post


BUILDERS = {"stopping": build_stopping, "bacteria": build_bacteria, "cpr": build_cpr, "compound": build_compound, "coldwater": build_coldwater}


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
