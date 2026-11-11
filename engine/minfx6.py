"""5 counter-style viral reels: heart | phone | blinks | parents | walk.  usage: python minfx6.py NAME out.mp4 --model <kokoro dir> [--stills png]"""
import argparse, math, os, sys
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa
from minfx import DIM, GOLD, ICE, SOFT, WHITE, MinReel  # noqa
from minfx2 import GREEN, RED, big, finish_reel, mkcfg  # noqa
from scenes import render  # noqa
from simfx import F_BOLD, F_MONO, F_SEMI, W, clamp, font  # noqa

T0, TD = 2.0, 19.0


# ------------------------------------------------------------ 1. heartbeats
BPM = 70
PER_YEAR = BPM * 60 * 24 * 365.25


class HeartReel(MinReel):
    def age(self, t): return clamp((t - T0) / TD) * 80
    def frame(self, t):
        a = self.age(t); img = self.new_frame(t); d = ImageDraw.Draw(img, "RGBA")
        ph = (t * BPM / 60) % 1.0
        k = 1 + 0.13 * math.exp(-ph * 7) + 0.04 * math.exp(-((ph - 0.28) * 9) ** 2)
        cx, cy, s = W / 2, 880, 17 * k
        pts = [(cx + s * 16 * math.sin(u) ** 3, cy - s * (13 * math.cos(u) - 5 * math.cos(2 * u) - 2 * math.cos(3 * u) - math.cos(4 * u))) for u in [i * 2 * math.pi / 120 for i in range(120)]]
        for r, al in ((1.5, 20), (1.28, 40)):
            d.polygon([(cx + (x - cx) * r, cy + (y - cy) * r) for x, y in pts], fill=(255, 70, 90, al))
        d.polygon(pts, fill=(255, 62, 86, 255))
        done = a >= 80
        lines, tag = ([("YOUR HEART WON'T", WHITE), ("STOP. EVER.", RED)], "AT 70 BEATS PER MINUTE") if not done else ([("2.9 BILLION", RED), ("BEATS", WHITE)], "IN AN 80 YEAR LIFE")
        self.title(d, lines, tag)
        big(d, 1420, f"{int(a * PER_YEAR):,}", f"BEATS BY AGE {int(a)}")
        return self.handle(img, t)

def build_heart(model):
    t = lambda age: T0 + age / 80 * TD; tend = T0 + TD
    cfg = mkcfg([0.4, t(5), t(20), t(40), tend + 0.3, tend + 3], 261)
    lines = [(0.2, "Your heart has never taken a day off. Watch it count."),
             (T0 + 2.0, "At seventy beats a minute, that's about thirty seven million beats every year."),
             (t(40) + 0.2, "Forty years old? Over one and a half billion beats. Zero breaks."),
             (tend + 0.3, "By eighty, about two point nine billion."),
             (tend + 4.0, "Do something worth that.")]
    reel, buf = finish_reel(cfg, lines, model, HeartReel, tend, 330)
    post = dict(title="VIRAL-A  Your heart beats 2.9 billion times",
        caption="70 beats/min x 60 x 24 x 365.25 = about 36.8 million beats a year. By 80 that's about 2.9 billion. Your resting rate is probably different, so treat 70 as an average. Send this to someone with a big heart.",
        tags=["heart", "didyouknow", "science", "perspective", "health", "shipwithpip"],
        pinned="Math: 70 x 60 x 24 x 365.25 = 36.8M beats/year. x 80 = 2.94B. Your own resting heart rate changes the number.",
        notes="Pure arithmetic from an assumed 70 bpm average (typical adult resting range is roughly 60-100 bpm; heart rate changes with age and fitness, so the total is an estimate). 2.94 billion at 80. No outside source required; avoid giving any health advice.")
    return reel, buf, cfg, post


# ------------------------------------------------------------ 2. phone years
class PhoneReel(MinReel):
    A0, A1 = 15, 75
    def age(self, t): return self.A0 + clamp((t - T0) / TD) * (self.A1 - self.A0)
    def frame(self, t):
        a = self.age(t); yrs = (a - self.A0) * 3 / 24
        img = self.new_frame(t); d = ImageDraw.Draw(img, "RGBA")
        cs, cols = 76, 10; x0, y0 = (W - cols * cs) // 2, 560
        for i in range(80):
            r, c = divmod(i, cols); x, y = x0 + c * cs, y0 + r * cs
            d.rounded_rectangle([x + 3, y + 3, x + cs - 5, y + cs - 5], radius=10, fill=(120, 138, 196, 255))
            f = clamp(yrs - i)
            if f > 0: d.rounded_rectangle([x + 3, y + 3, x + 3 + (cs - 8) * f, y + cs - 5], radius=10, fill=(255, 70, 90, 255))
        done = a >= self.A1
        lines, tag = ([("3 HOURS A DAY", WHITE), ("ON YOUR PHONE", RED)], "EACH SQUARE IS ONE YEAR OF YOUR LIFE") if not done else ([("7.5 YEARS", RED), ("GONE", WHITE)], "AGE 15 TO 75")
        self.title(d, lines, tag)
        big(d, 1460, f"{yrs:0.1f}", f"YEARS ON YOUR PHONE BY AGE {int(a)}")
        return self.handle(img, t)

def build_phone(model):
    tend = T0 + TD; t = lambda age: T0 + (age - 15) / 60 * TD
    cfg = mkcfg([0.4, t(25), t(45), t(65), tend + 0.3, tend + 3], 262)
    lines = [(0.2, "Three hours a day on your phone. Watch what that costs over a life."),
             (T0 + 2.5, "Each square is a year. Red is the time your screen takes."),
             (t(45) + 0.2, "By forty five, about three years of your life. Gone."),
             (tend + 0.3, "By seventy five, seven and a half years. Full days, nonstop."),
             (tend + 4.8, "Check your screen time. Then decide.")]
    reel, buf = finish_reel(cfg, lines, model, PhoneReel, tend, 392)
    post = dict(title="VIRAL-B  3 hours a day = 7.5 years",
        caption="3 hours a day from age 15 to 75 = 3 x 365.25 x 60 = 65,745 hours = 7.5 years of non-stop 24-hour days. Your number is different. Go check your screen time and comment it. No judgment, just math.",
        tags=["screentime", "phone", "perspective", "productivity", "math", "shipwithpip"],
        pinned="What's your daily screen time? Math: hours/day x 365.25 x 60 years / 8,766 hours per year.",
        notes="Pure arithmetic with an assumed 3 h/day for 60 years (age 15-75). 65,745 h / 8,766 h per year = 7.5 years of continuous time. It is an example, not a claim about average usage; do not present it as a statistic.")
    return reel, buf, cfg, post


# ------------------------------------------------------------ 3. blinks
BL_YEAR = 15 * 60 * 16 * 365.25  # blinks awake


class BlinkReel(MinReel):
    def age(self, t): return clamp((t - T0) / TD) * 80
    def frame(self, t):
        a = self.age(t); img = self.new_frame(t)
        ph = (t * 1.0) % 1.0
        o = 1 - math.exp(-((ph - 0.12) / 0.045) ** 2)
        cx, cy, hw, hh = W / 2, 880, 330, 230 * max(0.04, o)
        up = [(cx + hw * x, cy - hh * (1 - x * x)) for x in [i / 30 - 1 for i in range(61)]]
        dn = [(cx + hw * x, cy + hh * 0.75 * (1 - x * x)) for x in [1 - i / 30 for i in range(61)]]
        poly = up + dn
        mask = Image.new("L", img.size, 0); ImageDraw.Draw(mask).polygon(poly, fill=255)
        layer = Image.new("RGBA", img.size, (240, 244, 255, 255)); ld = ImageDraw.Draw(layer)
        ld.ellipse([cx - 150, cy - 150, cx + 150, cy + 150], fill=(70, 120, 200, 255)); ld.ellipse([cx - 150, cy - 150, cx + 150, cy + 150], outline=(30, 60, 120, 255), width=10)
        ld.ellipse([cx - 62, cy - 62, cx + 62, cy + 62], fill=(10, 10, 20, 255)); ld.ellipse([cx - 90, cy - 100, cx - 40, cy - 50], fill=(255, 255, 255, 230))
        img.paste(layer.convert("RGB") if img.mode == "RGB" else layer, (0, 0), mask)
        d = ImageDraw.Draw(img, "RGBA"); d.line(poly + [poly[0]], fill=(20, 20, 40, 255), width=10, joint="curve")
        done = a >= 80
        lines, tag = ([("YOU JUST BLINKED", WHITE), ("AGAIN", GOLD)], "ABOUT 15 TIMES A MINUTE") if not done else ([("420 MILLION", GOLD), ("BLINKS", WHITE)], "IN AN 80 YEAR LIFE")
        self.title(d, lines, tag)
        big(d, 1420, f"{int(a * BL_YEAR):,}", f"BLINKS BY AGE {int(a)}")
        return self.handle(img, t)

def build_blinks(model):
    tend = T0 + TD; t = lambda age: T0 + age / 80 * TD
    cfg = mkcfg([0.4, t(8), t(30), t(60), tend + 0.3, tend + 3], 263)
    lines = [(0.2, "You blink about fifteen times a minute. You never notice. Watch."),
             (T0 + 2.0, "Awake, that's over five million blinks a year."),
             (t(40) + 0.2, "By forty, over two hundred million."),
             (tend + 0.3, "By eighty, about four hundred and twenty million."),
             (tend + 4.2, "And you just blinked again.")]
    reel, buf = finish_reel(cfg, lines, model, BlinkReel, tend, 349)
    post = dict(title="VIRAL-C  420 million blinks",
        caption="15 blinks/min x 60 x 16 waking hours x 365.25 = about 5.3 million blinks a year. By 80 that's about 420 million. Rates vary a lot (about 15-20 a minute is a common figure, and you blink less when staring at a screen). Blink twice if you noticed.",
        tags=["didyouknow", "body", "science", "perspective", "facts", "shipwithpip"],
        pinned="Math: 15 x 60 x 16 x 365.25 = 5.26M/yr, x 80 = 421M. Typical figures vary by person and activity.",
        notes="Assumes 15 blinks per minute for 16 waking hours; commonly cited average is roughly 15-20 per minute, lower when concentrating on screens. 5.26M/yr, 421M at 80. It is an estimate; sources vary, none opened.")
    return reel, buf, cfg, post


# ------------------------------------------------------------ 4. parents
class ParentsReel(MinReel):
    TOT = 250
    def left(self, t): return self.TOT - int(clamp((t - T0) / TD) * self.TOT)
    def frame(self, t):
        n = self.left(t); img = self.new_frame(t); d = ImageDraw.Draw(img, "RGBA")
        cols, rows, cs = 10, 25, 31; x0, y0 = (W - cols * cs) // 2, 505
        used = self.TOT - n
        for i in range(self.TOT):
            r, c = divmod(i, cols); x, y = x0 + c * cs, y0 + r * cs
            col = (36, 40, 60, 255) if i < used else (255, 190, 90, 255)
            d.ellipse([x + 3, y + 3, x + cs - 4, y + cs - 4], fill=col)
        for yr in (0, 10, 20):
            d.text((x0 - 14, y0 + (yr + 0.5) * cs), f"{55 + yr}", font=font(F_MONO, 22), fill=SOFT + (255,), anchor="rm")
        done = n <= 0
        lines, tag = ([("TIMES YOU'LL SEE", WHITE), ("YOUR PARENTS", GOLD)], "EACH DOT IS ONE VISIT") if not done else ([("THAT'S ALL", RED), ("250 VISITS", WHITE)], "NOT AS MANY AS YOU THOUGHT")
        self.title(d, lines, tag)
        big(d, 1460, f"{n}", "VISITS LEFT")
        d.text((W / 2, 1585), "EXAMPLE: PARENTS 55 TO 80  -  10 VISITS A YEAR", font=font(F_MONO, 24), fill=SOFT + (255,), anchor="mm")
        return self.handle(img, t)

def build_parents(model):
    tend = T0 + TD
    cfg = mkcfg([0.4, T0 + 4, T0 + 10, T0 + 15, tend + 0.3, tend + 3.5], 264)
    lines = [(0.2, "How many times will you see your parents again? Count them."),
             (T0 + 1.5, "Say they're fifty five, and you see them ten times a year, until they're eighty."),
             (T0 + 11.5, "That's twenty five years. Two hundred and fifty visits. Watch them go."),
             (tend + 0.3, "That's all of it. Not endless. Just two hundred and fifty."),
             (tend + 4.8, "Call them tonight.")]
    reel, buf = finish_reel(cfg, lines, model, ParentsReel, tend, 294)
    post = dict(title="VIRAL-D  250 visits left with your parents",
        caption="Example math: parents are 55, you see them 10 times a year until they're 80. 25 years x 10 = 250 visits. Change the numbers for your own life: (80 - their age) x visits per year. Then call them. Tag someone who needs this tonight.",
        tags=["family", "parents", "perspective", "gratitude", "love", "shipwithpip"],
        pinned="Your number: (80 minus their age) x visits per year. Mine is an example, not a prediction. Go call them.",
        notes="Pure arithmetic on a stated hypothetical (parents 55, 10 visits a year, until 80). It is an example, not a prediction; the reel labels it as such on screen. Tone is emotional: avoid implying anything about any viewer's parents. Optional: credit Tim Urban's 'The Tail End' (Wait But Why) for the idea of counting remaining time with parents.")
    return reel, buf, cfg, post


# ------------------------------------------------------------ 5. walk around earth
STEP_M, STEPS_DAY = 0.75, 7500
KM_YEAR = STEP_M * STEPS_DAY * 365.25 / 1000
EARTH = 40075


class WalkReel(MinReel):
    A0, A1 = 5, 80
    def age(self, t): return self.A0 + clamp((t - T0) / TD) * (self.A1 - self.A0)
    def frame(self, t):
        a = self.age(t); km = (a - self.A0) * KM_YEAR
        img = self.new_frame(t); d = ImageDraw.Draw(img, "RGBA")
        cx, cy, R = W / 2, 880, 300
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(24, 60, 120, 255)); d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(120, 170, 255, 255), width=6)
        for lon in (-0.6, 0, 0.6):
            d.ellipse([cx - R * abs(math.cos(lon * 1.2)) * 0.9 - 1, cy - R, cx + R * abs(math.cos(lon * 1.2)) * 0.9 + 1, cy + R], outline=(255, 255, 255, 30), width=2)
        d.ellipse([cx - R + 90, cy - 80, cx - 30, cy + 90], fill=(60, 160, 100, 255)); d.ellipse([cx + 20, cy - 190, cx + 220, cy - 30], fill=(60, 160, 100, 255))
        laps = km / EARTH; R2 = R + 50
        d.ellipse([cx - R2, cy - R2, cx + R2, cy + R2], outline=(255, 255, 255, 40), width=10)
        ang = (laps % 1.0) * 360
        if laps >= 1: d.ellipse([cx - R2, cy - R2, cx + R2, cy + R2], outline=GOLD + (120,), width=10)
        if ang > 0.5: d.arc([cx - R2, cy - R2, cx + R2, cy + R2], -90, -90 + ang, fill=GOLD + (255,), width=10)
        th = math.radians(-90 + ang); px, py = cx + R2 * math.cos(th), cy + R2 * math.sin(th)
        d.ellipse([px - 18, py - 18, px + 18, py + 18], fill=(255, 255, 255, 255)); d.ellipse([px - 11, py - 11, px + 11, py + 11], fill=GOLD + (255,))
        done = a >= self.A1
        lines, tag = ([("YOU WILL WALK", WHITE), ("AROUND THE EARTH", GOLD)], "AT 7,500 STEPS A DAY") if not done else ([(f"{laps:0.1f} LAPS", GOLD), ("AROUND EARTH", WHITE)], "BETWEEN AGE 5 AND 80")
        self.title(d, lines, tag)
        big(d, 1420, f"{km:,.0f}", f"KM WALKED BY AGE {int(a)}   -   {laps:0.1f} LAPS")
        return self.handle(img, t)

def build_walk(model):
    tend = T0 + TD; t = lambda age: T0 + (age - 5) / 75 * TD
    cfg = mkcfg([0.4, t(15), t(40), t(70), tend + 0.3, tend + 3], 265)
    lines = [(0.2, "Walk seven and a half thousand steps a day, and you will walk around the Earth."),
             (T0 + 2.0, "Not once. Watch the counter."),
             (t(45) + 0.2, "The Earth is about forty thousand kilometers around. You pass the first lap in your thirties."),
             (tend + 0.3, "By eighty, almost four laps."),
             (tend + 4.3, "All one step at a time.")]
    reel, buf = finish_reel(cfg, lines, model, WalkReel, tend, 440)
    post = dict(title="VIRAL-E  Walk around the Earth ~4 times",
        caption=f"7,500 steps a day x 0.75 m x 365.25 = about {KM_YEAR:,.0f} km a year. Age 5 to 80 is about {(75*KM_YEAR):,.0f} km, and Earth's circumference is 40,075 km, so about {75*KM_YEAR/EARTH:0.1f} laps. Your steps and stride are different. How many do you walk?",
        tags=["walking", "steps", "perspective", "didyouknow", "fitness", "shipwithpip"],
        pinned="Math: 7,500 x 0.75 m x 365.25 / 1000 = 2,055 km a year. 75 years = 154,000 km = 3.8 laps of Earth (40,075 km).",
        notes=f"Assumes 7,500 steps/day and a 0.75 m stride for 75 years (age 5-80). Real averages vary a lot by person and country; the reel presents it as 'at 7,500 steps a day', a hypothetical. {KM_YEAR:,.0f} km/yr; 75 yrs = {75*KM_YEAR:,.0f} km = {75*KM_YEAR/EARTH:0.2f} laps. Earth's equatorial circumference is 40,075 km. No health claims.")
    return reel, buf, cfg, post


BUILDERS = {"heart": build_heart, "phone": build_phone, "blinks": build_blinks, "parents": build_parents, "walk": build_walk}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("which"); ap.add_argument("out"); ap.add_argument("--model", required=True); ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = BUILDERS[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)

if __name__ == "__main__":
    main()
