"""Binary search vs linear search reel: guess a number from 1 to a billion.
usage: python searchfx.py out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
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
N = 1_000_000_000
TARGET = 743_912_805


def binary_steps(n, target):
    lo, hi = 1, n
    steps = []
    while lo <= hi:
        mid = (lo + hi) // 2
        steps.append((lo, hi, mid))
        if mid == target:
            break
        elif mid < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return steps


STEPS = binary_steps(N, TARGET)
LIN_GUESSES = TARGET  # worst case scanning 1..target


def fmt(n):
    return f"{n:,}"


class SearchReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.t0, self.hold = cfg["t0"], cfg["hold"]

    def idx(self, t):
        i = int((t - self.t0) / self.hold) if t >= self.t0 else -1
        return max(-1, min(len(STEPS) - 1, i))

    def content(self, t):
        im = Image.new("RGB", (VS, VS), BG)
        d = ImageDraw.Draw(im, "RGBA")
        bx0, bx1, by = 60, 900, 330
        d.rounded_rectangle([bx0 - 10, by - 44, bx1 + 10, by + 44], radius=18, fill=(14, 10, 34, 255), outline=(150, 140, 220, 200), width=3)
        d.text((480, by - 70), f"GUESS A NUMBER FROM 1 TO {fmt(N)}", font=font(F_MONO, 22), fill=(220, 215, 250, 255), anchor="mm")

        def xpos(v):
            return bx0 + (bx1 - bx0) * (v - 1) / (N - 1)

        i = self.idx(t)
        if i < 0:
            d.text((480, by), "?", font=font(F_BOLD, 50), fill=YEL + (255,), anchor="mm")
        else:
            lo, hi, mid = STEPS[i]
            d.line([(xpos(lo), by), (xpos(hi), by)], fill=CYAN + (120,), width=20)
            gx = xpos(mid)
            col = GREEN if mid == TARGET else PINK
            d.line([(gx, by - 56), (gx, by + 56)], fill=col + (255,), width=6)
            d.text((gx, by - 86), fmt(mid), font=font(F_BOLD, 30), fill=col + (255,), anchor="mm", stroke_width=4, stroke_fill=INK)
            label = "TOO LOW, GO RIGHT" if mid < TARGET else ("TOO HIGH, GO LEFT" if mid > TARGET else "FOUND IT!")
            d.text((480, by + 100), label, font=font(F_MONO, 24), fill=col + (255,), anchor="mm")

        # step counter big
        n_guesses = i + 1 if i >= 0 else 0
        d.rounded_rectangle([300, 470, 660, 620], radius=24, fill=(14, 10, 34, 255), outline=YEL + (220,), width=4)
        d.text((480, 510), "GUESSES SO FAR", font=font(F_MONO, 20), fill=(220, 215, 250, 255), anchor="mm")
        d.text((480, 570), str(n_guesses), font=font(F_BOLD, 60), fill=YEL + (255,), anchor="mm")

        d.text((480, 700), "BINARY SEARCH: HALVE THE RANGE EVERY GUESS", font=font(F_MONO, 22), fill=(200, 195, 240, 255), anchor="mm")
        worst = math.ceil(math.log2(N))
        d.text((480, 740), f"WORST CASE FOR {fmt(N)} NUMBERS: {worst} GUESSES", font=font(F_MONO, 22), fill=(200, 195, 240, 255), anchor="mm")
        return im

    def stats(self, t):
        i = self.idx(t)
        n_guesses = i + 1 if i >= 0 else 0
        worst = math.ceil(math.log2(N))
        frac = clamp(n_guesses / worst)
        done = i >= 0 and STEPS[i][2] == TARGET
        return [(f"{n_guesses} GUESSES", YEL), ("FOUND" if done else "SEARCHING", GREEN if done else CYAN)], frac, "PROGRESS TO WORST CASE", CYAN


def build(model):
    worst = math.ceil(math.log2(N))
    print("binary guesses used", len(STEPS), "worst case", worst, "linear worst case", N, "steps", [(s[2]) for s in STEPS])
    t0, hold = 5.6, 1.15
    nlast = len(STEPS)
    t_found = t0 + (nlast - 1) * hold
    pay = round(t_found + 1.6, 1)
    cfg = dict(T=max(90.0, t_found + 20), c1=CYAN, c2=YEL, t0=t0, hold=hold, payoff=pay, seed=201,
               still_times=[0.5, t0 + 0.5, t0 + hold * 3, t0 + hold * 8, t0 + hold * 15, t_found - 0.5, t_found + 1.5],
               header=lambda tt: (([("GUESS A NUMBER", (255, 255, 255)), (f"1 TO {fmt(N)}", YEL)], "HOW FEW GUESSES?") if tt < pay else
                                  ([(f"FOUND IT IN {nlast} GUESSES.", (255, 255, 255)), ("NOT A MILLION.", GREEN)], "BINARY SEARCH")))
    lines = [(0.2, f"I am thinking of a number from 1 to {fmt(N)}. How many guesses would you need?"),
             (t0 - 1.0, "Every guess, just ask if the number is higher or lower, and cut the range in half."),
             (t0 + hold * 3 + 0.4, "Each guess throws away half of what is left."),
             (t0 + hold * 9 + 0.2, f"A billion numbers, and we are already most of the way there."),
             (t_found - 1.6, f"Found it in {nlast} guesses."),
             (t_found + 2.4, f"Checking one by one could take up to {fmt(N)} guesses. This took {nlast}, because every guess cuts the problem in half."),
             (t_found + 9.0, "That is binary search. It is why looking something up in a sorted list is fast. Comment SEARCH for the prompt.")]
    buf, vo = make_audio(cfg, lines, model)
    for k_, s in enumerate(STEPS):
        tone(buf, t0 + k_ * hold, 0.12, 420 + 20 * k_, 480 + 20 * k_, 0.05)
    chime(buf, cfg["payoff"])
    cfg["T"] = vo[-1][1] + 2.4
    reel = SearchReel(cfg, vo)
    post = dict(title="CS-1  guess a number from 1 to a billion (binary search)",
                caption=f"I picked a number from 1 to {fmt(N)}. Guessing one at a time could take up to {fmt(N)} tries. Using binary search, asking 'higher or lower' and cutting the range in half each time, it took {nlast} guesses. For any list of n items, binary search needs at most ceil(log2(n+1)) guesses, which is {worst} for a billion. That is why looking something up in a sorted list, a phone book, a dictionary, a database index, is fast.\n\nComment SEARCH and I'll send the prompt so you can build it.",
                tags=["computerscience", "algorithms", "binarysearch", "coding", "cs", "shipwithpip"],
                pinned="Build me a single-file HTML page that picks a random number from 1 to 1,000,000,000 and visually runs binary search to find it: show the current low/high range, the guess, whether it's too high or too low, and a counter of guesses used. Compare it to a counter showing how many guesses a one-by-one linear search would need for the same number.",
                notes=f"Real run in code: target {fmt(TARGET)}, range 1 to {fmt(N)} (one billion). Binary search sequence of guesses (midpoints): {[s[2] for s in STEPS]}. Found in {nlast} guesses. Worst case for n={fmt(N)} is ceil(log2(n+1)) = {worst} guesses (standard result for binary search). A one-by-one linear search would need up to {fmt(N)} guesses in the worst case (this run's target would need {fmt(TARGET)} if scanning from 1 upward); I did not literally run a billion-step linear search in the video, the comparison figure is the standard worst-case bound, stated as such in the voiceover.\nCS fact (binary search complexity): standard algorithms result (Knuth, The Art of Computer Programming vol. 3), not separately re-verified by web search this session since it is a well-established, non-time-sensitive fact.")
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
