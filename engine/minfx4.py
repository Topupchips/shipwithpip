"""Binance hack news reel (minimal dark).  usage: python minfx4.py binance out.mp4 --model <kokoro dir> [--stills png]"""
import argparse
import os
import sys

from PIL import ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa: E402
from minfx import GOLD, ICE, SOFT, WHITE, MinReel  # noqa: E402
from minfx2 import GREEN, RED, finish_reel, mkcfg  # noqa: E402
from scenes import render  # noqa: E402
from simfx import F_BOLD, F_MONO, F_SEMI, W, clamp, font  # noqa: E402

T_FROZE, T_KNOW = 12.8, 22.4


class BinanceReel(MinReel):
    def bars(self, d, t):
        x0, x1 = 110, W - 110
        scale = (x1 - x0) / 6.5
        k = clamp((t - 6.6) / 2.5)
        k2 = clamp((t - 1.0) / 3.0)
        rows = [("BYBIT  (FEB 2025)", 1.5 * k, 1.5, ICE), ("BINANCE  (OCT 2026)", 6.3 * k2, 6.3, RED)]
        y = 700
        for name, v, full, col in rows:
            d.text((x0, y - 40), name, font=font(F_SEMI, 32), fill=WHITE + (255,), anchor="lm")
            d.rounded_rectangle([x0, y, x1, y + 60], radius=16, fill=(255, 255, 255, 22))
            if v > 0.02:
                d.rounded_rectangle([x0, y, x0 + v * scale, y + 60], radius=16, fill=col + (255,))
            d.text((x1, y - 40), f"${v:0.1f}B", font=font(F_BOLD, 48), fill=col + (255,), anchor="rm")
            y += 190
        if t > 10.4:
            a = int(255 * clamp((t - 10.4) / 0.6))
            d.text((W / 2, 1110), "ABOUT 4X THE PREVIOUS RECORD", font=font(F_SEMI, 36), fill=GOLD + (a,), anchor="mm")

    def frozen(self, d, t):
        k = clamp((t - T_FROZE) / 4.0)
        x0, x1, y = 110, W - 110, 820
        d.text((W / 2, 690), "AFFECTED  $6.3B", font=font(F_BOLD, 60), fill=WHITE + (255,), anchor="mm")
        d.rounded_rectangle([x0, y, x1, y + 90], radius=22, fill=(255, 255, 255, 22))
        w = (x1 - x0) * (3.0 / 6.3) * k
        d.rounded_rectangle([x0, y, x0 + max(24, w), y + 90], radius=22, fill=GREEN + (255,))
        d.text((W / 2, y + 150), f"${3.0 * k:0.1f}B FROZEN", font=font(F_BOLD, 80), fill=GREEN + (255,), anchor="mm")
        if t > T_FROZE + 3:
            a = int(255 * clamp((t - T_FROZE - 3) / 0.6))
            d.text((W / 2, y + 260), "TETHER: $1.56B+ FROZEN", font=font(F_SEMI, 36), fill=WHITE + (a,), anchor="mm")
            d.text((W / 2, y + 320), "CIRCLE: 800M+ USDC FROZEN", font=font(F_SEMI, 36), fill=WHITE + (a,), anchor="mm")

    def know(self, d, t):
        s = t - T_KNOW
        items = [("BINANCE SAYS", GREEN, ["Users covered by SAFU fund", "No evidence private keys leaked", "Withdrawals resume in phases"]),
                 ("REPORTED, NOT CONFIRMED", GOLD, ["Engineer laptop compromised (early)", "TRM Labs: North Korea's TraderTraitor", "Binance hasn't named attackers"])]
        y = 660
        for hi, (head, col, rows) in enumerate(items):
            a0 = clamp((s - hi * 6.4) / 0.6)
            if a0 <= 0:
                continue
            d.text((110, y), head, font=font(F_BOLD, 44), fill=col + (int(255 * a0),), anchor="lm")
            for i, r in enumerate(rows):
                a = clamp((s - hi * 6.4 - 0.5 - i * 0.9) / 0.5)
                d.text((110, y + 70 + i * 62), r, font=font(F_SEMI, 34), fill=WHITE + (int(255 * a),), anchor="lm")
            y += 360

    def frame(self, t):
        img = self.new_frame(t)
        d = ImageDraw.Draw(img, "RGBA")
        if t < T_FROZE:
            self.bars(d, t)
            lines, tag = [("$6.3 BILLION", RED), ("BINANCE HACK", WHITE)], "DEVELOPING STORY - OCT 31"
        elif t < T_KNOW:
            self.frozen(d, t)
            lines, tag = [("ABOUT HALF", GREEN), ("ALREADY FROZEN", WHITE)], "DEVELOPING STORY - OCT 31"
        else:
            self.know(d, t)
            lines, tag = [("WHAT WE KNOW", WHITE), ("AND DON'T", GOLD)], "DEVELOPING STORY - OCT 31"
        self.title(d, lines, tag)
        d.text((W / 2, 1480), "FIGURES ARE BINANCE'S ESTIMATES", font=font(F_MONO, 28), fill=SOFT + (255,), anchor="mm")
        d.text((W / 2, 1530), "NOT FINANCIAL ADVICE", font=font(F_MONO, 26), fill=SOFT + (200,), anchor="mm")
        return self.handle(img, t)


def build_binance(model):
    cfg = mkcfg([0.4, 4, 8, T_FROZE + 5, T_KNOW + 3, T_KNOW + 9], 241)
    lines = [(0.2, "Binance says about six point three billion dollars in tokens was hit in a hack."),
             (4.8, "That would be the biggest crypto theft on record. The old record was Bybit in February 2025, about one point five billion."),
             (T_FROZE, "Binance says it has frozen about three billion. Tether froze over one point five six billion, and Circle over eight hundred million."),
             (T_KNOW, "Binance says user funds are safe, losses are covered by its safety fund, and there's no sign private keys leaked."),
             (T_KNOW + 6.2, "Early findings point to a hacked engineer laptop. Analysts at TRM Labs link it to North Korea's TraderTraitor group. Binance hasn't confirmed that."),
             (T_KNOW + 15.0, "Developing story. The numbers could change. This is not financial advice.")]
    reel, buf = finish_reel(cfg, lines, model, BinanceReel, 4.0, 262)
    post = dict(title="NEWS  Binance says $6.3B hit in hack (developing)",
                caption="Developing story (as of Oct 31, 2026). Binance says about $6.3 billion in token assets were affected in a hack. If confirmed, that would be the largest crypto theft on record, about 4x the roughly $1.5B Bybit hack in Feb 2025. Binance says it has frozen about $3B with partners (Tether says $1.56B+, Circle 800M+ USDC), will cover losses from its SAFU fund, found no evidence private keys leaked, and will resume withdrawals in phases. Early findings point to a compromised engineer laptop; TRM Labs links it to North Korea's TraderTraitor group, which Binance hasn't confirmed. Numbers may change. Not financial advice.",
                tags=["binance", "crypto", "cybersecurity", "news", "bitcoin", "shipwithpip"],
                pinned="This is developing and the figures are Binance's own estimates. Check Binance's official statements and a major outlet before you rely on any number here. Not financial advice.",
                notes="Facts in the video come from one article I could read in full: Forbes, 'Binance Says It Plans To Resume Withdrawals This Weekend After $6.3 Billion Hack' (Oct 30, 2026). It reports: ~$6.3B token assets affected (Binance's estimate, unconfirmed independently); would exceed Bybit's ~$1.5B (Feb 2025); preliminary cause an engineer's compromised laptop; ~$3B frozen with partners; Tether froze >$1.56B and Circle >800M USDC; no evidence private keys leaked; losses covered from SAFU and company funds; withdrawals to resume in phases (~34 hours from the statement); TRM Labs attributed it to TraderTraitor with 'high confidence', Binance did not comment on attribution. I could NOT open the CNBC or Bloomberg pieces that surfaced in search (CNBC fetch failed, Bloomberg blocks automated reads). A Bloomberg headline in search results ('Binance says 90% of $6.3 billion hacked crypto restored, withdrawals to return', Oct 31) suggests newer developments after the Forbes piece; the reel does NOT include that claim since I haven't read it. Re-check the latest before posting, as this story is moving quickly. I deliberately avoided price talk or predictions. Not financial advice.")
    return reel, buf, cfg, post


BUILDERS = {"binance": build_binance}


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
