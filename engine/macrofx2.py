"""Second batch of dark macro-explainer reels: NFC, camera sensor, fridge, smoke alarm, TV remote.
usage: python macrofx2.py nfc|cam|fridge|smoke|remote out.mp4 --model <kokoro dir> [--stills png]
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learnfx import write_post  # noqa: E402
from macrofx import (AMBER, BLUE, CX, DIM, GOLD, GRN, ICE, RED, WHT, Macro, arrow, col_a, label, lerp, mix, tiny_art)  # noqa: E402
from scenes import make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, F_SEMI, H, W, clamp, ease_out, font, smooth  # noqa: E402

COPPER = (205, 125, 70)
CPS = 12.3  # chars per second estimate for line spacing


def plan(lines, gap=1.1, first=0.2):
    """lines: list of text. returns [(start, text)] spaced by estimated speech length."""
    out, t = [], first
    for tx in lines:
        out.append((round(t, 2), tx))
        t += len(tx) / CPS + gap
    return out


def finish(cfg, L, fns, model, kw):
    cfg.update(T=300.0, seed=cfg.get("seed", 5), voice="af_heart", speed=0.98, kw=kw)
    starts = [s for s, _ in L] + [1e9]
    cfg["scenes"] = [(0 if i == 0 else starts[i], starts[i + 1], fn) for i, fn in enumerate(fns)]
    buf, vo = make_audio(cfg, L, model)
    cfg["T"] = vo[-1][1] + 2.4
    cfg["scenes"][-1] = (cfg["scenes"][-1][0], cfg["T"] + 0.5, cfg["scenes"][-1][2])
    cfg["still_times"] = [s + 2.5 for s, _ in L]
    return Macro(cfg, vo), buf


def panel(d, x0, y0, x1, y1):
    d.rounded_rectangle([x0, y0, x1, y1], radius=26, fill=(14, 16, 24, 255), outline=(70, 76, 96, 255), width=4)


def coil(d, cx, cy, w, h, col=COPPER, n=5, lw=4):
    for i in range(n):
        k = i / n
        d.rounded_rectangle([cx - w / 2 * (1 - k * 0.8), cy - h / 2 * (1 - k * 0.8), cx + w / 2 * (1 - k * 0.8), cy + h / 2 * (1 - k * 0.8)], radius=10, outline=col + (255,), width=lw)


# ===================================================================== NFC
def sc_nfc_field(self, layer, d, t, T, close=False):
    ry = 1330
    d.rounded_rectangle([160, ry, 920, ry + 150], radius=28, fill=(24, 26, 36, 255), outline=(90, 96, 116, 255), width=5)
    label(d, 180, ry + 170, "READER", 24)
    coil(d, CX, ry + 75, 360, 70)
    top = 1010 if close else lerp(780, 1010, smooth(clamp((t - 1.5) / 5.0)))
    cw, chh = 520, 280
    d.rounded_rectangle([CX - cw / 2, top, CX + cw / 2, top + chh], radius=26, fill=(18, 20, 28, 255), outline=(150, 156, 176, 255), width=5)
    coil(d, CX - 80, top + chh / 2, 260, 190, n=4)
    gap = ry - (top + chh)
    glow = math.exp(-max(gap, 0) / 170.0)
    d.rounded_rectangle([CX + 110, top + 90, CX + 210, top + 190], radius=12, fill=mix((40, 44, 58), (70, 255, 130), clamp((glow - 0.35) * 2.2)) + (255,), outline=(200, 205, 220, 255), width=3)
    label(d, CX + 160, top + 205, "CHIP", 22, anchor="ma")
    label(d, CX - cw / 2 + 4, top - 36, "CARD: NO BATTERY", 24)
    pulse = 0.5 + 0.5 * math.sin(t * 9)
    for i in range(1, 5):
        rr = 90 + 80 * i
        pts = [(CX - rr * math.cos(u), ry + 70 - rr * 0.95 * math.sin(u) * 0.8) for u in np.linspace(0, math.pi, 50)]
        d.line(pts, fill=col_a(ICE, (0.8 - 0.14 * i) * (0.35 + 0.65 * pulse)), width=4, joint="curve")
    st = "POWERED BY THE FIELD" if glow > 0.6 else "TOO FAR: NO POWER"
    d.text((CX, 660), st, font=font(F_BOLD, 44), fill=col_a(GRN if glow > 0.6 else AMBER, 1), anchor="mm")
    label(d, 60, 1640, "Simplified illustration.", 22)


def sc_nfc_wave(self, layer, d, t, T):
    x0, x1, y0 = 90, 990, 1000
    panel(d, x0 - 20, 700, x1 + 20, 1300)
    d.line([(x0, y0), (x1, y0)], fill=(90, 96, 116, 255), width=3)
    pts = [(x, y0 - 170 * math.sin((x - x0) / 900 * 2 * math.pi * 6 - 7 * t)) for x in range(x0, x1)]
    d.line(pts, fill=col_a(GOLD, 1), width=5, joint="curve")
    d.text((CX, 760), "READER FIELD", font=font(F_MONO, 28), fill=(200, 205, 220, 255), anchor="mm")
    d.text((CX, 1380), "13.56 MHz", font=font(F_BOLD, 84), fill=col_a(GOLD, 1), anchor="mm")
    label(d, CX, 1460, "13,560,000 cycles every second", 26, anchor="ma")
    label(d, 90, 1640, "Waveform slowed down enormously.", 22)


def sc_nfc_range(self, layer, d, t, T):
    x0, x1, y = 120, 960, 1180
    d.line([(x0, y), (x1, y)], fill=(120, 124, 138, 255), width=4)
    for cm in range(0, 11):
        x = lerp(x0, x1, cm / 10)
        d.line([(x, y - 12), (x, y + 12)], fill=(120, 124, 138, 255), width=3)
        d.text((x, y + 40), str(cm), font=font(F_MONO, 24), fill=(180, 184, 198, 255), anchor="mm")
    d.text((CX, y + 100), "DISTANCE (cm)", font=font(F_MONO, 24), fill=(150, 154, 168, 255), anchor="mm")
    dist = 10 * smooth(clamp((t - 1) / 6.0))
    px = lerp(x0, x1, dist / 10)
    d.rounded_rectangle([px - 40, y - 150, px + 40, y - 30], radius=12, fill=(18, 20, 28, 255), outline=(200, 205, 220, 255), width=4)
    d.rounded_rectangle([x0 - 50, y - 130, x0 - 10, y - 20], radius=8, fill=col_a(COPPER, 1))
    label(d, x0 - 30, y - 160, "READER", 22, anchor="ma")
    sig = math.exp(-(dist / 4.5) ** 3)
    bw = 760
    d.rounded_rectangle([CX - bw / 2, 700, CX + bw / 2, 760], radius=24, outline=(120, 124, 138, 255), width=3)
    d.rounded_rectangle([CX - bw / 2 + 5, 705, CX - bw / 2 + 5 + (bw - 10) * sig, 755], radius=20, fill=col_a(mix(RED, GRN, sig), 1))
    d.text((CX, 660), "FIELD STRENGTH AT THE CARD", font=font(F_MONO, 26), fill=(200, 205, 220, 255), anchor="mm")
    d.text((CX, 860), "Best under 4 cm. Usually under 10 cm.", font=font(F_BOLD, 40), fill=col_a(WHT, 1), anchor="mm")
    label(d, 60, 1640, "Fall-off curve is illustrative. Ranges per NFC guidance.", 22)


def sc_nfc_reply(self, layer, d, t, T):
    x0, x1, y0 = 80, 1000, 960
    panel(d, x0 - 20, 700, x1 + 20, 1260)
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    nb = len(bits)
    seg = (x1 - x0) / nb
    pts = []
    on = clamp((t - 1.0) / 1.0)
    for x in range(x0, x1):
        k = min(int((x - x0) / seg), nb - 1)
        env = 1 + 0.16 * on * (1 if bits[k] else -1) * 0.5 + 0.0
        pts.append((x, y0 - 150 * env * math.sin((x - x0) * 0.42 - 12 * t)))
    d.line(pts, fill=col_a(GOLD, 1), width=4, joint="curve")
    for k, b in enumerate(bits):
        d.text((x0 + seg * (k + 0.5), 1180), str(b), font=font(F_MONO, 44), fill=col_a(WHT if on > 0.5 else DIM, 1), anchor="mm")
    d.text((CX, 740), "FIELD WITH A TINY WOBBLE", font=font(F_MONO, 28), fill=(200, 205, 220, 255), anchor="mm")
    d.text((CX, 1340), "THE CARD ANSWERS BY LOADING THE FIELD", font=font(F_BOLD, 38), fill=col_a(WHT, 1), anchor="mm")
    d.text((CX, 1400), "The reader senses the change", font=font(F_SEMI, 32), fill=(200, 205, 220, 255), anchor="mm")
    label(d, 80, 1640, "Wobble exaggerated so you can see it.", 22)


def sc_nfc_pay(self, layer, d, t, T):
    d.rounded_rectangle([140, 1250, 940, 1420], radius=28, fill=(24, 26, 36, 255), outline=(90, 96, 116, 255), width=5)
    label(d, 160, 1440, "PAYMENT TERMINAL", 24)
    d.rounded_rectangle([CX - 150, 800, CX + 150, 1180], radius=40, fill=(18, 20, 28, 255), outline=(200, 205, 220, 255), width=5)
    d.rounded_rectangle([CX - 110, 880, CX + 110, 1010], radius=16, fill=(40, 70, 140, 255), outline=(150, 190, 255, 255), width=3)
    d.text((CX, 945), "CARD", font=font(F_BOLD, 40), fill=(255, 255, 255, 255), anchor="mm")
    for i in range(3):
        u = (t * 0.6 + i / 3) % 1
        y = lerp(1230, 1190, u)
        rr = 60 + 90 * u
        d.arc([CX - rr, 1250 - rr * 0.5, CX + rr, 1250 + rr * 0.5], 180, 360, fill=col_a(ICE, 0.8 * (1 - u)), width=4)
    d.text((CX, 660), "YOUR PHONE PLAYS A CARD", font=font(F_BOLD, 50), fill=col_a(WHT, 1), anchor="mm")
    d.text((CX, 730), "card emulation mode", font=font(F_MONO, 28), fill=(200, 205, 220, 255), anchor="mm")
    label(d, 60, 1640, "Illustrative.", 22)


def build_nfc(model):
    T = ["You tap a card on a reader and it works, yet the card has no battery. How?",
         "The reader makes a magnetic field that switches 13.56 million times every second.",
         "That field passes through a tiny coil in the card and pushes electrons, just enough to power a small chip.",
         "It only works up close. The field fades fast with distance, so NFC is built for a few centimeters, under ten at most.",
         "To reply, the card slightly loads the field, and the reader senses that tiny change. That is its answer.",
         "When you pay with a phone, it simply pretends to be a contactless card. Comment TAP for more."]
    L = plan(T)
    cfg = dict(seed=51)
    reel, buf = finish(cfg, L, [sc_nfc_field, sc_nfc_wave, lambda s, l, d, t, T_: sc_nfc_field(s, l, d, t, T_, True), sc_nfc_range, sc_nfc_reply, sc_nfc_pay], model,
                       ["battery", "magnetic", "field", "coil", "chip", "centimeters", "reply", "loads", "card", "phone", "contactless", "13.56", "million"])
    post = dict(title="MACRO-5  how a card with no battery works (NFC)",
                caption="A contactless card has no battery. The reader makes a 13.56 MHz magnetic field, which induces a current in the card's tiny coil and powers its chip. It only works at short range, optimally under about 4 cm and typically under 10 cm. The card replies by slightly loading the reader's field (load modulation), and the reader senses that change. When you pay with a phone, it uses card emulation to act like a contactless card.\n\nComment TAP for the next one.",
                tags=["science", "physics", "howthingswork", "nfc", "tapandpay", "tech", "learnontiktok", "shipwithpip"],
                pinned="Comment TAP and tell me which everyday gadget to break down next.",
                notes="Verified (EverythingRF, What is NFC): NFC runs at 13.56 MHz; range typically less than 10 cm with best performance below 4 cm; a passive tag has no power source and draws energy from the reader's RF field, responding by modulating that field; payments use card emulation mode. Visuals: field loops, the field-strength fall-off bar and the wobble depth are illustrations (wobble exaggerated). 13.56 MHz = 13.56 million cycles per second (arithmetic).")
    return reel, buf, cfg, post


# ===================================================================== camera sensor
def bayer_img(z, cx_img, cy_img, wpx=960, hpx=1160, n=40, base_px=16):
    art = np.asarray(tiny_art(n), np.float32)
    ps = base_px * z
    ys, xs = np.mgrid[0:hpx, 0:wpx]
    u = (xs - wpx / 2) / ps + cx_img
    v = (ys - hpx / 2) / ps + cy_img
    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
    inside = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
    iuc, ivc = np.clip(iu, 0, n - 1), np.clip(iv, 0, n - 1)
    pix = art[ivc, iuc]
    fu, fv = u - iu, v - iv
    ch = np.where((iuc % 2 == 0) & (ivc % 2 == 0), 0, np.where((iuc % 2 == 1) & (ivc % 2 == 1), 2, 1))
    mos = np.zeros_like(pix)
    for c in range(3):
        mos[..., c] = np.where(ch == c, pix[..., c], 0)
    m = np.clip((ps - 22) / 20, 0, 1)
    gapm = ((fu > 0.06) & (fu < 0.94) & (fv > 0.06) & (fv < 0.94)) | (m < 0.2)
    out = pix * (1 - m) + mos * 1.3 * gapm[..., None] * m
    out = np.where(inside[..., None], out, 6)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def sc_cam_zoom(self, layer, d, t, T):
    u = smooth(clamp((t - 1.0) / (T - 2.5)))
    z = math.exp(lerp(math.log(1.0), math.log(18.0), u))
    layer.paste(bayer_img(z, lerp(20, 24, u), lerp(20, 17, u)).convert("RGBA"), (60, 420))
    d.rectangle([60, 420, 1020, 1580], outline=(80, 86, 104, 255), width=5)
    d.text((CX, 1650), "EACH PIXEL SEES ONE COLOR" if u > 0.6 else "A PHOTO", font=font(F_BOLD, 44), fill=col_a(WHT, 1), anchor="mm")
    label(d, 60, 1590, "Magnified illustration of a Bayer color filter mosaic.", 22)


def sc_cam_tile(self, layer, d, t, T):
    cells = [(0, 0, RED, "R"), (1, 0, GRN, "G"), (0, 1, GRN, "G"), (1, 1, BLUE_, "B")]
    s = 300
    x0, y0 = CX - s, 700
    for (cx_, cy_, col, nm) in cells:
        a = smooth(clamp((t - 0.3 * (cx_ + 2 * cy_)) / 0.8))
        d.rounded_rectangle([x0 + cx_ * s + 6, y0 + cy_ * s + 6, x0 + (cx_ + 1) * s - 6, y0 + (cy_ + 1) * s - 6], radius=18, fill=col_a(col, 0.9 * a))
        d.text((x0 + cx_ * s + s / 2, y0 + cy_ * s + s / 2), nm, font=font(F_BOLD, 120), fill=(10, 10, 14, int(255 * a)), anchor="mm")
    a = smooth(clamp((t - 2.0) / 1.0))
    d.text((CX, 1380), "1 RED   2 GREEN   1 BLUE", font=font(F_BOLD, 46), fill=col_a(WHT, a), anchor="mm")
    bx0, bx1, by = 160, 920, 1470
    w = bx1 - bx0
    d.rounded_rectangle([bx0, by, bx0 + w * 0.25, by + 50], radius=12, fill=col_a(RED, a))
    d.rectangle([bx0 + w * 0.25, by, bx0 + w * 0.75, by + 50], fill=col_a(GRN, a))
    d.rounded_rectangle([bx0 + w * 0.75, by, bx1, by + 50], radius=12, fill=col_a(BLUE_, a))
    d.text((bx0 + w * 0.125, by + 90), "25%", font=font(F_MONO, 30), fill=col_a(WHT, a), anchor="mm")
    d.text((bx0 + w * 0.5, by + 90), "50%", font=font(F_MONO, 30), fill=col_a(WHT, a), anchor="mm")
    d.text((bx0 + w * 0.875, by + 90), "25%", font=font(F_MONO, 30), fill=col_a(WHT, a), anchor="mm")
    label(d, 60, 1640, "The repeating 2x2 Bayer pattern.", 22)


def sc_cam_demo(self, layer, d, t, T):
    n, cs, x0, y0 = 5, 150, CX - 375, 700
    for r in range(n):
        for c in range(n):
            col = RED if (c % 2 == 0 and r % 2 == 0) else (BLUE_ if (c % 2 == 1 and r % 2 == 1) else GRN)
            ctr = (r == 2 and c == 2)
            d.rounded_rectangle([x0 + c * cs + 6, y0 + r * cs + 6, x0 + (c + 1) * cs - 6, y0 + (r + 1) * cs - 6], radius=10, fill=col_a(col, 0.95 if ctr else 0.42))
    a = smooth(clamp((t - 1.0) / 1.0))
    cxm, cym = x0 + 2.5 * cs, y0 + 2.5 * cs
    for (dx, dy) in ((-cs, 0), (cs, 0), (0, -cs), (0, cs), (-cs, -cs), (cs, cs)):
        arrow(d, (cxm + dx * 0.85, cym + dy * 0.85), (cxm + dx * 0.3, cym + dy * 0.3), col_a(WHT, 0.8 * a), w=4, head=14)
    d.text((cxm, cym), "R", font=font(F_BOLD, 64), fill=(10, 10, 14, 255), anchor="mm")
    d.text((CX, 600), "THE MISSING COLORS ARE GUESSED", font=font(F_BOLD, 44), fill=col_a(WHT, 1), anchor="mm")
    b = smooth(clamp((t - 4.0) / 1.0))
    d.text((CX, 1530), "MEASURED: 1 of 3 values", font=font(F_MONO, 34), fill=col_a(GRN, b), anchor="mm")
    d.text((CX, 1590), "ESTIMATED: 2 of 3 values", font=font(F_MONO, 34), fill=col_a(AMBER, b), anchor="mm")
    label(d, 60, 1660, "Called demosaicing. Real phone sensors use variations of this layout.", 22)


BLUE_ = (80, 130, 255)
GRN = GRN
RED = RED


def build_cam(model):
    T = ["Your phone camera takes color photos, but its sensor is actually color blind.",
         "Each tiny light sensor sits under a filter, so each one only measures red, green, or blue.",
         "In the classic Bayer layout, every group of four has one red, two green, and one blue. Half of all the pixels are green.",
         "So every pixel knows just one of its three colors. The other two are guessed from its neighbors.",
         "That means two out of every three color values in your photo are estimated by software. Comment SENSOR for more."]
    L = plan(T)
    cfg = dict(seed=61)
    reel, buf = finish(cfg, L, [sc_cam_zoom, sc_cam_zoom, sc_cam_tile, sc_cam_demo, sc_cam_demo], model,
                       ["color", "blind", "red", "green", "blue", "filter", "bayer", "half", "guessed", "neighbors", "estimated", "software", "sensor"])
    post = dict(title="MACRO-6  your phone camera is color blind",
                caption="A camera sensor measures brightness, not color. A color filter array (the classic Bayer pattern) puts a red, green or blue filter over each photosite, in repeating groups of four with one red, two green and one blue, so half the pixels are green. Each pixel therefore records only one of its three color values, and software (demosaicing) estimates the other two from neighboring pixels. That means roughly two thirds of the color values in a basic Bayer photo are interpolated. Many modern phone sensors use variations on this layout.\n\nComment SENSOR for the next one.",
                tags=["science", "photography", "howthingswork", "iphone", "camera", "tech", "learnontiktok", "shipwithpip"],
                pinned="Comment SENSOR and tell me which everyday gadget to break down next.",
                notes="Verified (Bayer filter summary page): 2x2 grid of color filters, 50% green / 25% red / 25% blue (2:1:1), each pixel detects only one color, demosaicing algorithms interpolate the missing colors; invented by Bryce Bayer. The 2 of 3 figure is arithmetic for a basic Bayer sensor (1 measured value out of 3 per pixel); modern phone sensors may use variations such as quad layouts, said in the caption. The magnified mosaic is a computed illustration from a small generated image. I did not verify or claim why green is doubled.")
    return reel, buf, cfg, post


# ===================================================================== fridge
LOOP = [(260, 1380), (820, 1380), (820, 820), (260, 820), (260, 1380)]
SEGC = [(RED, "COMPRESSOR"), (AMBER, ""), (ICE, ""), (BLUE_, "")]


def loop_pos(u):
    segs = [(LOOP[i], LOOP[i + 1]) for i in range(4)]
    lens = [math.dist(a, b) for a, b in segs]
    tot = sum(lens)
    q = (u % 1) * tot
    for (a, b), ln, k in zip(segs, lens, range(4)):
        if q <= ln:
            return (lerp(a[0], b[0], q / ln), lerp(a[1], b[1], q / ln)), k
        q -= ln
    return LOOP[0], 0


def sc_fridge(self, layer, d, t, T, focus=None, heat=False):
    # zones
    d.rectangle([60, 700, 540, 1500], fill=(20, 40, 90, 70))
    d.rectangle([540, 700, 1020, 1500], fill=(120, 50, 20, 60))
    label(d, 80, 720, "INSIDE THE FRIDGE", 24, (150, 190, 255))
    label(d, 1000, 720, "YOUR KITCHEN", 24, (255, 180, 140), anchor="ra")
    segcol = [RED, AMBER, ICE, BLUE_]
    # path: bottom edge (evap exit->compressor) is cool gas; right (compressor->condenser) hot gas; top (post condenser) warm liquid; left (post valve) cold
    cols = [BLUE_, RED, AMBER, ICE]  # per segment index: bottom, right, top, left
    pcols = [ICE, RED, AMBER, BLUE_]
    for i in range(4):
        a, b = LOOP[i], LOOP[i + 1]
        d.line([a, b], fill=col_a(pcols[i], 0.45), width=16)
    for i in range(24):
        pos, k = loop_pos((t * 0.07 + i / 24))
        c = pcols[k]
        d.ellipse([pos[0] - 9, pos[1] - 9, pos[0] + 9, pos[1] + 9], fill=col_a(c, 1))
    comps = {"compressor": (820, 1380, "COMPRESSOR", RED), "condenser": (820, 1100, "CONDENSER", AMBER), "valve": (540, 820, "EXPANSION VALVE", ICE), "evaporator": (260, 1100, "EVAPORATOR", BLUE_)}
    for key, (x, y, nm, c) in comps.items():
        on = focus is None or focus == key
        al = 1.0 if on else 0.35
        if key == "valve":
            d.polygon([(x - 34, y - 28), (x + 34, y + 28), (x + 34, y - 28), (x - 34, y + 28)], fill=col_a(c, al))
        else:
            d.rounded_rectangle([x - 52, y - 70 if key != "compressor" else y - 36, x + 52, y + 70 if key != "compressor" else y + 36], radius=16, fill=col_a(c, 0.25 * al + 0.1), outline=col_a(c, al), width=5)
        tx = x + (0 if key in ("valve",) else 0)
        anch = "mm"
        d.text((x, y - 100 if key in ("valve", "condenser", "evaporator") else y + 70), nm, font=font(F_MONO, 24), fill=col_a(WHT, al), anchor=anch)
    if heat:
        a = smooth(clamp(t / 1.2))
        for k in range(3):
            y = 1010 + k * 90
            arrow(d, (110, y), (190, y), col_a(RED, 0.9 * a), w=7, head=20)
            arrow(d, (880, y), (970, y), col_a(AMBER, 0.9 * a), w=7, head=20)
        d.text((CX, 1560), "HEAT MOVES OUT. COLD IS JUST HEAT REMOVED.", font=font(F_BOLD, 34), fill=col_a(WHT, a), anchor="mm")
    d.text((CX, 620), "A HEAT PUMP IN A BOX", font=font(F_BOLD, 50), fill=col_a(WHT, 1), anchor="mm")
    label(d, 60, 1640, "Simplified cycle. Colors show relative temperature.", 22)


def build_fridge(model):
    T = ["Your fridge does not make cold. It moves heat out of the box.",
         "Inside is a loop of refrigerant. A compressor squeezes it into a hot, high pressure gas.",
         "That gas flows through coils at the back, the condenser, where it dumps its heat into your kitchen and turns into a liquid.",
         "Then an expansion valve drops the pressure, and the liquid turns very cold.",
         "Cold refrigerant runs through the evaporator inside the fridge, soaks up heat from your food, and boils into a gas. Then it goes back to the compressor and repeats.",
         "That is why the back of a fridge feels warm. Comment FRIDGE for more."]
    L = plan(T)
    cfg = dict(seed=71)
    F = lambda f, h=False: (lambda s, l, d, t, T_: sc_fridge(s, l, d, t, T_, f, h))
    reel, buf = finish(cfg, L, [F(None), F("compressor"), F("condenser"), F("valve"), F("evaporator"), F(None, True)], model,
                       ["heat", "refrigerant", "compressor", "condenser", "valve", "evaporator", "cold", "warm", "gas", "liquid", "kitchen", "back"])
    post = dict(title="MACRO-7  your fridge does not make cold",
                caption="A fridge moves heat instead of creating cold. A compressor squeezes refrigerant into a hot, high pressure gas. It flows through the condenser coils at the back and releases heat into the room, turning into a liquid. An expansion valve drops the pressure so the liquid becomes very cold, and in the evaporator inside the fridge it absorbs heat from the food and evaporates into gas. The cycle repeats, which is why the back of a fridge feels warm.\n\nComment FRIDGE for the next one.",
                tags=["science", "physics", "howthingswork", "fridge", "thermodynamics", "tech", "learnontiktok", "shipwithpip"],
                pinned="Comment FRIDGE and tell me which everyday gadget to break down next.",
                notes="Verified (ScienceABC, how a refrigerator works): the four components (compressor, condenser, expansion valve, evaporator) and refrigerant phases (hot high-pressure gas after the compressor, high-pressure liquid after the condenser, cold low-pressure liquid after the valve, evaporating to gas in the evaporator); heat released through the condenser at the back; a fridge moves heat rather than creating cold. Visual notes: layout and colors are a simplified diagram, not a real fridge's plumbing.")
    return reel, buf, cfg, post


# ===================================================================== smoke alarm
def sc_smoke(self, layer, d, t, T, phase=0):
    x0, x1, yt, yb = 150, 930, 820, 1300
    d.rounded_rectangle([x0, yt - 24, x1, yt], radius=8, fill=(110, 116, 140, 255))
    d.rounded_rectangle([x0, yb, x1, yb + 24], radius=8, fill=(110, 116, 140, 255))
    d.text((x0 + 6, yt - 60), "+", font=font(F_BOLD, 40), fill=col_a(RED, 1))
    d.text((x0 + 6, yb + 64), "−", font=font(F_BOLD, 40), fill=col_a(BLUE_, 1))
    label(d, x1, yt - 56, "PLATE", 22, anchor="ra")
    # source
    d.rounded_rectangle([CX - 70, 1020, CX + 70, 1080], radius=10, fill=col_a((200, 170, 60), 1), outline=(255, 235, 150, 255), width=3)
    d.text((CX, 1050), "Am-241", font=font(F_MONO, 26), fill=(20, 16, 6, 255), anchor="mm")
    rs = np.random.default_rng(5)
    smoke = 0.0 if phase < 3 else (smooth(clamp(t / 4.0)) if phase == 3 else 1.0)
    # air molecules
    for i in range(46):
        bx, by = rs.uniform(x0 + 20, x1 - 20), rs.uniform(yt + 20, yb - 20)
        d.ellipse([bx - 6 + 4 * math.sin(t * 2 + i), by - 6, bx + 6 + 4 * math.sin(t * 2 + i), by + 6], fill=(120, 126, 150, 140))
    flow = 1.0 - 0.62 * smoke
    if phase >= 1:
        for i in range(40):
            u = (t * 0.55 + i * 0.137) % 1
            # alpha particles flying outward from source
            ang = rs.uniform(0, 2 * math.pi)
            if i % 4 == 0:
                r = 40 + 360 * u
                d.ellipse([CX + r * math.cos(ang) - 7, 1050 + r * math.sin(ang) * 0.9 - 7, CX + r * math.cos(ang) + 7, 1050 + r * math.sin(ang) * 0.9 + 7], fill=col_a(GOLD, 1 - u))
        for i in range(36):
            sx = rs.uniform(x0 + 120, x1 - 120)
            u = (t * 0.5 * flow + rs.uniform(0, 1)) % 1
            if rs.uniform() > flow + 0.15:
                continue
            d.ellipse([sx - 7, lerp(1050, yt + 10, u) - 7, sx + 7, lerp(1050, yt + 10, u) + 7], fill=col_a(BLUE_, 0.9))
            d.ellipse([sx + 22 - 7, lerp(1050, yb - 10, u) - 7, sx + 22 + 7, lerp(1050, yb - 10, u) + 7], fill=col_a(RED, 0.9))
    # smoke
    if smoke > 0:
        for i in range(40):
            sx = rs.uniform(x0 + 30, x1 - 30)
            sy = lerp(yb - 10, rs.uniform(yt + 40, yb - 40), smoke) + 14 * math.sin(t + i)
            r = rs.uniform(20, 46) * smoke
            d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(150, 140, 130, int(120 * smoke)))
    # meter
    if phase >= 2:
        bw = 760
        cur = flow
        d.text((CX, 1470), "CURRENT BETWEEN THE PLATES", font=font(F_MONO, 24), fill=(200, 205, 220, 255), anchor="mm")
        d.rounded_rectangle([CX - bw / 2, 1500, CX + bw / 2, 1550], radius=22, outline=(120, 124, 138, 255), width=3)
        d.rounded_rectangle([CX - bw / 2 + 5, 1505, CX - bw / 2 + 5 + (bw - 10) * cur, 1545], radius=18, fill=col_a(GRN if cur > 0.7 else RED, 1))
        tx = CX - bw / 2 + 5 + (bw - 10) * 0.7
        d.line([(tx, 1490), (tx, 1560)], fill=col_a(WHT, 1), width=4)
        label(d, tx, 1566, "ALARM LEVEL", 20, anchor="ma")
        if cur < 0.7 and phase == 4:
            fl = 0.5 + 0.5 * math.sin(t * 12)
            d.text((CX, 680), "ALARM!", font=font(F_BOLD, 120), fill=col_a(RED, 0.6 + 0.4 * fl), anchor="mm")
    if phase < 4:
        d.text((CX, 680), "SMOKE ALARM CHAMBER", font=font(F_BOLD, 50), fill=col_a(WHT, 1), anchor="mm")
    label(d, 60, 1640, "Ionization-type alarm. Simplified, not to scale.", 22)


def build_smoke(model):
    T = ["A smoke alarm can sense smoke you cannot even see. It does it with a tiny bit of radioactive metal.",
         "Inside is americium 241. It shoots out alpha particles that knock electrons off air molecules, making charged ions.",
         "A small voltage pulls those ions across the gap, so a weak steady current flows.",
         "When smoke drifts in, the particles disrupt that flow, and the current drops.",
         "When the current falls below a set level, the alarm sounds. Comment SMOKE for more."]
    L = plan(T)
    cfg = dict(seed=81)
    S = lambda p: (lambda s, l, d, t, T_: sc_smoke(s, l, d, t, T_, p))
    reel, buf = finish(cfg, L, [S(0), S(1), S(2), S(3), S(4)], model,
                       ["radioactive", "americium", "alpha", "ions", "current", "smoke", "drops", "alarm", "gap", "voltage"])
    post = dict(title="MACRO-8  how a smoke alarm senses smoke you cannot see",
                caption="Ionization smoke alarms contain a tiny amount of americium-241, which emits alpha particles. They knock electrons off air molecules in a small chamber, making ions, and a small voltage pulls those ions between two electrodes so a weak steady current flows. Smoke particles disrupt that current. When it falls below a set threshold, the alarm sounds. This design can detect smoke particles too small to see.\n\nComment SMOKE for the next one.",
                tags=["science", "physics", "howthingswork", "smokealarm", "safety", "tech", "learnontiktok", "shipwithpip"],
                pinned="Comment SMOKE and tell me which everyday gadget to break down next.",
                notes="Verified (American Nuclear Society smoke detector page): ionization alarms contain americium-241; alpha particles ionize air between two electrodes, giving a weak steady current; smoke disrupts the current, and when it drops below a threshold the alarm sounds; can detect particles too small to see. Not every smoke alarm is ionization type (photoelectric alarms work differently); the video says nothing about that. Visuals are a simplified, not-to-scale illustration.")
    return reel, buf, cfg, post


# ===================================================================== TV remote
def sc_rm_remote(self, layer, d, t, T):
    d.rounded_rectangle([CX - 120, 1000, CX + 120, 1560], radius=44, fill=(20, 22, 30, 255), outline=(150, 156, 176, 255), width=5)
    for r in range(4):
        for c in range(3):
            d.ellipse([CX - 70 + c * 70 - 20, 1160 + r * 90 - 20, CX - 70 + c * 70 + 20, 1160 + r * 90 + 20], fill=(46, 50, 66, 255))
    d.ellipse([CX - 14, 1018, CX + 14, 1046], fill=(90, 30, 110, 255), outline=(190, 120, 255, 255), width=3)
    for i in range(6):
        u = (t * 0.8 + i / 6) % 1
        r = 40 + 560 * u
        d.arc([CX - r, 1032 - r, CX + r, 1032 + r], 250, 290, fill=(190, 120, 255, int(210 * (1 - u))), width=5)
    d.text((CX, 620), "A FLASHLIGHT YOU CANNOT SEE", font=font(F_BOLD, 50), fill=col_a(WHT, 1), anchor="mm")
    d.text((CX, 700), "infrared light", font=font(F_MONO, 30), fill=(200, 205, 220, 255), anchor="mm")
    label(d, 60, 1640, "Illustration: the real light is invisible to your eyes.", 22)


def sc_rm_code(self, layer, d, t, T, zoom=False):
    x0, x1, y0 = 70, 1010, 1100
    panel(d, x0 - 20, 700, x1 + 20, 1400)
    bursts = [(0.0, 0.12), (0.18, 0.24), (0.30, 0.42), (0.48, 0.54), (0.60, 0.66), (0.72, 0.84), (0.90, 0.96)]
    for (a, b) in bursts:
        xa, xb = lerp(x0, x1, a), lerp(x0, x1, b)
        if zoom:
            pts = [(x, y0 - 110 * math.sin((x - xa) * 0.9 - 30 * t)) for x in range(int(xa), int(xb))]
            if len(pts) > 2:
                d.line(pts, fill=col_a((190, 120, 255), 1), width=3)
        else:
            d.rectangle([xa, y0 - 110, xb, y0 + 110], fill=col_a((190, 120, 255), 0.9))
    d.line([(x0, y0 + 130), (x1, y0 + 130)], fill=(90, 96, 116, 255), width=2)
    d.text((CX, 760), "BURSTS AND GAPS = A CODE" if not zoom else "INSIDE EACH BURST: ~38,000 FLICKERS A SECOND", font=font(F_MONO, 28 if not zoom else 22), fill=(200, 205, 220, 255), anchor="mm")
    if zoom:
        d.text((CX, 1470), "38 kHz", font=font(F_BOLD, 90), fill=col_a((190, 120, 255), 1), anchor="mm")
    label(d, 60, 1640, "Pulse pattern is illustrative. Real remotes use several protocols.", 22)


def sc_rm_filter(self, layer, d, t, T):
    panel(d, 50, 700, 1030, 1000)
    panel(d, 50, 1060, 1030, 1360)
    d.text((CX, 740), "SUNLIGHT AND LAMPS: STEADY", font=font(F_MONO, 26), fill=(200, 205, 220, 255), anchor="mm")
    d.line([(90, 880), (990, 880)], fill=col_a(AMBER, 1), width=8)
    d.text((CX, 960), "IGNORED", font=font(F_BOLD, 34), fill=col_a(RED, 1), anchor="mm")
    d.text((CX, 1100), "REMOTE: FLICKERS AT 38 kHz", font=font(F_MONO, 26), fill=(200, 205, 220, 255), anchor="mm")
    pts = [(x, 1220 - 70 * math.sin(x * 0.5 - 20 * t)) for x in range(90, 990)]
    d.line(pts, fill=col_a((190, 120, 255), 1), width=3)
    d.text((CX, 1320), "HEARD", font=font(F_BOLD, 34), fill=col_a(GRN, 1), anchor="mm")
    d.text((CX, 620), "THE TV ONLY LISTENS FOR THE FLICKER", font=font(F_BOLD, 38), fill=col_a(WHT, 1), anchor="mm")
    label(d, 60, 1640, "Illustrative.", 22)


def sc_rm_cam(self, layer, d, t, T):
    d.rounded_rectangle([CX - 220, 760, CX + 220, 1500], radius=50, fill=(14, 16, 24, 255), outline=(200, 205, 220, 255), width=5)
    d.rounded_rectangle([CX - 190, 810, CX + 190, 1450], radius=22, fill=(6, 6, 10, 255))
    pr = 1 + 0.12 * math.sin(t * 8)
    if t > 1.0:
        for r in range(int(90 * pr), 10, -10):
            d.ellipse([CX - r, 1130 - r, CX + r, 1130 + r], fill=(255, 255, 255, int(255 * (1 - r / 100) ** 0.7)))
    d.text((CX, 660), "PHONE CAMERA VIEW", font=font(F_BOLD, 46), fill=col_a(WHT, 1), anchor="mm")
    d.text((CX, 1560), "The invisible light shows up as white", font=font(F_SEMI, 34), fill=(220, 225, 240, 255), anchor="mm")
    label(d, 60, 1640, "Dramatization. Some phones filter out infrared.", 22)


def build_remote(model):
    T = ["A TV remote is a tiny flashlight, but the light is infrared, so your eyes cannot see it.",
         "Press a button and an LED blinks out a code, a pattern of short bursts and gaps.",
         "Inside each burst, the light flickers about thirty eight thousand times a second.",
         "The TV receiver only listens for that flicker, so steady sunlight and lamps do not confuse it.",
         "And you can see it yourself. On many phone cameras, the invisible light shows up as a bright white glow. Comment REMOTE for more."]
    L = plan(T)
    cfg = dict(seed=91)
    reel, buf = finish(cfg, L, [sc_rm_remote, lambda s, l, d, t, T_: sc_rm_code(s, l, d, t, T_, False), lambda s, l, d, t, T_: sc_rm_code(s, l, d, t, T_, True), sc_rm_filter, sc_rm_cam], model,
                       ["flashlight", "infrared", "invisible", "led", "code", "bursts", "gaps", "flickers", "thirty", "receiver", "sunlight", "camera", "white", "glow"])
    post = dict(title="MACRO-9  your TV remote is an invisible flashlight",
                caption="A TV remote sends infrared light your eyes cannot see. Pressing a button makes an LED blink out a code as bursts and gaps, and inside each burst the light flickers at about 38 kHz. The TV's receiver is tuned to that modulation, so steady sources like sunlight and lamps do not confuse it. Many phone cameras lack a strong infrared-blocking filter, so pointing a remote at one usually shows the invisible light as a bright white glow.\n\nComment REMOTE for the next one.",
                tags=["science", "physics", "howthingswork", "tvremote", "infrared", "tech", "learnontiktok", "shipwithpip"],
                pinned="Comment REMOTE and tell me which everyday gadget to break down next.",
                notes="Verified (SparkFun IR communication basics; Adafruit IR sensor guide): IR is invisible to the eye; 38 kHz is a common modulation frequency and the receiver demodulates it, so a steady IR light is not detected; the pattern of pulses and spacing encodes data; typical detector peak sensitivity about 940 nm. Verified (Institute of Physics Spark): a mobile phone camera can show a TV remote's infrared light, usually as white; many cheap and phone cameras lack infrared-blocking filters. The video notes that some phones filter it out. The burst pattern shown is illustrative, not a real protocol's timing.")
    return reel, buf, cfg, post


BUILD = {"nfc": build_nfc, "cam": build_cam, "fridge": build_fridge, "smoke": build_smoke, "remote": build_remote}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = BUILD[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
