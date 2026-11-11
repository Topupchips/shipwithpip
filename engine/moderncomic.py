"""Modern flat-design comic reels: clean gradients, flat mascots, soft-shadow UI
cards. Same script JSON format as paint.py (cast/scene/beats/panel/rehook/
sticker/hook), reuses its TTS + music + sfx pipeline (build_audio).

usage: python moderncomic.py script.json out.mp4 --model <kokoro dir> [--stills out.png]
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import modern_characters as C  # noqa: E402
from paint import build_audio, VOICES, GAP, W, H, FPS, SR  # noqa: E402

BW, BH = 1242, 2208
INK = (30, 28, 48)
PAPER = None
_fc = {}

PALETTE = {
    "coral": (255, 107, 107), "pink": (255, 94, 150), "red": (238, 72, 90),
    "blue": (74, 134, 255), "indigo": (99, 102, 241), "green": (52, 199, 132),
    "amber": (255, 170, 60), "violet": (150, 110, 255), "slate": (71, 76, 106),
    "ink": INK,
}
F_DISPLAY = os.path.join(HERE, "assets", "Poppins-Bold.ttf")
F_HEAVY = os.path.join(HERE, "assets", "Poppins-Bold.ttf")
F_BODY = os.path.join(HERE, "assets", "InterDisplay-SemiBold.otf")
F_BODY_MED = os.path.join(HERE, "assets", "InterDisplay-Medium.otf")


def font(path, size):
    k = (path, int(size))
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, int(size))
    return _fc[k]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    return 1 - (1 - clamp(x)) ** 3


def back_out(x):
    x = clamp(x)
    c1, c3 = 1.7, 2.6
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def wrap(text, f, max_w, draw):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if draw.textlength(t, font=f) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def rounded_shadow(size, box, radius, blur=22, alpha=90, off=(0, 10)):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    x0, y0, x1, y1 = box
    ImageDraw.Draw(img).rounded_rectangle([x0 + off[0], y0 + off[1], x1 + off[0], y1 + off[1]], radius=radius, fill=(10, 8, 30, alpha))
    return img.filter(ImageFilter.GaussianBlur(blur))


# ------------------------------------------------------------------ backgrounds
def _blob(cv, cx, cy, r, col, soft):
    """Additively paint a soft radial color blob onto a float canvas (numpy)."""
    yy, xx = np.mgrid[0:cv.shape[0], 0:cv.shape[1]]
    d = np.hypot(xx - cx, yy - cy) / r
    m = np.clip(1 - d, 0, 1) ** soft
    for c in range(3):
        cv[..., c] += m * (col[c] - cv[..., c]) * 0.72


def flat_bg(kind, seed=0):
    rng = random.Random(hash((kind, seed)) % 10000)
    yy = np.linspace(0, 1, BH)[:, None, None]
    if kind == "study":  # soft modern studio
        top, bot = np.array([248, 244, 255.0]), np.array([232, 224, 248.0])
        cv = top * (1 - yy) + bot * yy
        cv = np.repeat(cv, BW, axis=1)
        _blob(cv, BW * 0.18, BH * 0.16, BW * 0.42, (255, 200, 160), 1.4)
        _blob(cv, BW * 0.85, BH * 0.30, BW * 0.36, (190, 200, 255), 1.6)
        _blob(cv, BW * 0.5, BH * 0.62, BW * 0.5, (255, 220, 235), 1.8)
        img = Image.fromarray(np.clip(cv, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(img, "RGBA")
        for gy in range(0, int(BH * 0.86), 86):
            for gx in range(70, BW - 40, 86):
                d.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], fill=(140, 120, 180, 30))
        floor_y = int(BH * 0.86)
        d.rectangle([0, floor_y, BW, BH], fill=(226, 216, 242, 255))
        d.rectangle([0, floor_y, BW, floor_y + 6], fill=(205, 190, 230, 255))
        # desk + monitor silhouette (modern flat)
        mx, my, mw, mh = BW * 0.68, floor_y - 300, 340, 220
        d.rounded_rectangle([mx, my, mx + mw, my + mh], radius=22, fill=(60, 56, 96, 255))
        d.rounded_rectangle([mx + 16, my + 16, mx + mw - 16, my + mh - 16], radius=10, fill=(150, 170, 255, 255))
        d.rectangle([mx + mw / 2 - 10, my + mh, mx + mw / 2 + 10, my + mh + 50], fill=(60, 56, 96, 255))
        d.rounded_rectangle([mx + mw / 2 - 70, my + mh + 50, mx + mw / 2 + 70, my + mh + 70], radius=10, fill=(60, 56, 96, 255))
        # plant
        px, py = BW * 0.1, floor_y - 10
        d.rounded_rectangle([px - 44, py - 70, px + 44, py], radius=14, fill=(210, 150, 110, 255))
        for i in range(6):
            a = -math.pi / 2 + (i - 2.5) * 0.3
            lx, ly = px + math.cos(a) * 90, py - 70 + math.sin(a) * 140
            d.ellipse([lx - 26, ly - 50, lx + 26, ly + 50], fill=(80, 170, 120, 255))
        return img
    if kind == "city":  # modern night skyline
        top, bot = np.array([26, 22, 58.0]), np.array([78, 54, 128.0])
        cv = top * (1 - yy) + bot * yy
        cv = np.repeat(cv, BW, axis=1)
        _blob(cv, BW * 0.78, BH * 0.14, BW * 0.5, (120, 90, 220), 1.6)
        _blob(cv, BW * 0.15, BH * 0.08, BW * 0.3, (255, 210, 160), 2.0)
        img = Image.fromarray(np.clip(cv, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(img, "RGBA")
        d.ellipse([BW * 0.68, BH * 0.07, BW * 0.68 + BW * 0.18, BH * 0.07 + BW * 0.18], fill=(255, 236, 200, 255))
        for i in range(40):
            x, y = rng.uniform(0.03, 0.97) * BW, rng.uniform(0.02, 0.33) * BH
            r = rng.uniform(2, 4)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, int(rng.uniform(120, 220))))
        floor_y = int(BH * 0.78)
        x = -20
        cols = [(46, 38, 92), (56, 46, 108), (40, 60, 120), (64, 50, 118)]
        k = 0
        while x < BW:
            bw = rng.uniform(110, 220)
            bh = rng.uniform(0.18, 0.4) * BH
            top_y = floor_y - bh
            col = cols[k % len(cols)]
            d.rounded_rectangle([x, top_y, x + bw, floor_y + 4], radius=16, fill=col + (255,))
            for wy in range(int(top_y) + 30, floor_y - 24, 46):
                for wx in range(int(x) + 18, int(x + bw) - 18, 34):
                    if rng.random() < 0.45:
                        d.rounded_rectangle([wx, wy, wx + 14, wy + 18], radius=3, fill=(255, 214, 140, 230))
            x += bw + rng.uniform(14, 30)
            k += 1
        d.rectangle([0, floor_y, BW, BH], fill=(34, 26, 64, 255))
        return img
    # "sunset" -> modern warm gradient
    top, bot = np.array([255, 198, 170.0]), np.array([160, 130, 210.0])
    cv = top * (1 - yy) + bot * yy
    cv = np.repeat(cv, BW, axis=1)
    _blob(cv, BW * 0.5, BH * 0.22, BW * 0.5, (255, 230, 170), 1.3)
    _blob(cv, BW * 0.12, BH * 0.7, BW * 0.4, (255, 160, 170), 1.7)
    _blob(cv, BW * 0.9, BH * 0.68, BW * 0.4, (140, 170, 230), 1.7)
    img = Image.fromarray(np.clip(cv, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([BW * 0.26, BH * 0.1, BW * 0.74, BH * 0.1 + BW * 0.48], fill=(255, 224, 170, 255))
    floor_y = int(BH * 0.74)
    layers = [((120, 110, 190), 0.60, 170, 0.7), ((70, 150, 158), 0.665, 130, 1.3), ((232, 172, 132), 0.73, 90, 2.1)]
    for col, yfrac, amp, ph in layers:
        yb = int(BH * yfrac)
        n = 7
        pts = [(0, BH)]
        for i in range(n + 1):
            pts.append((BW * i / n, yb - amp * math.sin(i * 0.9 + ph) - amp * 0.5))
        pts.append((BW, BH))
        d.polygon(pts, fill=col + (255,))
    d.rectangle([0, floor_y, BW, BH], fill=(236, 190, 150, 255))
    return img


def paper_tex():
    return 1.0  # flat style: no grain overlay


# ------------------------------------------------------------------ panels (modern cards)
def icon_badge(d, kind, cx, cy, r):
    if kind == "check":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PALETTE["green"] + (255,))
        d.line([(cx - r * 0.45, cy), (cx - r * 0.1, cy + r * 0.4), (cx + r * 0.5, cy - r * 0.42)], fill=(255, 255, 255, 255), width=int(r * 0.26), joint="curve")
    elif kind == "x":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PALETTE["red"] + (255,))
        d.line([(cx - r * 0.42, cy - r * 0.42), (cx + r * 0.42, cy + r * 0.42)], fill=(255, 255, 255, 255), width=int(r * 0.24))
        d.line([(cx + r * 0.42, cy - r * 0.42), (cx - r * 0.42, cy + r * 0.42)], fill=(255, 255, 255, 255), width=int(r * 0.24))
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PALETTE["indigo"] + (255,))
        f = font(F_DISPLAY, r * 1.1)
        d.text((cx, cy + r * 0.06), str(kind), font=f, fill=(255, 255, 255, 255), anchor="mm")


def panel_image(panel, prog, accent_name="indigo"):
    typ = panel["type"]
    w = 900
    accent = PALETTE.get(accent_name, PALETTE["indigo"])
    pad = 54
    if typ == "note":
        lines = panel["lines"]
        f = font(F_DISPLAY, 70)
        h = pad * 2 + len(lines) * 92 + (56 if panel.get("sub") else 0)
        canvas = Image.new("RGBA", (w + 60, h + 60), (0, 0, 0, 0))
        canvas.alpha_composite(rounded_shadow(canvas.size, (30, 30, 30 + w, 30 + h), 34, blur=26, alpha=95))
        d = ImageDraw.Draw(canvas, "RGBA")
        d.rounded_rectangle([30, 30, 30 + w, 30 + h], radius=34, fill=(255, 255, 255, 255))
        d.rounded_rectangle([30, 30, 46, 30 + h], radius=10, fill=accent + (255,))
        y = 30 + pad
        for ln in lines:
            txt, ck = (ln, "slate") if isinstance(ln, str) else ln
            col = PALETTE.get(ck, PALETTE["slate"]) if ck != "red" else PALETTE["red"]
            d.text((30 + w / 2, y + 36), txt, font=f, fill=col + (255,), anchor="mm")
            y += 92
        if panel.get("sub"):
            fs = font(F_BODY_MED, 36)
            d.text((30 + w / 2, y + 6), panel["sub"], font=fs, fill=(140, 140, 160, 255), anchor="mm")
        return canvas
    if typ == "list":
        rows = panel["rows"]
        h = pad * 2 + 76 + len(rows) * 116
        canvas = Image.new("RGBA", (w + 60, h + 60), (0, 0, 0, 0))
        canvas.alpha_composite(rounded_shadow(canvas.size, (30, 30, 30 + w, 30 + h), 34, blur=26, alpha=95))
        d = ImageDraw.Draw(canvas, "RGBA")
        d.rounded_rectangle([30, 30, 30 + w, 30 + h], radius=34, fill=(255, 255, 255, 255))
        d.text((30 + pad, 30 + pad + 10), panel.get("title", ""), font=font(F_DISPLAY, 58), fill=PALETTE["slate"] + (255,), anchor="lm")
        n = len(rows) if prog >= 1 else int(len(rows) * ease_out(prog / 0.8)) + 1
        y = 30 + pad + 100
        for r in rows[:n]:
            icon, txt = r[0], r[1]
            icon_badge(d, icon, 30 + pad + 32, y + 6, 32)
            d.text((30 + pad + 84, y - 12), txt, font=font(F_BODY, 46), fill=PALETTE["ink"] + (255,), anchor="lm")
            if len(r) > 2:
                d.text((30 + pad + 84, y + 34), r[2], font=font(F_BODY_MED, 32), fill=(150, 150, 168, 255), anchor="lm")
            y += 116
        return canvas
    raise ValueError(typ)


def banner_image(text, color_name, size=96, width=980):
    color = PALETTE.get(color_name, PALETTE["pink"])
    f = font(F_DISPLAY, size)
    dd = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    lines = wrap(text, f, width - 110, dd)
    lh = int(size * 1.18)
    h = len(lines) * lh + 70
    canvas = Image.new("RGBA", (width + 60, h + 60), (0, 0, 0, 0))
    canvas.alpha_composite(rounded_shadow(canvas.size, (30, 30, 30 + width, 30 + h), 30, blur=24, alpha=100))
    d = ImageDraw.Draw(canvas, "RGBA")
    d.rounded_rectangle([30, 30, 30 + width, 30 + h], radius=30, fill=color + (255,))
    for i, ln in enumerate(lines):
        d.text((30 + width / 2, 30 + 36 + i * lh + lh / 2), ln, font=f, fill=(255, 255, 255, 255), anchor="mm")
    return canvas


def sticker_image(top, word):
    w, h = 820, 420
    canvas = Image.new("RGBA", (w + 60, h + 60), (0, 0, 0, 0))
    canvas.alpha_composite(rounded_shadow(canvas.size, (30, 30, 30 + w, 30 + h - 50), 38, blur=28, alpha=110))
    d = ImageDraw.Draw(canvas, "RGBA")
    d.rounded_rectangle([30, 30, 30 + w, 30 + h - 50], radius=38, fill=(255, 255, 255, 255))
    cx = 30 + w / 2
    d.polygon([(cx - 36, 30 + h - 52), (cx + 36, 30 + h - 52), (cx, 30 + h + 2)], fill=(255, 255, 255, 255))
    d.text((cx, 30 + 92), top, font=font(F_BODY, 46), fill=(130, 130, 150, 255), anchor="mm")
    d.text((cx, 30 + 196), word, font=font(F_DISPLAY, 112), fill=PALETTE["indigo"] + (255,), anchor="mm")
    return canvas


# ------------------------------------------------------------------ render
class Painter:
    def __init__(self, script, tl, total):
        self.s, self.tl, self.total = script, tl, total
        kinds = []
        for b in script["beats"]:
            if b.get("scene") and b["scene"] not in kinds:
                kinds.append(b["scene"])
        self.default_scene = script.get("scene", "study")
        self.bgs = {k: flat_bg(k, script.get("seed", 1)) for k in set(kinds + [self.default_scene])}
        self.scenes = [(0.0, self.default_scene)]
        for seg in tl:
            if seg["beat"].get("scene") and seg["beat"]["scene"] != self.scenes[-1][1]:
                self.scenes.append((seg["start"] - 0.1, seg["beat"]["scene"]))
        self.cast = script["cast"]
        self.pos = {c: x for c, x in zip(self.cast, ({2: [290, 790], 3: [200, 540, 880]}[len(self.cast)]))}
        self.size = 460 if len(self.cast) == 2 else 390

    def bg_frame(self, t):
        def crop(kind, t0):
            z = 1.0 + 0.1 * (1 - (1 - clamp((t - t0) / 9.0)) ** 2)
            cw, ch = BW / z, BH / z
            ox = (BW - cw) / 2 + 16 * math.sin(t * 0.25)
            oy = (BH - ch) * 0.55
            return self.bgs[kind].crop((int(ox), int(oy), int(ox + cw), int(oy + ch))).resize((W, H), Image.BILINEAR)
        cur = 0
        for i, (st, k) in enumerate(self.scenes):
            if t >= st:
                cur = i
        st, k = self.scenes[cur]
        img = crop(k, st)
        if cur > 0 and t - st < 0.4:
            pst, pk = self.scenes[cur - 1]
            img = Image.blend(crop(pk, pst), img, (t - st) / 0.4)
        return img

    def seg_at(self, t):
        idx = 0
        for i, seg in enumerate(self.tl):
            if seg["start"] - GAP <= t <= seg["end"] + GAP:
                idx = i
        return idx

    def frame(self, t):
        img = self.bg_frame(t).convert("RGBA")
        end_card = t >= self.total - self.s.get("end_hold", 2.6)
        idx = self.seg_at(t)
        seg = self.tl[idx]
        speaker = seg["beat"]["who"] if not end_card else None
        for k, who in enumerate(self.cast):
            cx = self.pos[who]
            talking = who == speaker and seg["start"] <= t <= seg["end"]
            mouth = 0.0
            if talking:
                i = int((t - seg["start"]) * FPS)
                mouth = float(seg["env"][min(i, len(seg["env"]) - 1)]) ** 0.8
            blink = 1.0 if (t * 0.9 + k * 0.37) % 3.1 < 0.12 else 0.0
            mood = "happy"
            if not end_card:
                moods = seg["beat"].get("moods", {})
                mood = seg["beat"].get("mood", "happy") if who == speaker else moods.get(who, "happy")
            look = 0.6 if speaker and self.pos.get(speaker, cx) > cx else -0.6
            if speaker == who or not speaker:
                look = 0
            bob = -abs(math.sin(t * 9)) * 12 if talking else math.sin(t * 2 + k) * 5
            if end_card:
                bob = -abs(math.sin(t * 6 + k)) * 20
            wave = 1 if (end_card or (talking and seg["beat"].get("wave"))) else 0
            spr = C.render(who, {"t": t + k, "mouth": round(mouth * 6) / 6, "blink": blink, "mood": mood, "look": look, "bob": bob, "flip": cx > W / 2, "wave": wave}, size=self.size)
            if talking:
                spr = spr.resize((int(spr.width * 1.05), int(spr.height * 1.05)), Image.BILINEAR)
            img.alpha_composite(spr, (int(cx - spr.width / 2), int(1600 - spr.height)))
        # panel
        panel_beat = None
        for i, sg in enumerate(self.tl):
            if sg["beat"].get("panel") and sg["start"] - 0.05 <= t:
                panel_beat = i
        if panel_beat is not None and not end_card:
            sg = self.tl[panel_beat]
            prog = clamp((t - sg["start"]) / max(sg["end"] - sg["start"], 0.01))
            appear = t - sg["start"] + 0.05
            accent = sg["beat"].get("accent", "indigo")
            pi = panel_image(sg["beat"]["panel"], prog, accent)
            sc = back_out(appear / 0.35) if appear < 0.35 else 1.0
            pw, ph = int(pi.width * sc * 0.9), int(pi.height * sc * 0.9)
            if pw > 10 and ph > 10:
                p2 = pi.resize((pw, ph), Image.BILINEAR)
                img.alpha_composite(p2, (int(W / 2 - p2.width / 2), int(560 - p2.height / 2)))
        d = ImageDraw.Draw(img, "RGBA")
        hk = self.s.get("hook")
        hd = self.s.get("hook_dur", 3.4)
        if hk and t < hd + 0.5:
            q = ease_out(t / 0.3) if t < hd else 1 - ease_out((t - hd) / 0.4)
            bi = banner_image(hk, self.s.get("hook_color", "pink"))
            img.alpha_composite(bi, (int(W / 2 - bi.width / 2), int(150 - (1 - q) * 300)))
        elif self.s.get("tag"):
            tg = banner_image(self.s["tag"], self.s.get("hook_color", "pink"), size=56, width=880)
            img.alpha_composite(tg, (int(W / 2 - tg.width / 2), 70))
        for sg in self.tl:
            rb = sg["beat"].get("rehook")
            if rb and sg["start"] - 0.3 <= t <= sg["start"] + 1.9:
                u = t - (sg["start"] - 0.3)
                q = back_out(u / 0.3) if u < 0.3 else 1.0
                if u > 1.9:
                    q = 1 - (u - 1.9) / 0.3
                bi = banner_image(rb, "red", size=116, width=1000)
                s2 = max(0.05, q)
                bi = bi.resize((int(bi.width * s2), int(bi.height * s2)), Image.BILINEAR)
                img.alpha_composite(bi, (int(W / 2 - bi.width / 2), int(760 - bi.height / 2)))
        for sg in self.tl:
            st_ = sg["beat"].get("sticker")
            if st_ and t >= sg["start"] - 0.2:
                u = t - (sg["start"] - 0.2)
                q = back_out(u / 0.4) if u < 0.4 else 1.0
                si = sticker_image(st_["top"], st_["word"])
                s2 = max(0.05, q) * 0.95
                si = si.resize((int(si.width * s2), int(si.height * s2)), Image.BILINEAR)
                img.alpha_composite(si, (int(W / 2 - si.width / 2), int(660 - si.height / 2)))
        if not end_card and seg["start"] <= t <= seg["end"] + 0.15:
            words = seg["beat"]["say"].split()
            dur = seg["end"] - seg["start"]
            wts = np.array([len(w_) + 2 for w_ in words], float)
            cum = np.cumsum(wts) / wts.sum() * dur
            kk = min(int(np.searchsorted(cum, t - seg["start"])), len(words) - 1)
            g0 = (kk // 4) * 4
            chunk = words[g0:g0 + 4]
            size = 78
            while True:
                f = font(F_DISPLAY, size)
                ws = [d.textlength(w_, font=f) + 20 for w_ in chunk]
                if sum(ws) <= W - 90 or size <= 48:
                    break
                size -= 4
            tw = sum(ws)
            bx0 = (W - tw) / 2 - 30
            d.rounded_rectangle([bx0, 1740, bx0 + tw + 60, 1820], radius=28, fill=(22, 18, 40, 200))
            x = (W - tw) / 2
            for i, w_ in enumerate(chunk):
                cur = g0 + i == kk
                col = (255, 214, 90, 255) if cur else (255, 255, 255, 255)
                d.text((x, 1780), w_, font=f, fill=col, anchor="lm")
                x += ws[i]
        if end_card:
            u = t - (self.total - self.s.get("end_hold", 2.6))
            q = ease_out(u / 0.4)
            handle = self.s.get("end_text", "Follow @shipwithpip")
            f = font(F_DISPLAY, 80)
            tw = d.textlength(handle, font=f)
            d.rounded_rectangle([W / 2 - tw / 2 - 36, 1740, W / 2 + tw / 2 + 36, 1830], radius=30, fill=(22, 18, 40, int(220 * q)))
            d.text((W / 2, 1785), handle, font=f, fill=(255, 255, 255, int(255 * q)), anchor="mm")
        ImageDraw.Draw(img, "RGBA").rectangle([0, 0, W * clamp(t / self.total), 10], fill=PALETTE["indigo"] + (255,))
        return img.convert("RGB")


def write_post(s, path, total):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{s['title']}\nLength: {total:.1f}s\n\nCAPTION\n{s['caption']}\n\n")
        fh.write(" ".join("#" + h.lstrip("#") for h in s["hashtags"]) + "\n\n")
        if s.get("pinned_comment"):
            fh.write(f"PINNED COMMENT\n{s['pinned_comment']}\n\n")
        if s.get("sources"):
            fh.write("SOURCES\n" + "\n".join(s["sources"]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    s = json.load(open(a.script, encoding="utf-8"))
    tl, audio, total = build_audio(s, a.model)
    print("total", round(total, 1), "beat ends", [round(x["end"], 1) for x in tl])
    P = Painter(s, tl, total)
    if a.stills:
        ts = s.get("still_times") or [round(total * i / 9, 1) for i in range(1, 9)]
        fr = [P.frame(t).resize((300, 533)) for t in ts]
        sheet = Image.new("RGB", (300 * len(fr), 533))
        for i, f_ in enumerate(fr):
            sheet.paste(f_, (i * 300, 0))
        sheet.save(a.stills)
        return
    wav = os.path.splitext(a.out)[0] + "_a.wav"
    sf.write(wav, audio, SR)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-15:TP=-1.2:LRA=11",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for i in range(int(total * FPS)):
        ff.stdin.write(P.frame(i / FPS).tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    write_post(s, os.path.splitext(a.out)[0] + "_post.txt", total)
    print(a.out, round(total, 1))


if __name__ == "__main__":
    main()
