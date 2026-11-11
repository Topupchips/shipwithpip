"""Turn a reel script (JSON) into a finished 1080x1920 MP4 with voices, lip-sync, captions and music.

usage: python make_reel.py script.json out.mp4 --model <kokoro-en-v0_19 dir>
"""
import argparse
import json
import math
import os
import subprocess
import textwrap

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import characters as C
import voice

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
W, H, FPS, SR = 1080, 1920, 30, 24000
MAX_LEN = 30.0
GAP = 0.14
END_CARD = 2.4

F_BOLD = os.path.join(A, "Poppins-Bold.ttf")
F_MED = os.path.join(A, "Poppins-Medium.ttf")
F_MONO = os.path.join(A, "DejaVuSansMono.ttf")
INK = (24, 22, 38)

PALETTES = {
    "peach": {"bg1": (255, 236, 222), "bg2": (255, 214, 196), "dot": (255, 255, 255), "accent": (255, 92, 60), "hl": (255, 214, 102)},
    "mint": {"bg1": (226, 248, 240), "bg2": (196, 236, 222), "dot": (255, 255, 255), "accent": (20, 150, 130), "hl": (255, 214, 102)},
    "lilac": {"bg1": (236, 234, 255), "bg2": (214, 210, 250), "dot": (255, 255, 255), "accent": (92, 84, 214), "hl": (255, 214, 102)},
    "money": {"bg1": (14, 58, 44), "bg2": (6, 24, 22), "dot": (255, 214, 102), "accent": (16, 150, 96), "hl": (255, 214, 102), "coins": True},
    "neon": {"bg1": (44, 18, 96), "bg2": (16, 8, 44), "dot": (255, 120, 200), "accent": (232, 54, 140), "hl": (255, 232, 102), "coins": True},
    "sky": {"bg1": (226, 240, 255), "bg2": (200, 224, 252), "dot": (255, 255, 255), "accent": (36, 110, 220), "hl": (255, 214, 102)},
}

_fonts = {}


def font(path, size):
    k = (path, size)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(path, size)
    return _fonts[k]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


# ------------------------------------------------------------------ audio
def build_audio(script, model_dir):
    """Voice every beat, fit the reel under MAX_LEN, return timeline + mixed audio."""
    MAX_LEN = script.get("max_len", 30.0)
    scale = 1.0
    for _ in range(4):
        clips = []
        for b in script["beats"]:
            c = C.CAST[b["who"]]
            x, sr = voice.say(model_dir, b["say"], c["voice"], min(c["speed"] * scale, 1.45))
            assert sr == SR, sr
            clips.append(x)
        speech = 0.15 + sum(len(x) / SR + GAP for x in clips)
        if speech + END_CARD <= MAX_LEN or scale >= 1.3:
            break
        scale *= (speech + END_CARD) / (MAX_LEN - 0.3)
    total = speech + END_CARD
    n = int(total * SR) + SR // 10
    vox = np.zeros(n, np.float32)
    t = 0.15
    timeline = []
    for b, x in zip(script["beats"], clips):
        i = int(t * SR)
        vox[i:i + len(x)] += x
        timeline.append({"start": t, "end": t + len(x) / SR, "beat": b, "env": _envelope(x)})
        t += len(x) / SR + GAP
    peak = np.max(np.abs(vox)) or 1
    vox = vox / peak * 0.89
    return timeline, vox, total


def _envelope(x):
    hop = SR // FPS
    frames = len(x) // hop + 1
    env = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-9) for i in range(frames)])
    env = env / (np.percentile(env, 95) + 1e-6)
    return np.clip(env, 0, 1)


def _tone(f, dur, kind="sine"):
    t = np.arange(int(dur * SR)) / SR
    if kind == "tri":
        return (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f * t))
    return np.sin(2 * np.pi * f * t)


def music_bed(total, seed):
    """Original lo-fi loop: soft chords, kick, hats. Generated, so no copyright."""
    rng = np.random.default_rng(seed)
    bpm = 92
    beat = 60 / bpm
    n = int(total * SR) + SR
    out = np.zeros(n, np.float32)
    progs = [[(220.0, 277.18, 329.63), (196.0, 246.94, 293.66), (174.61, 220.0, 261.63), (196.0, 246.94, 329.63)],
             [(261.63, 329.63, 392.0), (220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 246.94, 293.66)]]
    prog = progs[seed % len(progs)]
    bar = beat * 4
    k = 0
    while k * bar < total + bar:
        chord = prog[k % 4]
        seg = sum(_tone(f, bar, "tri") * 0.06 for f in chord)
        seg = seg * np.exp(-np.arange(len(seg)) / SR / 2.2)
        i = int(k * bar * SR)
        out[i:i + len(seg)] += seg[: max(0, n - i)]
        for q in range(4):
            j = int((k * bar + q * beat) * SR)
            if q % 2 == 0:  # kick
                kt = np.arange(int(0.18 * SR)) / SR
                kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 18) * 0.35
                out[j:j + len(kick)] += kick[: max(0, n - j)]
            for half in (0, 0.5):  # hats
                jj = int((k * bar + (q + half) * beat) * SR)
                ht = int(0.03 * SR)
                hat = rng.normal(0, 1, ht) * np.exp(-np.arange(ht) / SR * 120) * 0.035
                out[jj:jj + ht] += hat[: max(0, n - jj)]
        k += 1
    return out[: int(total * SR) + SR // 10]


def sfx_pop():
    t = np.arange(int(0.09 * SR)) / SR
    return (np.sin(2 * np.pi * (500 + 900 * t / 0.09) * t) * np.exp(-t * 40) * 0.35).astype(np.float32)


def sfx_click():
    t = np.arange(int(0.012 * SR)) / SR
    return (np.random.default_rng(1).normal(0, 1, len(t)) * np.exp(-t * 400) * 0.06).astype(np.float32)


def sfx_whoosh():
    d = 0.45
    t = np.arange(int(d * SR)) / SR
    noise = np.random.default_rng(2).normal(0, 1, len(t))
    env = np.sin(np.pi * t / d) ** 2
    out = np.convolve(noise, np.ones(20) / 20, mode="same") * env * 0.25
    return out.astype(np.float32)


def mix(vox, total, timeline, script):
    n = len(vox)
    bed = music_bed(total, hash(script.get("id", "x")) % 997)[:n]
    bed = np.pad(bed, (0, n - len(bed)))
    duck = np.ones(n, np.float32) * 0.55
    for seg in timeline:
        duck[int(seg["start"] * SR): int(seg["end"] * SR)] = 0.28
    duck = np.convolve(duck, np.ones(2400) / 2400, mode="same")
    out = vox + bed * duck
    pop, click, whoosh = sfx_pop(), sfx_click(), sfx_whoosh()

    def add(sig, at):
        i = int(at * SR)
        if 0 <= i < n:
            out[i:i + len(sig)] += sig[: n - i]

    for seg in timeline:
        p = seg["beat"].get("panel")
        if p:
            add(pop, seg["start"] - 0.05)
            if p["type"] == "prompt":
                dur = (seg["end"] - seg["start"]) * 0.8
                for k in range(int(dur / 0.07)):
                    add(click, seg["start"] + k * 0.07)
    add(whoosh, total - END_CARD - 0.2)
    peak = np.max(np.abs(out)) or 1
    return (out / peak * 0.95).astype(np.float32)


# ------------------------------------------------------------------ visuals
def make_background(pal):
    y = np.linspace(0, 1, H)[:, None]
    bg = (np.array(pal["bg1"]) * (1 - y[..., None]) + np.array(pal["bg2"]) * y[..., None])
    bg = np.repeat(bg, W, axis=1).astype(np.uint8)
    img = Image.fromarray(bg, "RGB")
    d = ImageDraw.Draw(img, "RGBA")
    for yy in range(0, H, 64):
        for xx in range((yy // 64) % 2 * 32, W, 64):
            d.ellipse([xx - 4, yy - 4, xx + 4, yy + 4], fill=pal["dot"] + (110,))
    # floor
    d.rounded_rectangle([-40, 1500, W + 40, H + 60], radius=80, fill=(255, 255, 255, 90))
    return img


def wrap(d, text, f, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=f) <= maxw:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_coins(img, pal, t):
    d = ImageDraw.Draw(img, "RGBA")
    for k in range(9):
        sp = 60 + (k * 37) % 50
        y = (H + 120) - ((t * sp + k * 230) % (H + 240))
        x = 90 + (k * 331) % (W - 180) + math.sin(t * 1.3 + k) * 26
        r = 34 + (k % 3) * 12
        wob = abs(math.cos(t * 2.2 + k))
        rx = max(6, int(r * (0.35 + 0.65 * wob)))
        d.ellipse([x - rx, y - r, x + rx, y + r], fill=(255, 205, 64, 150), outline=(255, 240, 160, 200), width=4)
        if rx > r * 0.6:
            f = font(F_BOLD, int(r * 1.1))
            d.text((x, y), "$", font=f, fill=(150, 98, 10, 190), anchor="mm")


def draw_badge(img, text, pal, t):
    d = ImageDraw.Draw(img, "RGBA")
    f = font(F_BOLD, 34)
    w = d.textlength(text, font=f) + 64
    pop = ease(t / 0.3)
    y = 78 + (1 - pop) * -40
    d.rounded_rectangle([(W - w) / 2, y, (W + w) / 2, y + 66], radius=33, fill=pal["hl"] + (int(255 * pop),))
    d.text((W / 2, y + 33), text, font=f, fill=INK + (int(255 * pop),), anchor="mm")


def draw_hook(img, text, pal, t):
    d = ImageDraw.Draw(img, "RGBA")
    f = font(F_BOLD, 66)
    lines = wrap(d, text, f, 900)
    pop = ease(t / 0.35)
    y = 170 + (50 if False else 0) - (1 - pop) * 60
    for ln in lines:
        w = d.textlength(ln, font=f)
        x = (W - w) / 2
        d.rounded_rectangle([x - 22, y - 4, x + w + 22, y + 84], radius=18, fill=INK + (int(235 * pop),))
        d.text((x, y), ln, font=f, fill=(255, 255, 255, int(255 * pop)))
        y += 96


def draw_card(img, box, alpha=1.0):
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0, x1, y1 = box
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([x0 + 8, y0 + 14, x1 + 8, y1 + 14], radius=36, fill=(0, 0, 0, int(60 * alpha)))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(12)))
    d.rounded_rectangle(box, radius=36, fill=(255, 255, 255, int(250 * alpha)), outline=INK + (int(255 * alpha),), width=5)


def draw_panel(img, panel, prog, pal, appear):
    """prog: 0-1 progress through the beat that introduced the panel (1 once it's past)."""
    a = ease(appear / 0.3)
    dy = (1 - a) * 40
    box = [70, 440 + dy, W - 70, 1000 + dy]
    draw_card(img, box, a)
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0 = box[0] + 44, box[1] + 36
    typ = panel["type"]
    if typ == "big":
        f = font(F_BOLD, 74)
        lines = wrap(d, panel["text"], f, W - 240)
        yy = (box[1] + box[3]) / 2 - len(lines) * 46 - (40 if panel.get("source") else 0)
        for i, ln in enumerate(lines):
            col = pal["accent"] if i == len(lines) - 1 else INK
            d.text((W / 2, yy + i * 92), ln, font=f, fill=col, anchor="mt")
        if panel.get("source"):
            sf_ = font(F_MED, 30)
            for j, ln in enumerate(wrap(d, panel["source"], sf_, W - 260)):
                d.text((W / 2, box[3] - 110 + j * 40), ln, font=sf_, fill=(110, 104, 130), anchor="mt")
    elif typ == "number":
        lab = panel.get("label", "")
        d.text((W / 2, box[1] + 70), lab, font=font(F_BOLD, 42), fill=(110, 104, 130), anchor="mt")
        val = panel["value"]
        g = ease(min(1.0, prog / 0.7)) if panel.get("count", True) else 1.0
        if isinstance(val, (int, float)):
            txt = panel.get("prefix", "") + f"{int(val * g):,}" + panel.get("suffix", "")
        else:
            txt = val
        size = 190
        f = font(F_BOLD, size)
        while d.textlength(txt, font=f) > W - 240 and size > 80:
            size -= 8
            f = font(F_BOLD, size)
        d.text((W / 2, (box[1] + box[3]) / 2 - 10), txt, font=f, fill=pal["accent"], anchor="mm")
        if panel.get("sub"):
            sf_ = font(F_MED, 36)
            for j, ln in enumerate(wrap(d, panel["sub"], sf_, W - 260)):
                d.text((W / 2, box[3] - 130 + j * 46), ln, font=sf_, fill=INK, anchor="mt")
    elif typ == "steps":
        title = panel.get("title")
        yy = y0
        if title:
            d.text((x0, yy), title, font=font(F_BOLD, 40), fill=INK)
            yy += 80
        rows = panel["rows"]
        nshow = len(rows) if prog >= 1 else int(len(rows) * ease(prog / 0.75)) + 1
        for i, r in enumerate(rows[:nshow]):
            d.ellipse([x0, yy, x0 + 64, yy + 64], fill=pal["accent"])
            d.text((x0 + 32, yy + 32), str(i + 1), font=font(F_BOLD, 38), fill=(255, 255, 255), anchor="mm")
            d.text((x0 + 90, yy - 2), r[0], font=font(F_BOLD, 36), fill=INK)
            if len(r) > 1:
                d.text((x0 + 90, yy + 40), r[1], font=font(F_MED, 28), fill=(110, 104, 130))
            yy += 118
    elif typ == "prompt":
        d.ellipse([x0, y0 + 4, x0 + 30, y0 + 34], fill=pal["accent"])
        d.text((x0 + 44, y0), panel.get("label", "Your prompt to Claude"), font=font(F_BOLD, 34), fill=INK)
        yy = y0 + 64
        for fl in panel.get("files", []):
            fw = d.textlength(fl, font=font(F_MED, 26)) + 70
            d.rounded_rectangle([x0, yy, x0 + fw, yy + 48], radius=12, fill=pal["bg1"], outline=INK, width=3)
            d.rectangle([x0 + 14, yy + 10, x0 + 36, yy + 38], fill=pal["accent"])
            d.text((x0 + 48, yy + 8), fl, font=font(F_MED, 26), fill=INK)
            yy += 60
        text = panel["text"]
        shown = text[: int(len(text) * ease(prog / 0.8) if prog < 1 else len(text))]
        fm = font(F_MONO, 33)
        lines = []
        for para in shown.split("\n"):
            lines += textwrap.wrap(para, 44) or [""]
        for i, ln in enumerate(lines[:9]):
            d.text((x0, yy + 14 + i * 46), ln, font=fm, fill=INK)
        if prog < 1 and int(appear * 3) % 2 == 0 and lines:
            cx = x0 + d.textlength(lines[min(len(lines), 9) - 1], font=fm) + 4
            cy = yy + 14 + (min(len(lines), 9) - 1) * 46
            d.rectangle([cx, cy + 4, cx + 16, cy + 40], fill=pal["accent"])
    elif typ in ("list", "chat"):
        title = panel.get("title")
        yy = y0
        if title:
            d.text((x0, yy), title, font=font(F_BOLD, 40), fill=INK)
            yy += 74
        rows = panel.get("rows") if typ == "list" else panel.get("bubbles")
        nshow = len(rows) if prog >= 1 else int(len(rows) * ease(prog / 0.75)) + 1
        for i, r in enumerate(rows[:nshow]):
            if typ == "list":
                tag, txt = r[0], r[1]
                extra = r[2] if len(r) > 2 else ""
                tf = font(F_BOLD, 28)
                tw = d.textlength(tag, font=tf) + 32
                d.rounded_rectangle([x0, yy, x0 + tw, yy + 50], radius=25, fill=pal["accent"])
                d.text((x0 + 16, yy + 8), tag, font=tf, fill=(255, 255, 255))
                d.text((x0 + tw + 20, yy + 4), txt, font=font(F_MED, 34), fill=INK)
                if extra:
                    d.text((box[2] - 44, yy + 4), extra, font=font(F_BOLD, 34), fill=pal["accent"], anchor="ra")
                yy += 74
            else:
                who, txt = r
                f = font(F_MED, 32)
                lines = wrap(d, txt, f, 700)
                bh = len(lines) * 44 + 30
                if who == "you":
                    bx0, bx1, fill, col = box[2] - 44 - 760, box[2] - 44, pal["accent"], (255, 255, 255)
                else:
                    bx0, bx1, fill, col = x0, x0 + 760, (236, 234, 246), INK
                d.rounded_rectangle([bx0, yy, bx1, yy + bh], radius=24, fill=fill)
                for j, ln in enumerate(lines):
                    d.text((bx0 + 26, yy + 14 + j * 44), ln, font=f, fill=col)
                yy += bh + 20


def draw_caption(img, seg, t, pal):
    words = seg["beat"]["say"].split()
    dur = seg["end"] - seg["start"]
    weights = np.array([len(w) + 2 for w in words], float)
    cum = np.cumsum(weights) / weights.sum() * dur
    k = int(np.searchsorted(cum, t - seg["start"]))
    k = min(k, len(words) - 1)
    g0 = (k // 4) * 4
    chunk = words[g0:g0 + 4]
    d = ImageDraw.Draw(img, "RGBA")
    size = 64
    while True:
        f = font(F_BOLD, size)
        widths = [d.textlength(w, font=f) + 30 for w in chunk]
        if sum(widths) <= W - 100 or size <= 40:
            break
        size -= 4
    x = (W - sum(widths)) / 2
    y = 1052
    for i, w in enumerate(chunk):
        col = pal["hl"] if g0 + i == k else (255, 255, 255)
        d.text((x, y), w, font=f, fill=col, stroke_width=9, stroke_fill=INK)
        x += widths[i]


def stage_positions(cast):
    if len(cast) == 2:
        return {cast[0]: 290, cast[1]: 790}
    return {cast[0]: 200, cast[1]: 540, cast[2]: 880}


def draw_cast(img, cast, t, seg, idx, frame, pal, end_card):
    pos = stage_positions(cast)
    size = 440 if len(cast) == 2 else 380
    speaker = seg["beat"]["who"] if seg else None
    for k, who in enumerate(cast):
        cx = pos[who]
        talking = (who == speaker) and seg and seg["start"] <= t <= seg["end"]
        mouth = 0.0
        if talking:
            i = int((t - seg["start"]) * FPS)
            mouth = float(seg["env"][min(i, len(seg["env"]) - 1)]) ** 0.8
        blink_phase = (t * 0.9 + k * 0.37) % 3.1
        blink = 1.0 if blink_phase < 0.12 else 0.0
        mood = "happy"
        if end_card:
            mood = "happy"
        elif seg is not None:
            moods = seg["beat"].get("moods", {})
            mood = seg["beat"].get("mood", "happy") if who == speaker else moods.get(who, "happy")
        look = 0.6 if speaker and pos.get(speaker, cx) > cx else -0.6
        if speaker == who or not speaker:
            look = 0
        bob = -abs(math.sin(t * 9)) * 14 if talking else math.sin(t * 2 + k) * 4
        if end_card:
            bob = -abs(math.sin(t * 6 + k)) * 26
        flip = cx > W / 2
        spr = C.render(who, {"t": t + k, "mouth": mouth, "blink": blink, "mood": mood, "look": look,
                             "bob": bob, "flip": flip, "wave": 1 if (end_card or (talking and seg["beat"].get("wave"))) else 0}, size=size)
        scale = 1.07 if talking else 1.0
        if scale != 1.0:
            spr = spr.resize((int(spr.width * scale), int(spr.height * scale)), Image.BILINEAR)
        img.alpha_composite(spr, (int(cx - spr.width / 2), int(1600 - spr.height)))


def draw_end_card(img, script, pal, t):
    a = ease(t / 0.35)
    d = ImageDraw.Draw(img, "RGBA")
    box = [70, 470 - (1 - a) * 50, W - 70, 1000 - (1 - a) * 50]
    draw_card(img, box, a)
    f = font(F_BOLD, 70)
    lines = wrap(d, script["cta"], f, W - 260)
    yy = (box[1] + box[3]) / 2 - len(lines) * 44 - 40
    for i, ln in enumerate(lines):
        d.text((W / 2, yy + i * 90), ln, font=f, fill=INK + (int(255 * a),), anchor="mt")
    sub = script.get("cta_sub")
    if sub:
        sf_ = font(F_BOLD, 40)
        sw = d.textlength(sub, font=sf_) + 60
        y2 = yy + len(lines) * 90 + 30
        d.rounded_rectangle([(W - sw) / 2, y2, (W + sw) / 2, y2 + 72], radius=36, fill=pal["accent"] + (int(255 * a),))
        d.text((W / 2, y2 + 14), sub, font=sf_, fill=(255, 255, 255, int(255 * a)), anchor="mt")


def render(script, out, model_dir, tmp):
    pal = PALETTES[script.get("palette", "peach")]
    cast = script["cast"]
    timeline, vox, total = build_audio(script, model_dir)
    audio = mix(vox, total, timeline, script)
    wav = os.path.join(tmp, script["id"] + ".wav")
    sf.write(wav, audio, SR)
    bg = make_background(pal).convert("RGBA")
    # panel state per beat: the latest panel introduced, and when
    panel_at = []
    cur = None
    for i, seg in enumerate(timeline):
        if seg["beat"].get("panel"):
            cur = (seg["beat"]["panel"], seg["start"], seg["end"])
        panel_at.append(cur)
    nframes = int(total * FPS)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                           "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-14:TP=-1.2:LRA=11", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    speech_end = total - END_CARD
    for fr in range(nframes):
        t = fr / FPS
        img = bg.copy()
        end_card = t >= speech_end
        idx = None
        for i, seg in enumerate(timeline):
            if seg["start"] - GAP <= t <= seg["end"] + GAP:
                idx = i
        if idx is None:  # before first line
            idx = 0
        seg = timeline[idx]
        if pal.get("coins"):
            draw_coins(img, pal, t)
        if script.get("badge"):
            draw_badge(img, script["badge"], pal, t)
        draw_hook(img, script["hook"], pal, t)
        if end_card:
            draw_end_card(img, script, pal, t - speech_end)
        elif panel_at[idx]:
            p, ps, pe = panel_at[idx]
            prog = (t - ps) / max(pe - ps, 0.01)
            draw_panel(img, p, min(prog, 1.0) if t <= pe else 1.0, pal, t - ps + 0.05)
        draw_cast(img, cast, t, None if end_card else seg, idx, fr, pal, end_card)
        if not end_card and seg["start"] <= t <= seg["end"]:
            draw_caption(img, seg, t, pal)
        ff.stdin.write(img.convert("RGB").tobytes())
    ff.stdin.close()
    ff.wait()
    return total


def write_post(script, path, total):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{script['title']}\nLength: {total:.1f}s\n\nCAPTION\n{script['caption']}\n\n")
        fh.write(" ".join("#" + h.lstrip("#") for h in script["hashtags"]) + "\n\n")
        if script.get("pinned_comment"):
            fh.write(f"PINNED COMMENT\n{script['pinned_comment']}\n\n")
        if script.get("prompt"):
            fh.write(f"THE PROMPT (for comments/DMs)\n{script['prompt']}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    s = json.load(open(args.script, encoding="utf-8"))
    tmp = os.path.dirname(os.path.abspath(args.out))
    total = render(s, args.out, args.model, tmp)
    write_post(s, os.path.splitext(args.out)[0] + "_post.txt", total)
    os.remove(os.path.join(tmp, s["id"] + ".wav"))
    print(f"{args.out}: {total:.1f}s")
