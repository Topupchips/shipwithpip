"""Visual-math "scale climb" reels: something doubles (or multiplies) every step and a log-scale camera
zooms out past real landmarks. Dark style, word-by-word captions, same voice and music as minimal.py.

usage: python scale.py script.json out.mp4 --model <kokoro dir> [--stills]

script.json:
{
  "id": "folds", "start_m": 0.0001, "factor": 2, "unit_word": "FOLD",
  "landmarks": [{"name": "MOON", "m": 384400000, "value": "384,400 km", "icon": "moon"}, ...],
  "beats": [{"say": "...", "key": ["word"], "to": 42, "stamp": "optional red stamp"}, ...],
  "title", "caption", "hashtags", "sources", "facts"
}
Each beat animates the step counter from the previous beat's "to" to its own "to".
Every displayed thickness is start_m * factor ** step at an integer step: nothing is invented.
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter

import minimal as M
from minimal import W, H, FPS, SR, font, al, tc, clamp, ease, ease_io, F_BOLD, F_SEMI, F_MED, F_MONO

TH = M.THEMES["dark"]
PAPER = (244, 238, 222)
PAPER_D = (196, 188, 168)
LIME = (163, 230, 53)
YEL = (250, 204, 21)
Y_TOP = 600            # screen y of the stack top once the camera follows it
Y_BOT = 1290           # bottom of the visual zone
PXD = 150              # pixels per decade
CAM_MIN = -1.2         # camera never sits lower than this (log10 m), so the first folds grow upward
COL_X0, COL_X1 = 250, 410


def fmt_len(m):
    if m < 0.01: return f"{m * 1000:.1f} mm"
    if m < 1: return f"{m * 1000:.1f} mm" if m < 0.1 else f"{m * 100:.1f} cm"
    if m < 1000: return f"{m:,.1f} m"
    if m < 1e13: return f"{m / 1000:,.0f} km"
    ly = m / 9.46e15
    if ly < 1e6: return f"{ly:,.1f} light-years" if ly < 100 else f"{ly:,.0f} light-years"
    if ly < 1e9: return f"{ly / 1e6:,.1f} million light-years"
    return f"{ly / 1e9:,.0f} billion light-years"


SUP = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")


def tick_label(e):
    named = {-4: "0.1 mm", -3: "1 mm", -2: "1 cm", -1: "10 cm", 0: "1 m", 1: "10 m", 2: "100 m", 3: "1 km"}
    return named.get(e, "10" + str(e).translate(SUP) + " m")


# ------------------------------------------------------------------ icons
def icon(d, kind, x, y, a):
    if kind == "mountain":
        d.polygon([(x - 46, y + 26), (x, y - 34), (x + 46, y + 26)], fill=al((148, 163, 184), a))
        d.polygon([(x - 15, y - 14), (x, y - 34), (x + 15, y - 14), (x + 4, y - 8), (x - 5, y - 12)], fill=al((255, 255, 255), a))
    elif kind == "station":
        d.rectangle([x - 12, y - 8, x + 12, y + 8], fill=al((226, 232, 240), a))
        for s in (-1, 1):
            d.rectangle([x + s * 18 - 14, y - 22, x + s * 18 + 14, y + 22], fill=al((59, 130, 246), a), outline=al((191, 219, 254), a), width=2)
        d.line([(x - 46, y), (x + 46, y)], fill=al((226, 232, 240), a), width=3)
    elif kind == "moon":
        d.ellipse([x - 30, y - 30, x + 30, y + 30], fill=al((203, 213, 225), a))
        for cx, cy, r in ((-10, -8, 7), (9, 6, 9), (-4, 14, 4), (12, -14, 4)):
            d.ellipse([x + cx - r, y + cy - r, x + cx + r, y + cy + r], fill=al((148, 163, 184), a))
    elif kind == "sun":
        for r, op in ((60, 0.10), (46, 0.2)):
            d.ellipse([x - r, y - r, x + r, y + r], fill=al((251, 191, 36), op * a))
        d.ellipse([x - 32, y - 32, x + 32, y + 32], fill=al((253, 224, 71), a))
    elif kind == "star":
        for r, op in ((34, 0.12), (22, 0.3)):
            d.ellipse([x - r, y - r, x + r, y + r], fill=al((248, 113, 113), op * a))
        d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=al((254, 202, 202), a))
    elif kind in ("galaxy", "andromeda"):
        col = (196, 181, 253) if kind == "galaxy" else (147, 197, 253)
        sx, sy = (52, 20) if kind == "galaxy" else (50, 26)
        for k, op in ((1.0, 0.15), (0.7, 0.3), (0.4, 0.6)):
            d.ellipse([x - sx * k, y - sy * k, x + sx * k, y + sy * k], fill=al(col, op * a))
        d.ellipse([x - 8, y - 6, x + 8, y + 6], fill=al((255, 255, 255), a))
    elif kind == "universe":
        rng = np.random.default_rng(7)
        for _ in range(46):
            ang, rr = rng.uniform(0, 6.283), 40 * math.sqrt(rng.uniform())
            c = [(244, 114, 182), (96, 165, 250), (250, 204, 21), (167, 139, 250)][int(rng.integers(4))]
            px, py = x + rr * math.cos(ang), y + rr * math.sin(ang)
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=al(c, a))
        d.ellipse([x - 44, y - 44, x + 44, y + 44], outline=al((167, 139, 250), 0.7 * a), width=3)


# ------------------------------------------------------------------ timeline
def step_at(t, tl):
    """Continuous step value (for the camera) and integer step (for numbers) at time t."""
    prev = tl[0]["beat"].get("from", tl[0]["beat"]["to"])
    for i, s in enumerate(tl):
        b = s["beat"]; to = b.get("to", prev)
        nxt = tl[i + 1]["start"] if i + 1 < len(tl) else s["end"] + 3
        if t < nxt or i == len(tl) - 1:
            a0 = s["start"] + 0.25; a1 = max(a0 + 0.6, s["start"] + (s["end"] - s["start"]) * 0.85)
            f = ease_io(clamp((t - a0) / (a1 - a0)))
            v = prev + (to - prev) * f
            return v, (int(math.floor(v + 1e-6)) if to >= prev else int(math.ceil(v - 1e-6))), s
        prev = to
    return prev, int(prev), tl[-1]


def render_frame(t, S, tl, bg, total):
    v, n, seg = step_at(t, tl)
    b = seg["beat"]
    lf = math.log10(S["factor"])
    top = math.log10(S["start_m"]) + v * lf
    cam = max(top, CAM_MIN)
    Y = lambda e: Y_TOP - (e - cam) * PXD
    img = bg.copy(); d = ImageDraw.Draw(img, "RGBA")

    # ruler
    lo, hi = math.floor(cam - (Y_BOT - Y_TOP) / PXD) - 1, math.ceil(cam + Y_TOP / PXD) + 1
    for e in range(int(lo), int(hi) + 1):
        y = Y(e)
        if 360 < y < Y_BOT:
            d.line([(160, y), (200, y)], fill=al(TH["muted"], 0.9), width=3)
            d.line([(200, y), (1040, y)], fill=al(TH["line"], 0.35), width=1)
            tc(d, (140, y), tick_label(e), font(F_MED, 28), al(TH["muted"], 0.9), "rm")
    d.line([(200, 360), (200, Y_BOT)], fill=al(TH["muted"], 0.5), width=2)

    # stack column
    yt = max(Y(top), 360)
    pc, pd = tuple(S.get("col_rgb", PAPER)), tuple(S.get("col_rgb_d", PAPER_D))  # optional column colours
    col = Image.new("RGBA", (COL_X1 - COL_X0, int(Y_BOT - yt) + 1), pc + (255,))
    cd = ImageDraw.Draw(col)
    layers = S["factor"] ** max(n, 0)
    if layers <= 64:
        hgt = col.size[1]
        for k in range(1, int(layers)):
            yy = hgt - hgt * k / layers
            cd.line([(0, yy), (col.size[0], yy)], fill=pd + (255,), width=2)
    else:
        arr = np.asarray(col).astype(np.float32)
        g = np.linspace(0.78, 1.0, arr.shape[0])[:, None, None]
        arr[..., :3] *= g
        col = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")
    img.paste(col, (COL_X0, int(yt)), col)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([COL_X0, yt, COL_X1, Y_BOT], outline=al((255, 255, 255), 0.9), width=3)
    for r, op in ((30, 0.08), (16, 0.16)):
        d.rectangle([COL_X0 - r, yt - r * 0.6, COL_X1 + r, yt + r * 0.6], fill=al(YEL, op))
    d.line([(COL_X0 - 10, yt), (COL_X1 + 10, yt)], fill=al(YEL, 1), width=5)

    # landmarks
    for L in S["landmarks"]:
        e = math.log10(L["m"]); y = Y(e)
        if not (330 < y < Y_BOT + 40): continue
        passed = top >= e - 1e-9
        c = LIME if passed else (255, 255, 255)
        for x in range(COL_X1 + 20, 1040, 26):
            d.line([(x, y), (x + 14, y)], fill=al(c, 0.85), width=3)
        icon(d, L["icon"], 500, y - 52, 1.0)
        tc(d, (1040, y - 70), L["name"], font(F_BOLD, 40), al(c, 1), "rm")
        tc(d, (1040, y - 28), L["value"], font(F_MED, 30), al(TH["muted"], 1), "rm")

    # fade the edges of the visual zone
    for k in range(40):
        a = (1 - k / 40) * 0.9
        d.line([(0, 360 + k), (W, 360 + k)], fill=al(TH["bg"], a))
        d.line([(0, Y_BOT - k), (W, Y_BOT - k)], fill=al(TH["bg"], a))
    d.rectangle([0, 0, W, 360], fill=al(TH["bg"], 1))
    d.rectangle([0, Y_BOT, W, H], fill=al(TH["bg"], 1))

    # header: counter + thickness
    tc(d, (540, 92), S.get("unit_word", "FOLD"), font(F_SEMI, 40), al(TH["muted"], 1))
    tc(d, (540, 192), f"{max(n, 0)}", font(F_BOLD, 150), al(YEL, 1))
    thick = S["start_m"] * S["factor"] ** max(n, 0)
    txt = fmt_len(thick) + " " + S.get("readout_word", "thick")
    f = font(F_SEMI, 46); w = d.textlength(txt, font=f)
    d.rounded_rectangle([540 - w / 2 - 34, 278, 540 + w / 2 + 34, 344], radius=33, fill=al(TH["card"], 1), outline=al(TH["line"], 1), width=2)
    tc(d, (540, 311), txt, f, al(TH["ink"], 1))

    # stamp
    if b.get("stamp"):
        g = ease(clamp((t - seg["start"] - (seg["end"] - seg["start"]) * 0.55) / 0.25))
        if g > 0:
            fs = font(F_BOLD, 64); tw = d.textlength(b["stamp"], font=fs)
            ov = Image.new("RGBA", (int(tw + 90), 130), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
            od.rounded_rectangle([6, 6, ov.size[0] - 6, 124], radius=20, outline=M.RED + (int(255 * g),), width=8, fill=(40, 10, 12, int(220 * g)))
            od.text((ov.size[0] / 2, 65), b["stamp"], font=fs, fill=M.RED + (int(255 * g),), anchor="mm")
            ov = ov.rotate(-6, expand=True, resample=Image.BICUBIC)
            sc = 1.6 - 0.6 * g; ov = ov.resize((int(ov.size[0] * sc), int(ov.size[1] * sc)))
            img.paste(ov, (int(560 - ov.size[0] / 2), int(860 - ov.size[1] / 2)), ov)
            d = ImageDraw.Draw(img, "RGBA")

    # hook overlay on the first beat
    if b.get("hook"):
        g = clamp((seg["end"] - t) / 0.3)
        lines = b["hook"]
        for i, ln in enumerate(lines):
            tc(d, (720, 700 + i * 110), ln, font(F_BOLD, 92), al(YEL if i else TH["ink"], g))

    if not b.get("nocap"):
        M.caption(d, seg, min(t, seg["end"]), TH)
    tc(d, (540, 1840), "@shipwithpip", font(F_MED, 34), al(TH["muted"], 0.8))
    return img


LEX_ADD = {  # words the Kokoro US lexicon is missing (it silently skips them otherwise)
    "kilometers": "k ə l ˈ ɑ m ə ɾ ə ɹ z", "millimeters": "m ˈ ɪ l ə m ˌ i ɾ ə ɹ z", "that's": "ð ˈ æ t s",
    "britney": "b ɹ ˈ ɪ t n i", "gallivan": "ɡ ˈ æ l ɪ v ə n", "centimeters": "s ˈ ɛ n t ə m ˌ i ɾ ə ɹ z",
    "doublings": "d ˈ ʌ b l ɪ ŋ z",
}


def patch_lexicon(model_dir):
    p = os.path.join(model_dir, "lexicon-us-en.txt")
    if not os.path.exists(p): return
    have = {ln.split(" ", 1)[0].lower() for ln in open(p, encoding="utf-8")}
    add = [f"{w} {ph}\n" for w, ph in LEX_ADD.items() if w not in have]
    if add:
        with open(p, "a", encoding="utf-8") as fh: fh.writelines(add)


def render(S, out, model_dir, only=None):
    patch_lexicon(model_dir)
    for b in S["beats"]:  # optional "speak": what the voice says, when it must differ from the caption text
        b["_cap"] = b["say"]
        if b.get("speak"): b["say"] = b["speak"]
    tl, vox, total = M.build_audio(S, model_dir)
    for b in S["beats"]: b["say"] = b.pop("_cap")
    for s in tl: s["beat"].setdefault("cap_y", 1440)
    bg = M.make_bg(TH)
    if only is not None:
        return [render_frame(t, S, tl, bg, total) for t in only], tl, total
    audio = M.mix(vox, total, tl, [s["start"] for s in tl], sum(map(ord, S["id"])))
    wav = out[:-4] + ".wav"; sf.write(wav, audio, SR)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-af", "loudnorm=I=-14:TP=-1.2:LRA=11", "-c:a", "aac", "-b:a", "192k", "-shortest",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for fr in range(int(total * FPS)):
        ff.stdin.write(render_frame(fr / FPS, S, tl, bg, total).convert("RGB").tobytes())
    ff.stdin.close(); ff.wait(); os.remove(wav)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("script"); ap.add_argument("out"); ap.add_argument("--model", required=True)
    ap.add_argument("--stills", action="store_true"); a = ap.parse_args()
    S = json.load(open(a.script, encoding="utf-8"))
    if a.stills:
        _, tl, total = render(S, a.out, a.model, only=[0.0])
        ts = [0.5] + [x["start"] + (x["end"] - x["start"]) * 0.9 for x in tl]
        frames, _, _ = render(S, a.out, a.model, only=ts)
        sh = Image.new("RGB", (270 * 6, 480 * math.ceil(len(frames) / 6)))
        for i, fimg in enumerate(frames):
            sh.paste(fimg.convert("RGB").resize((270, 480)), ((i % 6) * 270, (i // 6) * 480))
        sh.save(a.out); print(a.out, f"{total:.1f}s")
    else:
        total = render(S, a.out, a.model); print(f"{a.out}: {total:.1f}s")
