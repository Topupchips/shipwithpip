"""Slow-motion 'bullet hits a can' retention hook. Stylised 2D, no weapon or person shown.

usage: python bulletfx.py cfg.json out.mp4 --model <kokoro dir> [--stills out.png]
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
import voice  # noqa: E402

W, H, FPS, SR = 1080, 1920, 30, 24000
A = os.path.join(HERE, "assets")
F_BOLD = os.path.join(A, "InterDisplay-Bold.otf")
F_SEMI = os.path.join(A, "InterDisplay-SemiBold.otf")
F_MONO = os.path.join(A, "JetBrainsMono-700.ttf")
_f = {}


def font(p, s):
    k = (p, int(s))
    if k not in _f:
        _f[k] = ImageFont.truetype(p, int(s))
    return _f[k]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x)
    return 3 * x * x - 2 * x ** 3


CAN_CX, CAN_W, CAN_H = 650, 190, 380
TABLE_Y = 1170
CAN_TOP = TABLE_Y - CAN_H
IMP_Y = 985
CAN_L = CAN_CX - CAN_W // 2
CAN_R = CAN_CX + CAN_W // 2


def make_bg(cfg):
    c = np.array(cfg["bg"], float)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - 640) / 720.0) ** 2 + ((yy - 980) / 900.0) ** 2)
    glow = np.clip(1 - d, 0, 1) ** 1.6
    img = c[None, None, :] * (0.55 + 0.9 * glow[..., None]) + np.array(cfg["accent"], float)[None, None, :] * 0.16 * glow[..., None]
    # table
    t = (yy >= TABLE_Y)[..., None]
    tab = img * 0.55 + 6
    img = np.where(t, tab, img)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im, "RGBA")
    d.line([(0, TABLE_Y), (W, TABLE_Y)], fill=tuple(cfg["accent"]) + (230,), width=4)
    for x in range(-200, W + 200, 120):  # table perspective lines
        d.line([(x, TABLE_Y), (540 + (x - 540) * 2.6, H)], fill=(255, 255, 255, 14), width=2)
    for y in (1230, 1330, 1480, 1700):
        d.line([(0, y), (W, y)], fill=(255, 255, 255, 12), width=2)
    return im


def make_vignette():
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - 540) / 700.0) ** 2 + ((yy - 960) / 1150.0) ** 2)
    a = (np.clip(d - 0.55, 0, 1) * 330).clip(0, 200).astype(np.uint8)
    v = np.zeros((H, W, 4), np.uint8)
    v[..., 3] = a
    return Image.fromarray(v, "RGBA")


def make_can(cfg):
    w, h = CAN_W, CAN_H
    u = np.linspace(-1, 1, w)
    shade = 0.32 + 0.68 * (1 - u ** 2) ** 0.55 + 0.55 * np.exp(-((u + 0.42) / 0.09) ** 2) + 0.25 * np.exp(-((u - 0.78) / 0.07) ** 2)
    shade = np.clip(shade, 0, 1.25)
    arr = np.zeros((h, w, 3), float)
    metal = np.array([204, 210, 222], float)
    lab = np.array(cfg["accent"], float)
    for y in range(h):
        col = lab if 92 <= y <= h - 92 else metal
        arr[y] = col[None, :] * shade[:, None]
    body = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    m = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(m)
    taper = 15
    md.polygon([(taper, 8), (w - taper, 8), (w - 3, 34), (w - 3, h - 30), (w - taper, h - 6), (taper, h - 6), (3, h - 30), (3, 34)], fill=255)
    md.ellipse([taper - 2, 0, w - taper + 2, 26], fill=255)
    body.putalpha(m)
    d = ImageDraw.Draw(body, "RGBA")
    d.ellipse([taper + 4, 3, w - taper - 4, 24], fill=(236, 240, 248, 255), outline=(150, 156, 170, 255), width=2)
    d.ellipse([w / 2 - 20, 8, w / 2 + 20, 20], outline=(140, 146, 160, 255), width=2)
    d.rectangle([4, 92, w - 4, 94], fill=(255, 255, 255, 90))
    d.rectangle([4, h - 94, w - 4, h - 92], fill=(0, 0, 0, 70))
    f = font(F_BOLD, 74)
    d.text((w / 2, h / 2 + 4), cfg.get("label", "AI"), font=f, fill=(255, 255, 255, 245), anchor="mm")
    d.text((w / 2, h / 2 + 62), cfg.get("sublabel", "ZERO SUGAR"), font=font(F_SEMI, 15), fill=(255, 255, 255, 190), anchor="mm")
    return body


def draw_bullet(layer, x, y, a=255):
    d = ImageDraw.Draw(layer, "RGBA")
    for k in range(14):  # trail
        tx = x - 64 - k * 16
        d.ellipse([tx - 14, y - 11 + k * 0.4, tx + 14, y + 11 - k * 0.4], fill=(255, 205, 150, int((70 - k * 5) * a / 255)))
    pts = [(x - 64, y - 13), (x - 18, y - 13), (x + 10, y - 9), (x + 40, y), (x + 10, y + 9), (x - 18, y + 13), (x - 64, y + 13)]
    d.polygon(pts, fill=(196, 112, 56, a))
    d.polygon([(x - 64, y - 13), (x - 18, y - 13), (x + 10, y - 9), (x + 40, y), (x + 4, y - 2), (x - 64, y - 3)], fill=(240, 168, 108, a))
    d.rectangle([x - 64, y - 13, x - 50, y + 13], fill=(150, 86, 40, a))
    for s in (-1, 1):  # shock cone
        d.line([(x + 38, y), (x - 4, y + s * 44)], fill=(255, 255, 255, 70), width=2)
        d.line([(x + 62, y), (x + 14, y + s * 74)], fill=(255, 255, 255, 36), width=2)


def make_particles(cfg, seed=7):
    r = random.Random(seed)
    liq = tuple(cfg["liquid"])
    P = []

    def add(n, a0, a1, s0, s1, ox, oy, delay, rad=(3, 11)):
        for _ in range(n):
            a = math.radians(r.uniform(a0, a1))
            sp = r.uniform(s0, s1)
            foam = r.random() < 0.3
            col = (255, 255, 255) if foam else tuple(int(clamp(c * r.uniform(0.85, 1.15), 0, 255)) for c in liq)
            P.append({"x": ox + r.uniform(-8, 8), "y": oy + r.uniform(-24, 24), "vx": math.cos(a) * sp, "vy": math.sin(a) * sp,
                      "r": r.uniform(*rad), "col": col, "t0": delay + r.uniform(0, 0.12), "life": r.uniform(1.4, 3.0), "foam": foam})

    add(150, -38, 38, 160, 980, CAN_R + 6, IMP_Y, 0.0)
    add(60, 140, 220, 80, 420, CAN_L - 4, IMP_Y, 0.0, (2.5, 8))
    add(50, -112, -68, 120, 520, CAN_CX, CAN_TOP + 20, 0.10, (2.5, 9))
    return P


def build_audio(cfg, model_dir):
    T, TI = cfg["T"], cfg["T_imp"]
    n = int(T * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(3)
    a = np.zeros(n, np.float32)
    # ticking clock accelerating until just before impact
    tt, k = 0.9, 0
    while tt < TI - 0.3:
        i = int(tt * SR)
        L = int(0.05 * SR)
        env = np.exp(-np.arange(L) / (0.012 * SR))
        click = (np.sin(2 * np.pi * 1900 * np.arange(L) / SR) * 0.6 + rng.standard_normal(L) * 0.4) * env * (0.22 + 0.2 * (tt / TI))
        a[i:i + L] += click[: n - i].astype(np.float32)
        rate = 1.6 + 5.0 * (tt / TI) ** 1.6
        tt += 1.0 / rate
    # drone riser
    prog = np.clip(t / TI, 0, 1)
    f0 = 52 + 130 * prog ** 1.5
    ph = 2 * np.pi * np.cumsum(f0) / SR
    drone = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)) * (0.05 + 0.26 * prog ** 1.4)
    nz = rng.standard_normal(n)
    nz = nz - np.convolve(nz, np.ones(40) / 40, mode="same")  # crude high-pass riser
    riser = nz * (0.0 + 0.10 * prog ** 3)
    gate = np.where(t < TI - 0.26, 1.0, 0.0)
    gate = np.convolve(gate, np.ones(200) / 200, mode="same")
    a += ((drone + riser) * gate).astype(np.float32)
    # impact
    i0 = int(TI * SR)
    m = n - i0
    tau = np.arange(m) / SR
    crack = rng.standard_normal(m) * np.exp(-tau / 0.018) * 0.9
    thump = np.sin(2 * np.pi * (34 + 60 * np.exp(-tau / 0.09)) * tau) * np.exp(-tau / 0.42) * 1.0
    ring = (np.sin(2 * np.pi * 1180 * tau) + 0.6 * np.sin(2 * np.pi * 1870 * tau)) * np.exp(-tau / 0.35) * 0.12
    sp = rng.standard_normal(m)
    sp = np.convolve(sp, np.ones(10) / 10, mode="same") - np.convolve(sp, np.ones(120) / 120, mode="same")
    splash = sp * np.exp(-tau / 1.1) * (1 - np.exp(-tau / 0.08)) * 0.55
    a[i0:] += (crack + thump + ring + splash).astype(np.float32)
    # voice
    for st, text in cfg["lines"]:
        x, sr = voice.say(model_dir, text, cfg.get("voice", "af_bella"), cfg.get("speed", 0.95))
        x = x / (np.max(np.abs(x)) or 1) * 0.9
        i = int(st * SR)
        a[i:i + len(x)] += x[: n - i]
    a = a / (np.max(np.abs(a)) or 1) * 0.95
    return a


class Scene:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bg = make_bg(cfg).convert("RGBA")
        self.vig = make_vignette()
        self.can = make_can(cfg)
        self.parts = make_particles(cfg)
        self.TI = cfg["T_imp"]
        self.rng = random.Random(11)

    def bullet_x(self, t):
        s = clamp(t / self.TI)
        x0, xi = 40, CAN_L - 40
        return x0 + (xi - x0) * (0.55 * s + 0.45 * (1 - (1 - s) ** 2))

    def frame(self, t):
        cfg, TI = self.cfg, self.TI
        img = self.bg.copy()
        tau = t - TI
        post = tau >= 0
        # can pose
        dx = dy = ang = 0.0
        if post:
            e = ease_out(tau / 1.6)
            dx, ang, dy = 120 * e, -9 * e, -26 * math.sin(clamp(tau / 1.1) * math.pi)
        can = self.can.copy()
        if post:
            cd = ImageDraw.Draw(can, "RGBA")
            cd.ellipse([-14, IMP_Y - CAN_TOP - 16, 20, IMP_Y - CAN_TOP + 16], fill=(20, 8, 16, 255), outline=(255, 190, 150, 220), width=3)
            cd.polygon([(CAN_W + 14, IMP_Y - CAN_TOP - 30), (CAN_W - 22, IMP_Y - CAN_TOP - 14), (CAN_W - 22, IMP_Y - CAN_TOP + 14),
                        (CAN_W + 14, IMP_Y - CAN_TOP + 34), (CAN_W + 3, IMP_Y - CAN_TOP)], fill=(20, 8, 16, 255))
        pad = 120
        cl = Image.new("RGBA", (CAN_W + pad * 2, CAN_H + pad * 2), (0, 0, 0, 0))
        cl.alpha_composite(can, (pad, pad))
        if ang:
            cl = cl.rotate(ang, resample=Image.BICUBIC, center=(pad + CAN_W / 2, pad + CAN_H - 6))
        # soft shadow
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sh).ellipse([CAN_CX + dx - 150, TABLE_Y - 14, CAN_CX + dx + 150, TABLE_Y + 28], fill=(0, 0, 0, 120))
        img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)))
        img.alpha_composite(cl, (int(CAN_L + dx - pad), int(CAN_TOP + dy - pad)))

        glow = Image.new("RGBA", (W // 4, H // 4), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        fx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        fd = ImageDraw.Draw(fx, "RGBA")
        liq = tuple(cfg["liquid"])
        if not post:
            bx = self.bullet_x(t)
            gd.ellipse([(bx - 60) / 4, (IMP_Y - 30) / 4, (bx + 60) / 4, (IMP_Y + 30) / 4], fill=(255, 190, 120, 130))
            draw_bullet(fx, bx, IMP_Y)
        else:
            bx = CAN_L - 40 + 1500 * tau ** 1.6 + 40
            if bx < W + 100:
                draw_bullet(fx, bx, IMP_Y - tau * 6, int(255 * clamp(1 - tau / 0.6)))
            # shock rings
            for k, dl in enumerate((0.0, 0.12)):
                q = (tau - dl) / 1.0
                if 0 < q < 1:
                    r_ = 20 + 460 * ease_out(q)
                    col = (255, 255, 255) if k == 0 else tuple(cfg["accent"])
                    fd.ellipse([CAN_L - r_, IMP_Y - r_, CAN_L + r_, IMP_Y + r_], outline=col + (int(200 * (1 - q)),), width=max(2, int(8 * (1 - q))))
            # exit mist
            q = clamp(tau / 1.8)
            gd.ellipse([(CAN_R - 20) / 4, (IMP_Y - 40 - 150 * q) / 4, (CAN_R + 60 + 480 * q) / 4, (IMP_Y + 40 + 150 * q) / 4], fill=liq + (int(170 * (1 - q)),))
            gd.ellipse([(CAN_L - 120 * q - 30) / 4, (IMP_Y - 30 - 60 * q) / 4, (CAN_L + 20) / 4, (IMP_Y + 30 + 60 * q) / 4], fill=liq + (int(110 * (1 - q)),))
            # particles
            for p in self.parts:
                pt = tau - p["t0"]
                if pt < 0 or pt > p["life"]:
                    continue
                x = p["x"] + p["vx"] * pt * 0.9 + dx * (0.5 if p["x"] < CAN_CX else 0)
                y = p["y"] + p["vy"] * pt * 0.9 + 0.5 * 330 * pt * pt
                if y > TABLE_Y - 2:
                    y = TABLE_Y - 2
                    fade = clamp(1 - (pt - 0.0) / p["life"])
                else:
                    fade = clamp(1 - pt / p["life"]) ** 0.7
                a_ = int(235 * fade)
                rr = p["r"] * (0.55 + 0.45 * fade)
                tx, ty = x - p["vx"] * 0.045, y - (p["vy"] + 330 * pt) * 0.045
                fd.line([(tx, ty), (x, y)], fill=p["col"] + (int(a_ * 0.55),), width=max(1, int(rr * 1.4)))
                fd.ellipse([x - rr, y - rr, x + rr, y + rr], fill=p["col"] + (a_,))
        glow = glow.filter(ImageFilter.GaussianBlur(9)).resize((W, H), Image.BILINEAR)
        img.alpha_composite(glow)
        img.alpha_composite(fx)
        if post and tau < 0.14:
            fl = int(235 * (1 - tau / 0.14) ** 1.5)
            img.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, fl)))
        img.alpha_composite(self.vig)
        # camera: slow push-in toward the impact, shake on hit
        z = 1 + 0.55 * ease_io(t / TI) if not post else 1.55 + 0.08 * clamp(tau / 3.0)
        fxp, fyp = 600, 1000
        cw, ch = W / z, H / z
        sx, sy = 0.0, 0.0
        if post:
            amp = 14 * math.exp(-tau / 0.16)
            sx, sy = self.rng.uniform(-amp, amp), self.rng.uniform(-amp, amp)
        x0 = clamp(fxp - cw / 2 + sx, 0, W - cw)
        y0 = clamp(fyp - ch / 2 + sy, 0, H - ch)
        img = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.BICUBIC)
        self.overlay(img, t)
        return img

    def overlay(self, img, t):
        cfg, TI = self.cfg, self.TI
        T = cfg["T"]
        d = ImageDraw.Draw(img, "RGBA")
        acc = tuple(cfg["accent"])
        # progress bar
        pr = clamp(t / TI)
        d.rounded_rectangle([90, 70, W - 90, 84], radius=7, fill=(255, 255, 255, 40))
        if pr > 0:
            d.rounded_rectangle([90, 70, 90 + (W - 180) * pr, 84], radius=7, fill=acc + (255,))
        if t < TI:
            txt = cfg.get("kicker", "DON'T BLINK")
            pulse = 0.65 + 0.35 * math.sin(t * 7)
            f = font(F_BOLD, 108)
            d.text((W / 2, 230), txt, font=f, fill=(255, 255, 255, 255), anchor="mm", stroke_width=8, stroke_fill=(10, 6, 24, 255))
            left = max(0.0, TI - t)
            red = left < 1.0
            ptxt = f"IMPACT IN {left:0.1f}s"
            f2 = font(F_MONO, 54)
            tw = d.textlength(ptxt, font=f2) + 70
            col = (255, 70, 70) if red else acc
            d.rounded_rectangle([(W - tw) / 2, 330, (W + tw) / 2, 410], radius=40, fill=(10, 6, 24, 215), outline=col + (int(255 * pulse),), width=4)
            d.text((W / 2, 370), ptxt, font=f2, fill=(255, 255, 255, 255), anchor="mm")
        else:
            q = clamp((t - TI) / 0.35)
            f = font(F_BOLD, 96)
            lines = cfg.get("post_kicker", ["THE PART YOUR", "EYES MISS"])
            for i, ln in enumerate(lines):
                d.text((W / 2, 220 + i * 112 - (1 - q) * 40), ln, font=f, fill=(255, 255, 255, int(255 * q)), anchor="mm", stroke_width=8,
                       stroke_fill=(10, 6, 24, int(255 * q)))
        # caption for the current voice line
        for i, (st, text) in enumerate(cfg["lines"]):
            en = cfg["lines"][i + 1][0] - 0.05 if i + 1 < len(cfg["lines"]) else T
            dur = min(en - st, 0.5 + 0.075 * len(text))
            if st - 0.05 <= t <= st + dur:
                fcap = font(F_BOLD, 80)
                words, ls, cur = text.split(), [], ""
                for w in words:
                    tr = (cur + " " + w).strip()
                    if d.textlength(tr, font=fcap) <= W - 140:
                        cur = tr
                    else:
                        ls.append(cur)
                        cur = w
                ls.append(cur)
                for j, ln in enumerate(ls):
                    d.text((W / 2, 1520 + j * 96), ln, font=fcap, fill=(255, 245, 200, 255), anchor="mm", stroke_width=10, stroke_fill=(10, 6, 24, 255))
                break
        # fade to next scene colour
        if t > T - 0.3:
            q = (t - (T - 0.3)) / 0.3
            img.alpha_composite(Image.new("RGBA", (W, H), tuple(cfg["bg"]) + (int(255 * q),)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    args = ap.parse_args()
    cfg = json.load(open(args.cfg))
    sc = Scene(cfg)
    if args.stills:
        ts = cfg.get("still_times", [0, 2, 4, 5.0, 5.3, 5.6, 6.2, 7.4, 8.8])
        fr = [sc.frame(t).convert("RGB").resize((300, 533)) for t in ts]
        sheet = Image.new("RGB", (300 * len(fr), 533))
        for i, f_ in enumerate(fr):
            sheet.paste(f_, (i * 300, 0))
        sheet.save(args.stills)
        return
    audio = build_audio(cfg, args.model)
    wav = os.path.splitext(args.out)[0] + "_hook.wav"
    sf.write(wav, audio, SR)
    n = int(cfg["T"] * FPS)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                           "-shortest", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(sc.frame(i / FPS).convert("RGB").tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    print(args.out, cfg["T"])


if __name__ == "__main__":
    main()
