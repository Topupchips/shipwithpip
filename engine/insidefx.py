"""'Hidden mechanism of something you use daily' reels with real tests.
usage: python insidefx.py qr|jpeg out.mp4 --model <kokoro-multi-lang-v1_0> [--stills png]
"""
import argparse
import io
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qrsim as Q  # noqa: E402
from learnfx import write_post  # noqa: E402
from scenes import CYAN, GREEN, PINK, VIOLET, YEL, ORANGE, Reel2, chime, make_audio, render  # noqa: E402
from simfx import F_BOLD, F_MONO, INK, VS, clamp, font, tone  # noqa: E402

BG = (10, 6, 28)


# ------------------------------------------------------------------ QR
class QrReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.mats = cfg["mats"]
        self.res = cfg["res"]
        self.cache = {}

    def idx(self, t):
        c = self.cfg
        i = int((t - c["t0"]) / c["hold"]) if t >= c["t0"] else 0
        return max(0, min(len(c["KS"]) - 1, i))

    def k_now(self, t):
        return self.cfg["KS"][self.idx(t)]

    def panel(self, lv, k):
        key = (lv, k)
        if key not in self.cache:
            arr = Q.render(self.mats[lv], k, px=10, quiet=2)
            self.cache[key] = Image.fromarray(arr, "RGB").resize((440, 440), Image.NEAREST)
        return self.cache[key]

    def content(self, t):
        im = Image.new("RGB", (VS, VS), BG)
        d = ImageDraw.Draw(im, "RGBA")
        k = self.k_now(t)
        i = self.idx(t)
        for lv, x0, name, nom in (("L", 20, "LOW", "7%"), ("H", 500, "HIGH", "30%")):
            ok = self.res[lv][i]
            col = GREEN if ok else PINK
            d.rounded_rectangle([x0 - 4, 70, x0 + 444, 580], radius=20, fill=(14, 10, 34, 255), outline=col + (230,), width=4)
            d.text((x0 + 220, 100), f"ERROR CORRECTION: {name}", font=font(F_MONO, 24), fill=(235, 230, 255, 255), anchor="mm")
            im.paste(self.panel(lv, k), (x0, 130))
            d = ImageDraw.Draw(im, "RGBA")
            d.rounded_rectangle([x0 + 70, 600, x0 + 370, 680], radius=20, fill=col + (60,), outline=col + (255,), width=3)
            d.text((x0 + 220, 640), "SCANS" if ok else "FAILS", font=font(F_BOLD, 46), fill=col + (255,), anchor="mm", stroke_width=3, stroke_fill=INK)
        n = self.cfg["n"]
        cov = 100.0 * k * k / (n * n)
        d.text((480, 770), f"LOGO COVERS {cov:0.0f}% OF THE CODE", font=font(F_BOLD, 40), fill=YEL + (255,), anchor="mm", stroke_width=4, stroke_fill=INK)
        d.text((480, 835), "REAL SCAN TEST, OPENCV DECODER", font=font(F_MONO, 22), fill=(200, 195, 240, 255), anchor="mm")
        return im

    def stats(self, t):
        k = self.k_now(t)
        n = self.cfg["n"]
        cov = 100.0 * k * k / (n * n)
        i = self.idx(t)
        lo, hi = self.res["L"][i], self.res["H"][i]
        return [(f"COVERED {cov:0.0f}%", YEL), (f"LOW {'OK' if lo else 'FAIL'}  HIGH {'OK' if hi else 'FAIL'}", GREEN if (lo and hi) else (YEL if hi else PINK))], clamp(cov / 30.0), "LOGO SIZE", PINK


def build_qr(model):
    n = None
    mats, res, firstfail, maxok = {}, {}, {}, {}
    KS = [0] + list(range(1, 27, 2))   # odd sizes keep the patch exactly centered
    for lv in "LMQH":
        mat, n = Q.modules(lv)
        mats[lv] = mat
        res[lv] = [Q.scans(Q.render(mat, k, px=10, quiet=4)) for k in KS]
        ff = next((i for i, v in enumerate(res[lv]) if not v), None)
        firstfail[lv] = KS[ff] if ff is not None else None
        maxok[lv] = KS[max(i for i in range(len(KS)) if all(res[lv][:i + 1]))]
    pct = lambda k: 100.0 * k * k / (n * n)  # noqa: E731
    print({lv: (maxok[lv], round(pct(maxok[lv]), 1), firstfail[lv]) for lv in "LMQH"})
    print({lv: [int(v) for v in res[lv]] for lv in "LH"})
    t0, hold = 8.4, 1.1
    kL, kH = firstfail["L"], firstfail["H"]
    tL, tH = t0 + KS.index(kL) * hold, t0 + KS.index(kH) * hold
    cfg = dict(T=60.0, c1=GREEN, c2=PINK, mats=mats, res=res, n=n, KS=KS, t0=t0, hold=hold, payoff=round(tH, 1), seed=171,
               still_times=[0.5, t0 + 0.5, tL - 1, tL + 1.5, tH - 1, tH + 1.5, tH + 5],
               header=lambda tt: (([("COVER A QR CODE.", (255, 255, 255)), ("IT STILL SCANS?", YEL)], "REAL TEST") if tt < tH else
                                  ([("HIGH SURVIVES.", (255, 255, 255)), ("LOW DOESN'T.", PINK)], "ERROR CORRECTION")))
    lines = [(0.2, "Cover part of a QR code and it can still scan. Here is a real test."),
             (4.4, "Same link, two settings. Low repair data on the left. High on the right."),
             (9.0, "Now a logo grows over the middle."),
             (tL - 0.9, f"At {pct(kL):0.0f} percent covered, the low one fails."),
             (tL + 2.6, "The high one keeps scanning. Extra repair data lets missing pieces get rebuilt."),
             (tH + 2.4, f"At {pct(kH):0.0f} percent, even the high one gives up."),
             (tH + 6.0, f"It survived up to {pct(maxok['H']):0.0f} percent. That is why logos can sit on QR codes. Comment QR for the prompt.")]
    buf, vo = make_audio(cfg, lines, model)
    chime(buf, cfg["payoff"])
    cfg["T"] = vo[-1][1] + 2.4
    reel = QrReel(cfg, vo)
    post = dict(title="INSIDE-1  cover a QR code and it still scans (real test)",
                caption=f"A QR code stores extra repair data, so a damaged scan can be rebuilt. I tested it for real: same link, a logo growing over the middle, low vs high error correction. Low failed at {pct(kL):.0f}% covered. High kept scanning up to {pct(maxok['H']):.0f}% and failed at {pct(kH):.0f}%. The official levels recover roughly 7% (L), 15% (M), 25% (Q) and 30% (H) of the code's data words, but how much visible area you can cover depends on where the damage lands.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment QR and I'll send the prompt so you can build it.",
                tags=["qrcode", "tech", "howitworks", "science", "coding", "shipwithpip"],
                pinned="Build me a single-file HTML page with a QR code generator that has an error correction dropdown (L, M, Q, H), a slider that grows a solid square logo over the center, and a live check that tries to decode the code (use a JS QR library like qrcode-generator for encoding and jsQR for decoding) and shows SCANS or FAILS.",
                notes=f"Real experiment, run in code: QR version 6 ({n}x{n} modules), payload '{Q.TEXT}', encoded with OpenCV QRCodeEncoder at levels L, M, Q, H. A solid square patch is centered on the symbol and grown one module at a time; each size is rendered with a 4-module quiet zone and decoded with OpenCV QRCodeDetector. Patch size is shown as % of the symbol's area ({n}x{n} modules).\nOdd patch sizes only (keeps it centered). Max contiguous patch that still scanned (all smaller tested sizes also scanned): " + ", ".join(f"{lv} {maxok[lv]} modules ({pct(maxok[lv]):.1f}%)" for lv in "LMQH") + f". First failure: " + ", ".join(f"{lv} at {firstfail[lv]} modules ({pct(firstfail[lv]):.1f}%)" for lv in "LMQH") + ".\nNominal levels (Denso Wave via https://maketecheasier.com/how-qr-codes-survive-damage/): L ~7%, M ~15%, Q ~25%, H ~30% of codewords, not visible area; results depend on damage pattern. My measured coverage is lower than nominal because one solid patch concentrates damage and OpenCV's decoder is less forgiving than many phone scanners. Phone scanners may differ. M and Q measured the same here at this size.")
    return reel, buf, cfg, post


# ------------------------------------------------------------------ JPEG
def make_photo(S=960, seed=7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    sky = np.zeros((S, S, 3), np.float32)
    g = yy / S
    sky[..., 0] = 255 * (0.95 - 0.55 * g)
    sky[..., 1] = 255 * (0.55 - 0.30 * g)
    sky[..., 2] = 255 * (0.45 + 0.35 * g)
    sun = np.exp(-((xx - 640) ** 2 + (yy - 430) ** 2) / (2 * 110 ** 2))
    sky += sun[..., None] * np.array([120, 90, 30])
    sd = np.hypot(xx - 640, yy - 430)
    sky[sd < 58] = (255, 244, 205)
    # clouds
    c = np.zeros((S, S), np.float32)
    for sc, amp in ((120, 1.0), (60, 0.6), (30, 0.35), (15, 0.2)):
        small = rng.random((S // sc + 2, S // sc + 2)).astype(np.float32)
        c += amp * np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC), np.float32) / 255
    c = np.clip((c - 1.1) * 1.6, 0, 1) * (yy < 520)
    sky = sky * (1 - 0.45 * c[..., None]) + 0.45 * c[..., None] * np.array([255, 200, 190])
    img = sky
    # mountain ridges with texture
    for k, (base, amp, col) in enumerate(((520, 90, (92, 66, 112)), (600, 70, (58, 42, 84)), (690, 50, (30, 24, 56)))):
        ridge = base + sum(amp / (2 ** i) * np.sin(xx[0] * (0.006 * 2 ** i) + k * 1.7 + i) for i in range(5))
        ridge = ridge + rng.normal(0, 2.0, S)
        m = yy > ridge[None, :]
        tex = rng.normal(0, 7, (S, S, 1))
        img = np.where(m[..., None], np.array(col, np.float32) + tex, img)
    # foreground trees (high frequency detail)
    for _ in range(260):
        x = rng.integers(0, S); h = rng.integers(40, 150); y = S - rng.integers(0, 130)
        w = h // 5 + 4
        pts = [(x, y - h), (x - w, y), (x + w, y)]
        ImgD = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        ImageDraw.Draw(ImgD).polygon(pts, fill=(int(8 + rng.integers(0, 14)), int(14 + rng.integers(0, 22)), int(20 + rng.integers(0, 22))))
        img = np.asarray(ImgD, np.float32)
    img += rng.normal(0, 3.0, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def jpeg_at(photo, q):
    b = io.BytesIO()
    photo.save(b, "JPEG", quality=int(q))
    n = b.tell()
    b.seek(0)
    return Image.open(b).convert("RGB"), n


class JpegReel(Reel2):
    def __init__(self, cfg, vo):
        super().__init__(cfg, vo)
        self.photo = cfg["photo"]
        self.cache = cfg["cache"]

    def q_at(self, t):
        pts = self.cfg["qstops"]
        if t <= pts[0][0]:
            return pts[0][1]
        for (a, qa), (b, qb) in zip(pts, pts[1:]):
            if a <= t <= b:
                # log interpolation of quality
                f = (t - a) / (b - a)
                return int(round(math.exp(math.log(qa) + (math.log(qb) - math.log(qa)) * f)))
        return pts[-1][1]

    def content(self, t):
        q = self.q_at(t)
        jimg, size = self.cache[q]
        im = self.photo.copy()
        right = jimg.crop((480, 0, 960, 960))
        im.paste(right, (480, 0))
        d = ImageDraw.Draw(im, "RGBA")
        d.line([(480, 0), (480, 960)], fill=(255, 255, 255, 230), width=4)
        for x, txt in ((20, "ORIGINAL"), (940, f"JPEG QUALITY {q}")):
            d.rounded_rectangle([x - 10 if x < 480 else x - 320, 18, x + 190 if x < 480 else x + 10, 62], radius=14, fill=(10, 6, 28, 200))
            d.text((x + 90 if x < 480 else x - 155, 40), txt, font=font(F_MONO, 22), fill=(255, 255, 255, 255), anchor="mm")
        return im

    def stats(self, t):
        q = self.q_at(t)
        _, size = self.cache[q]
        raw = self.cfg["raw"]
        col = GREEN if q >= 40 else (YEL if q >= 12 else PINK)
        return [(f"JPEG SIZE {size / 1000:,.0f} KB", col), (f"{raw / size:0.0f}x SMALLER", YEL)], size / raw, "SIZE VS UNCOMPRESSED", col


def build_jpeg(model):
    photo = make_photo()
    raw = 960 * 960 * 3
    cache = {}
    for q in range(1, 96):
        cache[q] = jpeg_at(photo, q)
    sz = {q: cache[q][1] for q in (95, 75, 40, 20, 10, 5, 2)}
    png = io.BytesIO(); photo.save(png, "PNG", optimize=True)
    print("raw", raw, "png", png.tell(), "sizes", sz)
    ratio = lambda q: raw / cache[q][1]  # noqa: E731
    qstops = [(0.0, 95), (6.0, 95), (9.0, 75), (13.5, 75), (17.5, 30), (20.5, 12), (23.5, 6), (26.5, 2), (60.0, 2)]
    tb = 20.5
    cfg = dict(T=60.0, c1=CYAN, c2=ORANGE, photo=photo, cache=cache, raw=raw, qstops=qstops, payoff=tb + 1.0, seed=181,
               still_times=[0.5, 8.5, 12, 16, 20, 22.5, 25.5, 28],
               header=lambda tt: (([("A 2.8 MB PICTURE.", (255, 255, 255)), ("SHRINK IT.", YEL)], "WATCH WHAT BREAKS") if tt < tb else
                                  ([("THE 8X8 TRICK", (255, 255, 255)), ("IN EVERY PHOTO", ORANGE)], "HOW JPEG WORKS")))
    lines = [(0.2, "This picture is two point eight megabytes, uncompressed."),
             (4.2, "Now I shrink the right half. Can you spot the difference?"),
             (8.8, f"Quality 75. Only {sz[75] / 1000:0.0f} kilobytes. That is {ratio(75):0.0f} times smaller."),
             (14.2, "JPEG splits the picture into little eight by eight blocks, and rounds away fine detail your eyes barely notice."),
             (21.4, f"Push it too far, and you see the blocks. Quality 2 is {ratio(2):0.0f} times smaller, but ugly."),
             (27.6, "Every photo you send gets this treatment. Comment SHRINK for the prompt.")]
    buf, vo = make_audio(cfg, lines, model)
    chime(buf, cfg["payoff"])
    cfg["T"] = vo[-1][1] + 2.4
    reel = JpegReel(cfg, vo)
    post = dict(title="INSIDE-2  how JPEG shrinks a photo (real test)",
                caption=f"A 2.8 MB picture (uncompressed, a synthetic landscape) shrinks to about {sz[75] / 1000:.0f} KB at JPEG quality 75, {ratio(75):.0f}x smaller, with the original on the left half and the JPEG on the right. JPEG splits the picture into 8x8 blocks and rounds away fine detail your eyes barely notice. Push the quality too low and the blocks show.\n\nBuilding your own app? Use code AT100 on Base44 (link in bio).\n\nComment SHRINK and I'll send the prompt so you can build it.",
                tags=["jpeg", "tech", "howitworks", "photography", "coding", "shipwithpip"],
                pinned="Build me a single-file HTML page where I can upload a photo and drag a quality slider from 1 to 100. Show the original on the left half and a JPEG re-encoded (canvas.toBlob with image/jpeg and the slider value) on the right half, with the file size in KB and how many times smaller than the original it is.",
                notes=f"Real test in code (Pillow): a synthetic 960x960 landscape (gradient sky, clouds, ridges, trees, mild noise) generated procedurally, not a real photograph. Uncompressed size = 960*960*3 = {raw:,} bytes. JPEG sizes at quality 95/75/40/20/10/5/2 = {sz} bytes. PNG (lossless) = {png.tell():,} bytes. Sizes depend on image content; a real photo would differ.\nMethod facts (JPEG 8x8 blocks, DCT, quantization, high-frequency detail rounded more coarsely): https://www.img2go.com/blog/how-does-jpeg-compression-work. 'Looks the same' is for the viewer to judge from the split-screen; no perceptual metric is claimed.")
    return reel, buf, cfg, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    reel, buf, cfg, post = {"qr": build_qr, "jpeg": build_jpeg}[a.which](a.model)
    render(reel, a.out, buf, cfg["T"], a.stills)
    if not a.stills:
        write_post(os.path.splitext(a.out)[0] + "_post.txt", post["title"], post["caption"], post["tags"], post["pinned"], post["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
