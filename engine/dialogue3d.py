"""3D low-poly dialogue reels: dev + orange Claude bot, HTML UI cards, rendered in headless Chromium (three.js).
usage: python dialogue3d.py script.json out.mp4 --model <kokoro dir> [--stills t1,t2,...]
"""
import argparse
import functools
import http.server
import json
import os
import re
import subprocess
import sys
import threading

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice  # noqa: E402
from make_reel import music_bed  # noqa: E402
from simfx import SR, tone  # noqa: E402
from learnfx import write_post  # noqa: E402

SITE = os.environ.get("G3D_DIR", "/tmp/claude-0/-home-claude/a75793b8-e870-5638-9e92-ffc2e9c0e649/scratchpad/g3d")
VOICES = {"dev": ("am_adam", 0.97), "bot": ("af_heart", 1.0)}
FPS = 30


def synth(model, text, who):
    import hashlib
    key = hashlib.md5((text + who + str(VOICES[who])).encode()).hexdigest()
    cp = os.path.join(SITE, "cache_" + key + ".npy")
    if os.path.exists(cp):
        return np.load(cp)
    x = _synth(model, text, who)
    np.save(cp, x)
    return x


def _synth(model, text, who):
    v, sp = VOICES[who]
    parts = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        x, sr = voice.say(model, s, v, sp)
        assert sr == SR, sr
        parts += [x / (np.max(np.abs(x)) or 1) * 0.95, np.zeros(int(SR * (0.3 if s.endswith("?") else 0.22)), np.float32)]
    return np.concatenate(parts[:-1])


def env_of(x):
    n = int(len(x) / SR * FPS) + 2
    out = []
    for i in range(n):
        seg = x[int(i / FPS * SR): int((i + 1) / FPS * SR)]
        out.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    e = np.array(out)
    e = e / (np.percentile(e, 95) or 1)
    e = np.clip(e, 0, 1)
    e = np.convolve(e, [0.25, 0.5, 0.25], "same")
    return [round(float(v), 2) for v in e]


def build(script, model):
    sc = json.load(open(script))
    t = 0.35
    clips, beats = [], []
    for b in sc["beats"]:
        x = synth(model, b.get("say", b["text"]), b["who"])
        dur = len(x) / SR
        words = b["text"].split()
        w = np.array([len(z) + 2 for z in words], float)
        beats.append(dict(who=b["who"], text=b["text"], mood=b.get("mood", "neutral"), shot=b.get("shot", "two"), t0=round(t, 3), t1=round(t + dur, 3), env=env_of(x),
                          words=words, wcum=[round(float(v), 4) for v in np.cumsum(w) / w.sum()], **({"card": b["card"]} if "card" in b else {})))
        clips.append((t, x))
        t += dur + b.get("gap", 0.28)
    last = beats[-1]
    end_at = last["t0"] + 0.15
    T = last["t1"] + 2.4
    buf = np.zeros(int(T * SR) + SR, np.float32)
    bed = music_bed(T, sc.get("seed", 7))
    buf[:len(bed)] += bed[: len(buf)] * 0.30
    for t0, x in clips:
        i = int(t0 * SR)
        buf[i:i + len(x)] += x
    prev = None
    for b in beats:
        if "card" in b and b["card"] != prev and b["card"] != "none":
            for k, f in enumerate((1, 1.5)):
                tone(buf, b["t0"] + k * 0.05, 0.25, 700 * f, 900 * f, 0.05, "sine")
        if "card" in b:
            prev = b["card"]
    buf = buf[: int(T * SR)]
    tl = dict(T=T, endAt=end_at, beats=beats, cardSpecs=sc["cards"], end=sc["end"])
    return sc, tl, buf


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve():
    h = functools.partial(Quiet, directory=SITE)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stills")
    a = ap.parse_args()
    sc, tl, buf = build(a.script, a.model)
    print("T", round(tl["T"], 1), "beats", len(tl["beats"]), "endAt", tl["endAt"])
    from playwright.sync_api import sync_playwright
    srv = serve()
    port = srv.server_address[1]
    with sync_playwright() as p:
        br = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--enable-webgl"])
        pg = br.new_page(viewport={"width": 1080, "height": 1920}, device_scale_factor=1)
        pg.on("console", lambda m: print("console:", m.text) if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: print("pageerror:", e))
        pg.goto(f"http://127.0.0.1:{port}/index.html")
        pg.wait_for_function("window.ready === true", timeout=60000)
        pg.evaluate("tl => window.loadTL(tl)", tl)
        pg.wait_for_timeout(600)
        if a.stills:
            for t in [float(v) for v in a.stills.split(",")]:
                pg.evaluate("t => window.setFrame(t)", t)
                pg.screenshot(path=f"{os.path.splitext(a.out)[0]}_s{t:05.1f}.png")
            br.close()
            return
        wav = os.path.splitext(a.out)[0] + "_a.wav"
        sf.write(wav, buf / (np.max(np.abs(buf)) or 1) * 0.92, SR)
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", "-", "-i", wav,
                               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-15:TP=-1.2:LRA=11",
                               "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
        n = int(tl["T"] * FPS)
        for i in range(n):
            pg.evaluate("t => window.setFrame(t)", i / FPS)
            ff.stdin.write(pg.screenshot(type="jpeg", quality=90))
            if i % 150 == 0:
                print("frame", i, "/", n, flush=True)
        ff.stdin.close()
        ff.wait()
        br.close()
        os.remove(wav)
    write_post(os.path.splitext(a.out)[0] + "_post.txt", sc["title"], sc["caption"], sc["tags"], sc["pinned"], sc["notes"])
    print("done", a.out)


if __name__ == "__main__":
    main()
