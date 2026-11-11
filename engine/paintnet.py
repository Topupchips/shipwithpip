"""Coordinate network that learns a picture (real numpy training)."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from learn import MLP

R = 96   # target resolution
EV = 144  # eval resolution


def target():
    S = 4
    n = R * S
    yy, xx = np.mgrid[0:n, 0:n] / n
    top, bot = np.array([60, 30, 140]), np.array([255, 140, 90])
    img = (top[None, None, :] * (1 - yy[..., None]) + bot[None, None, :] * yy[..., None]).astype(np.float32)
    im = Image.fromarray(img.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    s = n
    # sun
    d.ellipse([0.58 * s, 0.14 * s, 0.86 * s, 0.14 * s + 0.28 * s], fill=(255, 226, 120))
    # hills
    d.polygon([(0, 0.78 * s), (0.3 * s, 0.6 * s), (0.62 * s, 0.76 * s), (0.85 * s, 0.64 * s), (s, 0.74 * s), (s, s), (0, s)], fill=(30, 18, 70))
    # heart
    cx, cy, k = 0.36 * s, 0.4 * s, 0.17 * s
    pts = []
    for i in range(200):
        t = i / 200 * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x / 16 * k, cy - y / 16 * k))
    d.polygon(pts, fill=(255, 70, 140))
    d.ellipse([cx - 0.11 * s, cy - 0.12 * s, cx - 0.06 * s, cy - 0.075 * s], fill=(255, 190, 220))
    im = im.resize((R, R), Image.LANCZOS)
    return im, np.asarray(im, np.float32) / 255.0


FREQ = [1, 2, 4, 8]


def feats(xy):
    f = [xy]
    for q in FREQ:
        f += [np.sin(2 * math.pi * q * xy), np.cos(2 * math.pi * q * xy)]
    return np.concatenate(f, 1)


def coords(n):
    g = (np.arange(n) + 0.5) / n
    gx, gy = np.meshgrid(g, g)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def run(epochs=1500, snaps=None, seed=1, width=64, lr=0.004):
    im, arr = target()
    X = feats(coords(R))
    Y = arr.reshape(-1, 3)
    Xe = feats(coords(EV))
    net = MLP([X.shape[1], width, width, width, 3], seed=seed, out="linear", scale=0.12)
    out = []
    for e in range(epochs + 1):
        if snaps is None or e in snaps:
            pred = np.clip(net.predict(X), 0, 1)
            mae = float(np.mean(np.abs(pred - Y)))
            mse = float(np.mean((net.predict(X) - Y) ** 2))
            ev = np.clip(net.predict(Xe), 0, 1).reshape(EV, EV, 3)
            out.append((e, mse, mae, (ev * 255).astype(np.uint8)))
        net.train_step(X, Y, lr)
    return im, out
