"""Five more real sims: walk, shortcut, drone, password, keepups."""
import numpy as np
from sims2 import bmlp, nw, breed

# ====================================================================== 1. walk (soft-body creatures, 4 nodes 6 muscles)
WN, WM = 4, 6; WSTEPS = 900; WREC = 3
MUSC = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
# genome: node x,y (8), node friction (4), per muscle: short, long, period, phase (24) -> 36


def walk_decode(G):
    n = len(G); pos = G[:, :8].reshape(n, 4, 2) * 60; fr = 1 / (1 + np.exp(-G[:, 8:12]))
    m = G[:, 12:36].reshape(n, 6, 4); short = 40 + 40 * np.abs(m[:, :, 0]); long = short + 20 + 60 * np.abs(m[:, :, 1]); per = 20 + 40 * np.abs(m[:, :, 2]); ph = m[:, :, 3] * 3.0
    return pos, fr, short, long, per, ph


def walk_episode(G, record=True):
    n = len(G); pos, fr, short, long, per, ph = walk_decode(G); pos = pos.copy(); pos[:, :, 1] -= pos[:, :, 1].min(axis=1, keepdims=True) - 2; pos[:, :, 0] -= pos[:, :, 0].mean(axis=1, keepdims=True)
    prev = pos.copy(); rec = []; A = np.array([a for a, b in MUSC]); B = np.array([b for a, b in MUSC]); g = 0.5; dt = 1.0
    for s in range(WSTEPS):
        tgt = np.where(np.sin(2 * np.pi * s / per + ph) > 0, long, short)
        vel = (pos - prev) * 0.97; sp = np.linalg.norm(vel, axis=2, keepdims=True); vel = vel * np.minimum(1, 5.0 / (sp + 1e-9)); prev = pos.copy(); pos = pos + vel; pos[:, :, 1] += g
        for _ in range(3):
            d = pos[:, B] - pos[:, A]; L = np.linalg.norm(d, axis=2) + 1e-6; corr = (L - tgt) / L * 0.45; dv = d * corr[:, :, None] * 0.5
            for k in range(6): pos[:, A[k]] += dv[:, k]; pos[:, B[k]] -= dv[:, k]
        below = pos[:, :, 1] > 0
        # ground: push up, friction on x proportional to node friction
        vx = pos[:, :, 0] - prev[:, :, 0]; pos[:, :, 0] = np.where(below, pos[:, :, 0] - vx * fr, pos[:, :, 0]); pos[:, :, 1] = np.where(below, 0, pos[:, :, 1])
        if record and s % WREC == 0: rec.append(np.round(pos, 1))
    fit = pos[:, :, 0].mean(axis=1)
    return fit, (np.stack(rec, axis=1) if record else None)  # (n, T, 4, 2)


def walk_sim(seed=1, npop=100, gens=40):
    rng = np.random.default_rng(seed); G = rng.normal(0, 1, (npop, 36)); out = []
    for g in range(1, gens + 1):
        fit, rec = walk_episode(G); order = np.argsort(-fit)
        out.append(dict(g=g, best=float(fit.max()), mean=float(fit.mean()), runs=rec.tolist(), fit=fit.tolist(), fr=(1 / (1 + np.exp(-G[:, 8:12]))).tolist()))
        G = breed(G, fit + rng.normal(0, 1e-3, npop), rng, ne=10, sig=0.3, pm=0.2)
    return dict(gens=out, stats=dict(seed=seed, best=[(o["g"], round(o["best"], 1), round(o["mean"], 1)) for o in out]))


# ====================================================================== 2. shortcut (race track with a gap in the inner wall)
OUT = (20, 20, 980, 980); INN = (300, 300, 700, 700); GAPX = (440, 560)  # corridor through the inner block
SEGS = [((20, 20), (980, 20)), ((980, 20), (980, 980)), ((980, 980), (20, 980)), ((20, 980), (20, 20)),
        ((300, 300), (440, 300)), ((560, 300), (700, 300)), ((700, 300), (700, 700)), ((700, 700), (560, 700)), ((440, 700), (300, 700)), ((300, 700), (300, 300)),
        ((440, 300), (440, 700)), ((560, 300), (560, 700))]
GATES = [((620, 20), (620, 300)), ((840, 20), (840, 300)), ((980, 500), (700, 500)), ((840, 980), (840, 700)), ((500, 980), (500, 700)), ((160, 980), (160, 700)), ((20, 500), (300, 500)), ((160, 20), (160, 300))]
SGATE = ((440, 500), (560, 500))  # crossing this while on the bottom half jumps straight to the finish
SSTEPS = 1400; RAYS = np.radians([-70, -35, 0, 35, 70])
_S = np.array([[a[0], a[1], b[0], b[1]] for a, b in SEGS], float)


def raycast(px, py, ang):
    """px,py,ang (n,) -> (n,5) distances up to 350."""
    n = len(px); ox = px[:, None, None]; oy = py[:, None, None]; th = ang[:, None] + RAYS[None, :]; dx = np.cos(th)[:, :, None]; dy = np.sin(th)[:, :, None]
    x1, y1, x2, y2 = _S[None, None, :, 0], _S[None, None, :, 1], _S[None, None, :, 2], _S[None, None, :, 3]
    ex, ey = x2 - x1, y2 - y1; den = dx * ey - dy * ex; den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    t = ((x1 - ox) * ey - (y1 - oy) * ex) / den; u = ((x1 - ox) * dy - (y1 - oy) * dx) / den
    ok = (t > 0) & (u >= 0) & (u <= 1); t = np.where(ok, t, 350); return np.minimum(t.min(axis=2), 350)


def crossed(p0, p1, seg):
    (ax, ay), (bx, by) = seg; d = (bx - ax, by - ay)
    def side(p): return d[0] * (p[1] - ay) - d[1] * (p[0] - ax)
    s0, s1 = side(p0), side(p1);
    if s0 * s1 > 0: return False
    # projection within segment
    t = ((p1[0] - ax) * d[0] + (p1[1] - ay) * d[1]) / (d[0] ** 2 + d[1] ** 2); return 0 <= t <= 1


def shortcut_episode(W, record=True):
    n = len(W); f = bmlp(W, 6, 8, 2); x = np.full(n, 500.0); y = np.full(n, 160.0); ang = np.zeros(n); v = np.full(n, 2.0); alive = np.ones(n, bool)
    prog = np.zeros(n, int); laps = np.zeros(n, int); used = np.zeros(n, bool); bumps = np.zeros(n, int); rec = [[] for _ in range(n)]; usedat = np.full(n, -1)
    for s in range(SSTEPS):
        d = raycast(x, y, ang) / 350; o = f(np.column_stack([d, v / 7.5])); steer = o[:, 0] * 0.09; thr = o[:, 1]
        ang = ang + steer * alive; v = np.clip(v + thr * 0.35 - 0.04, 0.5, 7.5) * alive
        nx, ny = x + np.cos(ang) * v, y + np.sin(ang) * v
        dmin = raycast(nx, ny, ang).min(axis=1); hit = (dmin < 10) | (nx < 25) | (nx > 975) | (ny < 25) | (ny > 975) | ((nx > 300) & (nx < 700) & (ny > 300) & (ny < 700) & ~((nx > GAPX[0]) & (nx < GAPX[1])))
        for i in np.where(alive)[0]:
            p0, p1 = (x[i], y[i]), (nx[i], ny[i]); k = prog[i] % 8
            if crossed(p0, p1, GATES[k]): prog[i] += 1
            if 3 <= (prog[i] % 8) <= 5 and crossed(p0, p1, SGATE) and p0[1] > p1[1]: prog[i] = (prog[i] // 8 + 1) * 8; used[i] = True; usedat[i] = s
        x, y = np.where(hit, x, nx), np.where(hit, y, ny); v = np.where(hit, 0.5, v); bumps = bumps + hit
        if record:
            for i in range(n):
                if alive[i] or (hit[i] and not np.isnan(x[i])): rec[i].append([round(float(x[i]), 1), round(float(y[i]), 1), round(float(ang[i]), 2)])
        if not alive.any(): break
    fit = prog.astype(float) - bumps * 0.01
    return fit, prog, used, rec, usedat


def shortcut_sim(seed=2, npop=100, gens=40, record=(1, 2, 4, 6, 8, 9, 11, 17, 23, 28, 34, 40)):
    rng = np.random.default_rng(seed); W = rng.normal(0, 1, (npop, nw(6, 8, 2))); out = []
    for g in range(1, gens + 1):
        rc = record is None or g in record; fit, prog, used, rec, usedat = shortcut_episode(W, record=rc)
        out.append(dict(g=g, laps=int(prog.max() // 8), meanprog=float(prog.mean()), used=int(used.sum()), top_used=bool(used[int(np.argmax(fit))]), runs=rec if rc else None, usedat=usedat.tolist(), prog=prog.tolist(), used_i=used.tolist()))
        if used.sum() >= 0.7 * npop: break
        W = breed(W, fit + rng.normal(0, 0.01, npop), rng, ne=10, sig=0.25, pm=0.15); W[-8:] = rng.normal(0, 1, (8, W.shape[1]))
    return dict(gens=out, stats=dict(seed=seed, hist=[(o["g"], o["laps"], round(o["meanprog"], 1), o["used"], o["top_used"]) for o in out]))


# ====================================================================== 3. drone (racing rings)
RINGS0 = [(250, 700), (500, 380), (780, 600), (820, 220), (480, 150), (180, 320)]; RR = 46; DST = 700


def rings_at(t, moving):
    return RINGS0


def drone_episode(W, moving, record=True):
    RRe = 26 if moving else RR; n = len(W); f = bmlp(W, 6, 8, 2); x = np.full(n, 120.0); y = np.full(n, 880.0); vx = np.zeros(n); vy = np.zeros(n); k = np.zeros(n, int); alive = np.ones(n, bool); rec = [[] for _ in range(n)]; passed_at = [[] for _ in range(n)]; bumps = np.zeros(n, int)
    for s in range(DST):
        R = rings_at(s, moving); tx = np.array([R[min(ki, 5)][0] for ki in k]); ty = np.array([R[min(ki, 5)][1] for ki in k])
        o = f(np.column_stack([(tx - x) / 500, (ty - y) / 500, vx / 8, vy / 8, np.ones(n), k / 6]))
        th = (o[:, 0] + 1) / 2 * 0.55; side = o[:, 1] * 0.25
        vy = (vy + 0.22 - th) * 0.985; vx = (vx + side) * 0.985; x = x + vx * alive; y = y + vy * alive
        done = k >= 6; hit = (np.hypot(x - tx, y - ty) < RRe - 6) & ~done
        for i in np.where(hit)[0]: passed_at[i].append(s)
        k = k + hit; oob = (x < 0) | (x > 1000) | (y < 0) | (y > 1000); x = np.clip(x, 0, 1000); y = np.clip(y, 0, 1000); vx = np.where((x <= 0) | (x >= 1000), 0, vx); vy = np.where((y <= 0) | (y >= 1000), 0, vy); bumps = bumps + oob
        if record:
            for i in range(n):
                if alive[i]: rec[i].append([round(float(x[i]), 1), round(float(y[i]), 1), round(float(th[i]), 2), round(float(side[i]), 2)])
        if not alive.any(): break
    R = rings_at(DST, moving); tx = np.array([R[min(ki, 5)][0] for ki in k]); ty = np.array([R[min(ki, 5)][1] for ki in k])
    fit = k * 1000 - np.hypot(x - tx, y - ty) * (k < 6) - bumps * 0.5
    return fit, k, rec, passed_at


def drone_sim(seed=5, npop=100, gens=73, record=(1, 6, 11, 12, 14, 18, 30, 42, 43, 46, 55, 65, 73), sig=0.3, pm=0.15):
    rng = np.random.default_rng(seed); W = rng.normal(0, 1, (npop, nw(6, 8, 2))); out = []; phase = 0; move_at = None
    for g in range(1, gens + 1):
        rc = record is None or g in record; fit, k, rec, pa = drone_episode(W, phase == 1, record=rc); done = int((k >= 6).sum())
        out.append(dict(g=g, phase=phase, done=done, rings=k.tolist(), runs=rec if rc else None, passed=pa))
        if done >= 0.4 * npop and phase == 0: phase = 1; move_at = g + 1
        elif done >= 0.4 * npop and phase == 1: break
        W = breed(W, fit + rng.normal(0, 0.01, npop), rng, ne=10, sig=sig, pm=pm)
    return dict(gens=out, stats=dict(seed=seed, move_at=move_at, final=out[-1]["g"], done=[(o["g"], o["done"]) for o in out]))


# ====================================================================== 4. password
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
TARGET = "SHIP WITH PIP"


def password_sim(seed=4, npop=100, gens=80, target=TARGET):
    rng = np.random.default_rng(seed); L = len(target); T = np.array([ALPHA.index(c) for c in target]); pop = rng.integers(0, 27, (npop, L)); out = []
    for g in range(1, gens + 1):
        fit = (pop == T).sum(axis=1); order = np.argsort(-fit); solved = int((fit == L).sum())
        out.append(dict(g=g, best=int(fit.max()), solved=solved, guesses=["".join(ALPHA[c] for c in row) for row in pop[order]], right=fit[order].tolist()))
        if solved >= 0.5 * npop: break
        elite = pop[order[:10]]; kids = [e.copy() for e in elite]
        for _ in range(npop - 10):
            a, b = elite[rng.integers(10)], elite[rng.integers(10)]; k = np.where(rng.random(L) < 0.5, a, b); m = rng.random(L) < 1.0 / L; k[m] = rng.integers(0, 27, m.sum()); kids.append(k)
        pop = np.vstack(kids)
    return dict(gens=out, stats=dict(seed=seed, target=target, hist=[(o["g"], o["best"], o["solved"]) for o in out]))


# ====================================================================== 5. keepups (paddle keeps a ball up; twist: 2 balls)
KW, KH = 200, 240; KSTEPS = 900; PADW = 70


def keep_episode(W, nballs, seed, record=True):
    n = len(W); f = bmlp(W, 9, 8, 1); rng = np.random.default_rng(seed); px = np.full(n, KW / 2.0)
    bx = np.full((n, nballs), KW / 2.0); by = np.full((n, nballs), 60.0); bvx = np.tile(rng.uniform(-2, 2, nballs), (n, 1)); bvy = np.zeros((n, nballs)); by[:, 1:] = 20.0
    alive = np.ones(n, bool); hits = np.zeros(n, int); died = np.full(n, KSTEPS); rec = [[] for _ in range(n)]
    for s in range(KSTEPS):
        # nearest-to-floor ball as primary input, other as secondary
        order = np.argsort(-by, axis=1); i0 = order[:, 0]; i1 = order[:, 1] if nballs > 1 else i0; ar = np.arange(n)
        X = np.column_stack([(bx[ar, i0] - px) / KW, by[ar, i0] / KH, bvx[ar, i0] / 5, bvy[ar, i0] / 8, (bx[ar, i1] - px) / KW, by[ar, i1] / KH, bvx[ar, i1] / 5, bvy[ar, i1] / 8, np.ones(n)])
        mv = f(X)[:, 0] * 7 * alive; px = np.clip(px + mv, PADW / 2, KW - PADW / 2)
        bvy += 0.25; bx += bvx; by += bvy
        wall = (bx < 6) | (bx > KW - 6); bvx = np.where(wall, -bvx, bvx); bx = np.clip(bx, 6, KW - 6); top = by < 6; bvy = np.where(top, np.abs(bvy), bvy)
        hit = (by >= KH - 18) & (bvy > 0) & (np.abs(bx - px[:, None]) < PADW / 2 + 6); bvy = np.where(hit, -8.5, bvy); bvx = np.where(hit, bvx + (bx - px[:, None]) * 0.12 + mv[:, None] * 0.3, bvx); bvx = np.clip(bvx, -5, 5)
        hits += hit.sum(axis=1) * alive; miss = (by > KH + 10).any(axis=1); newly = miss & alive; died[newly] = s; alive &= ~miss
        if record:
            for i in range(n):
                if alive[i] or died[i] == s: rec[i].append([round(float(px[i]), 1)] + [round(float(v), 1) for b in range(nballs) for v in (bx[i, b], by[i, b])])
        if not alive.any(): break
    return hits, died, rec


def keep_sim(seed=5, npop=100, gens=95):
    rng = np.random.default_rng(seed); W = rng.normal(0, 1, (npop, nw(9, 8, 1))); out = []; phase = 0; two_at = None
    for g in range(1, gens + 1):
        hits, died, rec = keep_episode(W, 1 + phase, seed); ns = int((died >= KSTEPS).sum()); order = np.argsort(-(hits + died * 0.01))
        out.append(dict(g=g, phase=phase, surv=ns, best=int(hits.max()), mean=float(hits.mean()), runs=[rec[i] for i in order[:20]], hits=[int(hits[i]) for i in order[:20]]))
        if ns >= 0.5 * npop and phase == 0: phase = 1; two_at = g + 1
        elif ns >= 0.3 * npop and phase == 1: break
        W = breed(W, hits + died * 0.01 + rng.normal(0, 0.001, npop), rng, ne=10, sig=0.15, pm=0.05)
    return dict(gens=out, stats=dict(seed=seed, two_at=two_at, final=out[-1]["g"], hist=[(o["g"], o["surv"], o["best"], round(o["mean"], 1)) for o in out]))


SIMS = dict(walk=walk_sim, shortcut=shortcut_sim, drone=drone_sim, password=password_sim, keepups=keep_sim)

if __name__ == "__main__":
    import json, sys, time
    which = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[3:]); kw = {k: int(v) for k, v in kw.items()}
    t = time.time(); r = SIMS[which](**kw)
    print(which, "stats", r["stats"], "gens", len(r["gens"]), "%.1fs" % (time.time() - t))
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    if len(sys.argv) > 2 and sys.argv[2] != "-": json.dump(r, open(sys.argv[2], "w"), cls=NE)
