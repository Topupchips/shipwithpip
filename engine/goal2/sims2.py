"""Five more real simulations: maze, flappy, moon, dodge, pole. Same output shape as sims.py.
World coordinates: 1000x1000 play area."""
import numpy as np
from collections import deque


def bmlp(W, nin, nh, nout):
    """Batched MLP: W (n, NW) -> f(X (n, nin)) -> (n, nout), tanh."""
    i = 0; W1 = W[:, i:i + nin * nh].reshape(-1, nin, nh); i += nin * nh; b1 = W[:, i:i + nh]; i += nh
    W2 = W[:, i:i + nh * nout].reshape(-1, nh, nout); i += nh * nout; b2 = W[:, i:i + nout]
    return lambda X: np.tanh(np.einsum('nh,nho->no', np.tanh(np.einsum('ni,nih->nh', X, W1) + b1), W2) + b2)


def nw(nin, nh, nout): return nin * nh + nh + nh * nout + nout


def breed(pop, fit, rng, ne=10, sig=0.2, pm=0.2, cx=0.5):
    order = np.argsort(-fit); elite = pop[order[:ne]]; kids = [e.copy() for e in elite]; n, L = pop.shape
    for _ in range(n - ne):
        a, b = elite[rng.integers(ne)], elite[rng.integers(ne)]
        k = (np.where(rng.random(L) < 0.5, a, b) if rng.random() < cx else a).copy()
        k += rng.normal(0, sig, L) * (rng.random(L) < pm); kids.append(k)
    return np.vstack(kids)


# ====================================================================== 1. maze
MZ = 14; CELL = 1000 / 14


def make_maze(rng, extra=20):
    """walls[y][x] = set of blocked dirs (0 R,1 D,2 L,3 U). DFS perfect maze + extra openings for loops."""
    W = [[{0, 1, 2, 3} for _ in range(MZ)] for _ in range(MZ)]
    D = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    seen = np.zeros((MZ, MZ), bool); st = [(0, 0)]; seen[0, 0] = True
    while st:
        x, y = st[-1]; opts = [(d, x + dx, y + dy) for d, (dx, dy) in enumerate(D) if 0 <= x + dx < MZ and 0 <= y + dy < MZ and not seen[y + dy, x + dx]]
        if not opts: st.pop(); continue
        d, nx, ny = opts[rng.integers(len(opts))]; W[y][x].discard(d); W[ny][nx].discard((d + 2) % 4); seen[ny, nx] = True; st.append((nx, ny))
    k = 0
    while k < extra:
        x, y, d = rng.integers(MZ), rng.integers(MZ), rng.integers(4); dx, dy = D[d]; nx, ny = x + dx, y + dy
        if 0 <= nx < MZ and 0 <= ny < MZ and d in W[y][x]: W[y][x].discard(d); W[ny][nx].discard((d + 2) % 4); k += 1
    return W


def maze_bfs(W, goal):
    D = [(1, 0), (0, 1), (-1, 0), (0, -1)]; dist = np.full((MZ, MZ), 9999); dist[goal[1], goal[0]] = 0; q = deque([goal])
    while q:
        x, y = q.popleft()
        for d, (dx, dy) in enumerate(D):
            if d in W[y][x]: continue
            nx, ny = x + dx, y + dy
            if dist[ny, nx] > dist[y, x] + 1: dist[ny, nx] = dist[y, x] + 1; q.append((nx, ny))
    return dist


def maze_run(genes, W, dist, goal, start=(0, 0)):
    D = [(1, 0), (0, 1), (-1, 0), (0, -1)]; x, y = start; path = [(x, y)]; reached = False; bumps = 0
    for g in genes:
        d = int(g)
        if d not in W[y][x]: x += D[d][0]; y += D[d][1]
        else: bumps += 1
        path.append((x, y))
        if (x, y) == goal: reached = True; break
    fit = -dist[y, x] * 10 - bumps + (1000 - len(path) if reached else 0)
    return path, fit, reached


def maze_sim(seed=6, npop=100, gens=200, L=100):
    rng = np.random.default_rng(seed); W = make_maze(rng); goal = (MZ - 1, MZ - 1); dist = maze_bfs(W, goal)
    pop = rng.integers(0, 4, (npop, L)); out = []; phase = 0; blocked_at = None; wall = None; gp = 0
    walls_dump = lambda: [[sorted(W[y][x]) for x in range(MZ)] for y in range(MZ)]
    for g in range(1, gens + 1):
        R = [maze_run(p, W, dist, goal) for p in pop]; fit = np.array([r[1] for r in R]); nr = sum(1 for r in R if r[2])
        out.append(dict(g=g, phase=phase, reached=nr, paths=[r[0] for r in R], ok=[r[2] for r in R], walls=walls_dump() if (g == 1 or g == blocked_at) else None, block=wall))
        if nr >= 0.8 * npop and phase == 0:
            # block the elite's path at ~55% of its length, keep exit reachable
            best = R[int(np.argmax(fit))][0]; Dd = [(1, 0), (0, 1), (-1, 0), (0, -1)]; cands = []
            for i in range(int(len(best) * 0.3), int(len(best) * 0.8)):
                (x, y), (nx, ny) = best[i], best[i + 1]
                if (x, y) == (nx, ny): continue
                d = Dd.index((nx - x, ny - y)); W[y][x].add(d); W[ny][nx].add((d + 2) % 4); nd = maze_bfs(W, goal)
                if nd[0, 0] < 9999 and nd[0, 0] <= 58: cands.append((nd[0, 0], i, x, y, nx, ny, d))
                W[y][x].discard(d); W[ny][nx].discard((d + 2) % 4)
            if cands:
                _, i, x, y, nx, ny, d = max(cands); W[y][x].add(d); W[ny][nx].add((d + 2) % 4); dist = maze_bfs(W, goal); wall = [x, y, nx, ny]
            phase = 1; blocked_at = g + 1; gp = g
        elif nr >= 0.8 * npop and phase == 1: break
        order = np.argsort(-fit); elite = pop[order[:10]]; kids = [e.copy() for e in elite]
        for _ in range(npop - 10):
            a, b = elite[rng.integers(10)], elite[rng.integers(10)]; c = rng.integers(1, L); k = np.concatenate([a[:c], b[c:]]) if rng.random() < 0.5 else a.copy()
            pm = max(0.006, 0.03 * 0.9 ** (g - gp)); m = rng.random(L) < pm; k[m] = rng.integers(0, 4, m.sum())
            if rng.random() < 0.1: j = rng.integers(L); k[j:] = rng.integers(0, 4, L - j)  # tail re-roll: keeps the prefix, explores the rest
            kids.append(k)
        pop = np.vstack(kids)
    return dict(gens=out, stats=dict(seed=seed, blocked_at=blocked_at, final=out[-1]["g"], reached=[(o["g"], o["reached"]) for o in out]))


# ====================================================================== 2. flappy
BX = 220; FG = 0.45; FLAP = -8.5; PIPE_W = 90; GAP = 250; PIPE_DX = 320; FSPEED = 5


def pipes_for(seed, n=40):
    r = np.random.default_rng(seed + 77); return [(600 + i * PIPE_DX, float(r.uniform(260, 740))) for i in range(n)]


def flappy_sim(seed=5, npop=100, gens=60, maxsteps=1500):
    rng = np.random.default_rng(seed); NWf = nw(4, 6, 1); pop = rng.normal(0, 1, (npop, NWf)); pipes = pipes_for(seed); out = []
    px = np.array([p[0] for p in pipes]); pc = np.array([p[1] for p in pipes])
    for g in range(1, gens + 1):
        f = bmlp(pop, 4, 6, 1); y = np.full(npop, 500.0); vy = np.zeros(npop); alive = np.ones(npop, bool); ys = [[] for _ in range(npop)]; died = np.full(npop, maxsteps)
        for s in range(maxsteps):
            dist = s * FSPEED; sx = px - dist  # pipe screen x relative to world origin; bird at BX
            k = int(np.argmax(sx + PIPE_W > BX - 12)); gx = sx[k]; gc = pc[k]
            X = np.column_stack([(gc - y) / 400, np.full(npop, (gx - BX) / 300), vy / 10, np.ones(npop)])
            o = f(X)[:, 0]; flap = (o > 0) & alive; vy = np.where(flap, FLAP, vy + FG); vy = np.minimum(vy, 12); y = y + vy * alive
            hit = (y < 0) | (y > 1000) | ((gx < BX + 14) & (gx + PIPE_W > BX - 14) & (np.abs(y - gc) > GAP / 2 - 12))
            newly = hit & alive; died[newly] = s; alive &= ~hit
            for i in range(npop):
                if alive[i] or died[i] == s: ys[i].append(round(float(y[i]), 1))
            if not alive.any(): break
        fit = died.astype(float) + rng.normal(0, 0.01, npop); nsurv = int((died >= maxsteps).sum()); best = int(died.max())
        out.append(dict(g=g, surv=nsurv, best=best, ys=ys, pipes_passed=int(((best * FSPEED) - 600 + PIPE_W) // PIPE_DX + 1) if best * FSPEED > 600 else 0))
        if nsurv >= 0.5 * npop: break
        pop = breed(pop, fit, rng, ne=10, sig=0.12, pm=0.06)
    return dict(gens=out, stats=dict(seed=seed, pipes=pipes, surv=[(o["g"], o["surv"], o["best"]) for o in out], maxsteps=maxsteps))


# ====================================================================== 3. moon (gravity slingshot)
EARTH = (110, 520); PLANET = (520, 500); PR = 80; MOON = (880, 300); MR = 34


def moon_shot(angle, speed, mu, steps=420):
    a = np.radians(angle); x, y = EARTH; vx, vy = np.cos(a) * speed, -np.sin(a) * speed; pts = []; best = 1e9; hit = False; crash = False
    for s in range(steps):
        dx, dy = PLANET[0] - x, PLANET[1] - y; d2 = dx * dx + dy * dy; d = np.sqrt(d2); ax, ay = mu * dx / (d2 * d), mu * dy / (d2 * d)
        vx += ax; vy += ay; x += vx; y += vy; pts.append((round(x, 1), round(y, 1)))
        dm = np.hypot(x - MOON[0], y - MOON[1]); best = min(best, dm)
        if dm < MR: hit = True; break
        if d < PR: crash = True; break
        if x < -50 or x > 1050 or y < -50 or y > 1050: break
    return pts, best, hit, crash


def moon_sim(seed=3, npop=100, gens=80):
    rng = np.random.default_rng(seed); pop = np.column_stack([rng.uniform(-60, 60, npop), rng.uniform(4, 12, npop)]); mu = 9000.0; out = []; phase = 0; heavy_at = None; gp = 0
    for g in range(1, gens + 1):
        R = [moon_shot(a, s, mu) for a, s in pop]; fit = np.array([-b + (500 if h else 0) - (200 if c else 0) for _, b, h, c in R]); nh = sum(1 for r in R if r[2])
        out.append(dict(g=g, phase=phase, hit=nh, mu=mu, shots=[r[0] for r in R], ok=[r[2] for r in R], crash=[r[3] for r in R]))
        if nh >= 0.9 * npop and phase == 0: phase = 1; mu *= 2; heavy_at = g + 1; gp = g
        elif nh >= 0.9 * npop and phase == 1: break
        order = np.argsort(-fit); elite = pop[order[:10]]; kids = [e.copy() for e in elite]
        for _ in range(npop - 10):
            a, b = elite[rng.integers(10)], elite[rng.integers(10)]; k = np.where(rng.random(2) < 0.5, a, b).copy()
            sg = max(0.2, 0.85 ** (g - gp)); k += rng.normal(0, [3.0 * sg, 0.5 * sg]) * (rng.random(2) < 0.7); k[0] = np.clip(k[0], -85, 85); k[1] = np.clip(k[1], 3, 14); kids.append(k)
        pop = np.vstack(kids)
    return dict(gens=out, stats=dict(seed=seed, heavy_at=heavy_at, final=out[-1]["g"], hits=[(o["g"], o["hit"]) for o in out]))


# ====================================================================== 4. dodge (asteroids)
DSTEPS = 480; DY = 900


def rocks_for(seed, rate, steps=DSTEPS + 200):
    r = np.random.default_rng(seed + 11); rocks = []; t = 0
    while t < steps:
        rocks.append((t, float(r.uniform(30, 970)), float(r.uniform(4.5, 7.5)), float(r.uniform(16, 34)))); t += int(max(1, r.exponential(rate)))
    return rocks


def dodge_episode(pop, rocks, rng, record=True):
    npop = len(pop); f = bmlp(pop, 8, 8, 1); x = np.full(npop, 500.0); alive = np.ones(npop, bool); died = np.full(npop, DSTEPS); xs = [[] for _ in range(npop)]
    R = np.array(rocks)  # t, x, vy, r
    for s in range(DSTEPS):
        act = R[(R[:, 0] <= s) & (R[:, 0] + (DY + 60) / R[:, 2] >= s)]
        if len(act):
            ry = (s - act[:, 0]) * act[:, 2]  # y position (falls from 0)
            dx = act[None, :, 1] - x[:, None]; dy = np.broadcast_to(DY - ry[None, :], dx.shape); above = dy > -40
            ttl = np.where(above, dy / act[None, :, 2] + np.abs(dx) * 0.02, 1e9)  # time-to-impact-ish, prefer near ones
            idx = np.argsort(ttl, axis=1)[:, :3]
            feats = []
            for j in range(3):
                if j < idx.shape[1]: jj = idx[:, j]; ok = ttl[np.arange(npop), jj] < 1e8; feats += [np.where(ok, dx[np.arange(npop), jj] / 300, 0), np.where(ok, dy[np.arange(npop), jj] / 600, 1.5)]
                else: feats += [np.zeros(npop), np.full(npop, 1.5)]
            hit = ((np.abs(dx) < act[None, :, 3] + 14) & (np.abs(DY - ry[None, :]) < act[None, :, 3] + 14)).any(axis=1)
        else: feats = [np.zeros(npop), np.full(npop, 1.5)] * 3; hit = np.zeros(npop, bool)
        X = np.column_stack(feats + [x / 500 - 1, np.ones(npop)]); o = f(X)[:, 0]
        x = np.clip(x + o * 6 * alive, 14, 986)
        newly = hit & alive; died[newly] = s; alive &= ~hit
        if record:
            for i in range(npop):
                if alive[i] or died[i] == s: xs[i].append(round(float(x[i]), 1))
        if not alive.any(): break
    return died, xs


def dodge_sim(seed=2, npop=100, gens=60):
    rng = np.random.default_rng(seed); pop = rng.normal(0, 1, (npop, nw(8, 8, 1))); rate = 8; rocks = rocks_for(seed, rate); out = []; phase = 0; storm_at = None
    for g in range(1, gens + 1):
        died, xs = dodge_episode(pop, rocks, rng); ns = int((died >= DSTEPS).sum())
        out.append(dict(g=g, phase=phase, surv=ns, xs=xs, rocks=rocks))
        if ns >= 0.65 * npop and phase == 0: phase = 1; rocks = sorted(rocks + rocks_for(seed + 4, 12)); storm_at = g + 1
        elif ns >= 0.65 * npop and phase == 1: break
        pop = breed(pop, died.astype(float) + rng.normal(0, 0.01, npop), rng, ne=10, sig=0.15, pm=0.05)
    return dict(gens=out, stats=dict(seed=seed, storm_at=storm_at, final=out[-1]["g"], surv=[(o["g"], o["surv"]) for o in out]))


# ====================================================================== 5. pole (cartpole with a push)
PSTEPS = 450; PUSH_AT = 220; PUSH = 1.9


def pole_episode(pop, push):
    n = len(pop); f = bmlp(pop, 4, 6, 1); x = np.zeros(n); v = np.zeros(n); th = np.random.default_rng(1).uniform(-0.05, 0.05, n); w = np.zeros(n)
    alive = np.ones(n, bool); died = np.full(n, PSTEPS); rec = [[] for _ in range(n)]
    g, mc, mp, l, F, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    for s in range(PSTEPS):
        if push and s == PUSH_AT: w += PUSH
        o = f(np.column_stack([x / 2.4, v / 3, th / 0.21, w / 3]))[:, 0]; force = F * o
        ct, st = np.cos(th), np.sin(th); temp = (force + mp * l * w * w * st) / (mc + mp)
        acc = (g * st - ct * temp) / (l * (4 / 3 - mp * ct * ct / (mc + mp))); xacc = temp - mp * l * acc * ct / (mc + mp)
        x = x + tau * v * alive; v = v + tau * xacc * alive; th = th + tau * w * alive; w = w + tau * acc * alive
        fail = (np.abs(th) > 0.21) | (np.abs(x) > 2.4); newly = fail & alive; died[newly] = s; alive &= ~fail
        for i in range(n):
            if alive[i] or died[i] == s: rec[i].append((round(float(x[i]), 3), round(float(th[i]), 3)))
        if not alive.any(): break
    return died, rec


def pole_sim(seed=1, npop=50, gens=60):
    rng = np.random.default_rng(seed); pop = rng.normal(0, 1, (npop, nw(4, 6, 1))); out = []; phase = 0; push_at = None
    for g in range(1, gens + 1):
        died, rec = pole_episode(pop, phase == 1); ns = int((died >= PSTEPS).sum())
        out.append(dict(g=g, phase=phase, surv=ns, runs=rec))
        if ns >= 0.8 * npop and phase == 0: phase = 1; push_at = g + 1
        elif ns >= 0.7 * npop and phase == 1: break
        pop = breed(pop, died.astype(float) + rng.normal(0, 0.01, npop), rng, ne=10, sig=0.2, pm=0.15)
    return dict(gens=out, stats=dict(seed=seed, push_at=push_at, final=out[-1]["g"], surv=[(o["g"], o["surv"]) for o in out]))


SIMS = dict(maze=maze_sim, flappy=flappy_sim, moon=moon_sim, dodge=dodge_sim, pole=pole_sim)

if __name__ == "__main__":
    import json, sys, time
    which = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[3:]); kw = {k: int(v) for k, v in kw.items()}
    t = time.time(); r = SIMS[which](**kw)
    print(which, "stats", {k: v for k, v in r["stats"].items() if k != "pipes"}, "gens", len(r["gens"]), "%.1fs" % (time.time() - t))
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    if len(sys.argv) > 2 and sys.argv[2] != "-": json.dump(r, open(sys.argv[2], "w"), cls=NE)
