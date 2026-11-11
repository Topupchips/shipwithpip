"""Real simulations for the 'many AIs, one goal' reels. Each returns dict(gens=[...], stats={...}).
Every gen entry: dict(g=gen number (1-based), frames=[...state dicts...], **per-gen stats).
World coordinates are a 1000x1000 play area (pong: 1000x700).
"""
import numpy as np

FPS = 30


def _tanh(x): return np.tanh(x)
def _sig(x): return 1 / (1 + np.exp(-x))


# ====================================================================== 1. hoops
HOOP = (820, 420); RIM = 40; SHOOTER = (120, 820); G = 0.5


def hoops_shot(angle, power, hoop, steps=130):
    a = np.radians(angle); x, y = SHOOTER; vx, vy = np.cos(a) * power, -np.sin(a) * power
    pts = []; best = 1e9; swish = False; prev_y = y
    for s in range(steps):
        x += vx; y += vy; vy += G
        # backboard
        if x > HOOP[0] + 50 and y < HOOP[1] + 120 and y > HOOP[1] - 220: vx = -abs(vx) * 0.6; x = HOOP[0] + 50
        d = np.hypot(x - hoop[0], y - hoop[1]); best = min(best, d)
        if prev_y < hoop[1] <= y and vy > 0 and abs(x - hoop[0]) < RIM - 12: swish = True
        pts.append((round(x, 1), round(y, 1)))
        if swish or y > 1000 or x < 0: break
        prev_y = y
    return pts, best, swish


def hoops_sim(seed=2, npop=100, gens=60):
    rng = np.random.default_rng(seed)
    pop = np.column_stack([rng.uniform(15, 85, npop), rng.uniform(8, 32, npop)])
    hoop = HOOP; out = []; moved_at = None; phase = 0; gp = 0
    for g in range(gens):
        res = [hoops_shot(a, p, hoop) for a, p in pop]
        fit = np.array([(-b + (400 if sw else 0)) for _, b, sw in res]); nsw = sum(1 for r in res if r[2])
        out.append(dict(g=g + 1, phase=phase, hoop=hoop, swish=nsw, shots=[r[0] for r in res], sw=[r[2] for r in res]))
        if nsw >= 0.9 * npop and phase == 0: phase = 1; moved_at = g + 1; hoop = (480, 680); gp = g + 1
        elif nsw >= 0.9 * npop and phase == 1: break
        order = np.argsort(-fit); elite = pop[order[:10]]
        kids = []
        for i in range(npop - 10):
            a, b = elite[rng.integers(10)], elite[rng.integers(10)]; mask = rng.random(2) < 0.5; k = np.where(mask, a, b).copy()
            sg = max(0.25, 0.8 ** (g - gp)); k += rng.normal(0, [2.5 * sg, 1.2 * sg]) * (rng.random(2) < 0.7); k[0] = np.clip(k[0], 10, 88); k[1] = np.clip(k[1], 5, 33); kids.append(k)
        pop = np.vstack([elite, kids])
    return dict(gens=out, stats=dict(moved_at=moved_at, final=out[-1]["g"], seed=seed))


# ====================================================================== 2. lander
PADX = (400, 600); PADY = 940; GRAV = 0.12


def mlp(w, nin, nh, nout):
    i = 0; W1 = w[i:i + nin * nh].reshape(nin, nh); i += nin * nh; b1 = w[i:i + nh]; i += nh; W2 = w[i:i + nh * nout].reshape(nh, nout); i += nh * nout; b2 = w[i:i + nout]
    return lambda x: _tanh(_tanh(x @ W1 + b1) @ W2 + b2)


NIN, NH, NOUT = 5, 6, 2; NW = NIN * NH + NH + NH * NOUT + NOUT


def lander_run(w, start, steps=600):
    f = mlp(w, NIN, NH, NOUT); x, y, vx, vy = start; fr = []; fuel = 0
    for s in range(steps):
        o = f(np.array([(x - 500) / 300, (PADY - y) / 800, vx / 5, vy / 5, 1.0]))
        th = (o[0] + 1) / 2 * 0.32; side = o[1] * 0.07; fuel += th
        vy += GRAV - th; vx += side; x += vx; y += vy
        fr.append((round(x, 1), round(y, 1), round(th, 3), round(side, 3)))
        if y >= PADY - 30: break
        if x < 0 or x > 1000 or y < 0: break
    landed = y >= PADY - 30 and PADX[0] < x < PADX[1] and abs(vy) < 2.0 and abs(vx) < 1.5
    crash = y >= PADY - 30 and not landed
    onpad = PADX[0] < x < PADX[1]
    fit = -abs(x - 500) * 0.5 - abs(vy) * 40 - abs(vx) * 40 - (0 if y >= PADY - 30 else (PADY - y) * 0.5) + (600 - len(fr) * 0.6 if landed else 0) - fuel * 0.2
    return fr, fit, landed, crash, onpad


def lander_sim(seed=3, npop=60, gens=120):
    rng = np.random.default_rng(seed); pop = rng.normal(0, 1, (npop, NW)); out = []
    starts = [(500, 80, 1.5, 0), (330, 80, 0, 0), (680, 80, -1.0, 0)]
    for g in range(gens):
        R = [[lander_run(w, st) for st in starts] for w in pop]
        fit = np.array([np.mean([r[1] for r in rr]) for rr in R]); nl = sum(1 for rr in R if rr[0][2])
        out.append(dict(g=g + 1, landed=nl, runs=[rr[0][0] for rr in R], ok=[rr[0][2] for rr in R], crash=[rr[0][3] for rr in R]))
        if nl >= 0.75 * npop: break
        order = np.argsort(-fit); elite = pop[order[:6]]; kids = [e.copy() for e in elite]
        for i in range(npop - 6):
            a, b = elite[rng.integers(6)], elite[rng.integers(6)]; k = (np.where(rng.random(NW) < 0.5, a, b) if rng.random() < 0.3 else a).copy(); k += rng.normal(0, 0.15, NW) * (rng.random(NW) < 0.12); kids.append(k)
        pop = np.vstack(kids)
    return dict(gens=out, stats=dict(final=out[-1]["g"], seed=seed, final_landed=out[-1]["landed"]))


# ====================================================================== 3. snake (tabular Q-learning, 11-feature state)
GRID = 12
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # R, D, L, U


def snake_state(body, food, d):
    hx, hy = body[0]; dx, dy = DIRS[d]
    def danger(ddx, ddy):
        nx, ny = hx + ddx, hy + ddy
        return nx < 0 or ny < 0 or nx >= GRID or ny >= GRID or (nx, ny) in body[:-1]
    r = DIRS[(d + 1) % 4]; l = DIRS[(d - 1) % 4]
    f = [danger(dx, dy), danger(*r), danger(*l), d == 0, d == 1, d == 2, d == 3, food[0] < hx, food[0] > hx, food[1] < hy, food[1] > hy]
    return sum(int(b) << i for i, b in enumerate(f))


def snake_episode(Q, rng, eps, alpha=0.1, gamma=0.9, learn=True, record=False, maxsteps=1200):
    body = [(6, 6), (5, 6), (4, 6)]; d = 0
    def place():
        while True:
            p = (int(rng.integers(GRID)), int(rng.integers(GRID)))
            if p not in body: return p
    food = place(); score = 0; frames = []; since = 0
    for step in range(maxsteps):
        s = snake_state(body, food, d)
        a = int(rng.integers(3)) if rng.random() < eps else int(np.argmax(Q[s]))
        nd = d if a == 0 else (d + 1) % 4 if a == 1 else (d - 1) % 4
        hx, hy = body[0]; nx, ny = hx + DIRS[nd][0], hy + DIRS[nd][1]
        dead = nx < 0 or ny < 0 or nx >= GRID or ny >= GRID or (nx, ny) in body[:-1]
        if record: frames.append((list(body), food, dead))
        if dead:
            if learn: Q[s, a] += alpha * (-10 - Q[s, a])
            break
        body.insert(0, (nx, ny)); ate = (nx, ny) == food
        if ate: score += 1; food = place(); since = 0
        else: body.pop(); since += 1
        rwd = 10 if ate else -0.01
        if since > 120: rwd = -10; dead = True
        s2 = snake_state(body, food, nd)
        if learn: Q[s, a] += alpha * (rwd + (0 if dead else gamma * np.max(Q[s2])) - Q[s, a])
        d = nd
        if dead: break
    if record: frames.append((list(body), food, True))
    return score, frames


def snake_sim(seed=3, nagents=16, episodes=1500, show=(1, 30, 150, 500, 1500)):
    out = {e: [] for e in show}; scores_hist = {e: [] for e in show}
    for ai in range(nagents):
        rng = np.random.default_rng(seed * 100 + ai); Q = np.zeros((2048, 3))
        for ep in range(1, episodes + 1):
            eps = max(0.01, 0.5 * (0.995 ** ep))
            if ep in show:
                rr = np.random.default_rng(777 + ai + ep)
                sc, fr = snake_episode(Q, rr, 0.0, learn=False, record=True)
                out[ep].append(fr); scores_hist[ep].append(sc)
            snake_episode(Q, rng, eps)
    gens = [dict(g=e, scores=scores_hist[e], games=out[e]) for e in show]
    return dict(gens=gens, stats=dict(seed=seed, avg={e: float(np.mean(scores_hist[e])) for e in show}, best={e: int(np.max(scores_hist[e])) for e in show}))


# ====================================================================== 4. dino (GA plays a Chrome-dino style runner)
GY = 760; JUMPV = -16.5; GRAVD = 0.9


def cacti_for(seed, n=80):
    rng = np.random.default_rng(seed); xs = []; x = 900
    for i in range(n):
        x += rng.integers(420, 760); xs.append((int(x), int(rng.choice([30, 44, 60])), int(rng.choice([55, 75, 95]))))
    return xs


def dino_play(w, cacti, maxsteps=3000):
    y, vy = GY, 0.0; dist = 0.0; speed = 7.0; ys = []
    for s in range(maxsteps):
        nxt = next(((cx - dist, cw, ch) for cx, cw, ch in cacti if cx + cw > dist + 60), None)
        if nxt is None: break
        d, cw, ch = nxt
        if y >= GY and (w[0] * d / 300 + w[1] * speed / 10 + w[2] * ch / 95 + w[3]) > 0: vy = JUMPV
        vy += GRAVD; y = min(GY, y + vy)
        if y >= GY: vy = 0
        dist += speed; speed = min(11.5, 7 + s * 0.002)
        ys.append(round(y, 1))
        # collision: dino box x in [60, 60+44], cactus at [d, d+cw], height ch above ground
        if d < 104 and d + cw > 60 and y > GY - ch: break
    return dist, ys


def dino_sim(seed=7, npop=100, gens=40):
    rng = np.random.default_rng(seed); cacti = cacti_for(seed); pop = rng.normal(0, 1, (npop, 4)); out = []
    for g in range(gens):
        R = [dino_play(w, cacti) for w in pop]; sc = np.array([r[0] for r in R])
        out.append(dict(g=g + 1, best=float(sc.max()), ys=[r[1] for r in R], sc=sc.tolist()))
        done = sum(1 for r in R if len(r[1]) >= 3000); out[-1]['done'] = done
        if done >= 0.6 * npop: break
        order = np.argsort(-sc); elite = pop[order[:10]]; kids = [e.copy() for e in elite[:10]]
        for i in range(npop - 10):
            a, b = elite[rng.integers(10)], elite[rng.integers(10)]; k = np.where(rng.random(4) < 0.5, a, b).copy(); k += rng.normal(0, 0.08, 4) * (rng.random(4) < 0.4); kids.append(k)
        pop = np.vstack(kids)
    return dict(gens=out, stats=dict(seed=seed, final=out[-1]["g"], cacti=cacti, track_len=cacti[-1][0], final_done=out[-1]['done']))


# ====================================================================== 5. hunt (predators vs prey co-evolution)
NPREY, NPRED, STEPS = 40, 8, 320


def hunt_episode(Wprey, Wpred, seed, record=False):
    rng = np.random.default_rng(seed)
    fprey = [mlp(w, 6, 6, 2) for w in Wprey]; fpred = [mlp(w, 6, 6, 2) for w in Wpred]
    P = rng.uniform(100, 900, (NPREY, 2)); Pa = rng.uniform(0, 6.28, NPREY); alive = np.ones(NPREY, bool)
    D = rng.uniform(100, 900, (NPRED, 2)); Da = rng.uniform(0, 6.28, NPRED)
    catches = np.zeros(NPRED, int); surv = np.zeros(NPREY); frames = []; caught_at = []
    for s in range(STEPS):
        # prey
        for i in range(NPREY):
            if not alive[i]: continue
            dv = D - P[i]; dist = np.hypot(dv[:, 0], dv[:, 1]); k = int(np.argmin(dist)); v = dv[k] / max(dist[k], 1)
            # rotate into prey frame
            c, sn = np.cos(Pa[i]), np.sin(Pa[i]); lx, ly = v[0] * c + v[1] * sn, -v[0] * sn + v[1] * c
            wall = np.array([P[i, 0] / 1000 * 2 - 1, P[i, 1] / 1000 * 2 - 1])
            o = fprey[i](np.array([lx, ly, min(1, dist[k] / 300), wall[0], wall[1], 1.0]))
            Pa[i] += o[0] * 0.35; sp = 3.2 + 1.3 * o[1]
            P[i] += sp * np.array([np.cos(Pa[i]), np.sin(Pa[i])]); P[i] = np.clip(P[i], 10, 990)
            surv[i] += 1
        # predators
        for j in range(NPRED):
            idx = np.where(alive)[0]
            if len(idx) == 0: break
            dv = P[idx] - D[j]; dist = np.hypot(dv[:, 0], dv[:, 1]); k = int(np.argmin(dist)); v = dv[k] / max(dist[k], 1)
            c, sn = np.cos(Da[j]), np.sin(Da[j]); lx, ly = v[0] * c + v[1] * sn, -v[0] * sn + v[1] * c
            wall = np.array([D[j, 0] / 1000 * 2 - 1, D[j, 1] / 1000 * 2 - 1])
            o = fpred[j](np.array([lx, ly, min(1, dist[k] / 300), wall[0], wall[1], 1.0]))
            Da[j] += o[0] * 0.3; sp = 3.6 + 1.2 * o[1]
            D[j] += sp * np.array([np.cos(Da[j]), np.sin(Da[j])]); D[j] = np.clip(D[j], 10, 990)
            if dist[k] < 14: alive[idx[k]] = False; catches[j] += 1; caught_at.append((s, idx[k]))
        if record: frames.append(dict(p=[(round(x, 1), round(y, 1), round(a, 2)) for (x, y), a in zip(P, Pa)], al=alive.tolist(), d=[(round(x, 1), round(y, 1), round(a, 2)) for (x, y), a in zip(D, Da)], c=int((~alive).sum())))
    return surv, catches, frames


def hunt_sim(seed=6, gens=36, show=(1, 4, 16, 19, 23, 30, 36), prey_from=19):
    rng = np.random.default_rng(seed); NWh = 6 * 6 + 6 + 6 * 2 + 2
    Wprey = rng.normal(0, 1, (NPREY, NWh)); Wpred = rng.normal(0, 1, (NPRED, NWh)); out = []; hist = []
    for g in range(1, gens + 1):
        surv, catches, frames = hunt_episode(Wprey, Wpred, seed * 1000 + g, record=(g in show))
        hist.append(int(catches.sum()))
        if g in show: out.append(dict(g=g, caught=int(catches.sum()), prey_learning=g >= prey_from, frames=frames))
        def breed(P, f, ne):
            order = np.argsort(-f); elite = P[order[:ne]]; kids = [elite[0].copy()]
            for i in range(len(P) - 1):
                a, b = elite[rng.integers(ne)], elite[rng.integers(ne)]; k = np.where(rng.random(NWh) < 0.5, a, b).copy(); k += rng.normal(0, 0.3, NWh) * (rng.random(NWh) < 0.3); kids.append(k)
            return np.vstack(kids)
        if g >= prey_from: Wprey = breed(Wprey, surv + rng.normal(0, 0.01, NPREY), 8)
        Wpred = breed(Wpred, catches + rng.normal(0, 0.01, NPRED), 3)
    return dict(gens=out, stats=dict(seed=seed, caught=hist, prey_from=prey_from))


SIMS = dict(hoops=hoops_sim, lander=lander_sim, snake=snake_sim, dino=dino_sim, hunt=hunt_sim)

if __name__ == "__main__":
    import json, sys, time
    which = sys.argv[1]; t = time.time(); r = SIMS[which]()
    print(which, "stats", r["stats"], "gens", [g["g"] for g in r["gens"]], "%.1fs" % (time.time() - t))
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    if len(sys.argv) > 2: json.dump(r, open(sys.argv[2], "w"), cls=NE)
