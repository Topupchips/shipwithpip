"""Real simulations for the dumb-AI-vs-goal reels: evolving rockets, double-bridge ants."""
import math
import numpy as np

# ------------------------------------------------------------------ rockets
S = np.array([480.0, 900.0])
GOAL = np.array([480.0, 90.0])
GR = 42.0
WALLS = [(0, 560, 650, 600), (310, 300, 960, 340)]   # x0,y0,x1,y1
NG, CH, NR = 64, 5, 100          # genes, steps per gene, rockets
L = NG * CH
ACC, VMAX = 0.55, 7.5


def rocket_run(genomes):
    """genomes: (NR, NG, 2) unit vectors. returns pos (L+1, NR, 2), state (L+1, NR) 0 alive 1 crashed 2 reached, fitness"""
    n = len(genomes)
    p = np.tile(S, (n, 1))
    v = np.zeros((n, 2))
    st = np.zeros(n, int)
    t_reach = np.full(n, L)
    P = [p.copy()]
    ST = [st.copy()]
    for t in range(L):
        alive = st == 0
        g = genomes[:, t // CH, :]
        v[alive] += g[alive] * ACC
        sp = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        v = np.where(sp > VMAX, v / sp * VMAX, v)
        p[alive] += v[alive]
        for i in np.where(alive)[0]:
            x, y = p[i]
            if x < 6 or x > 954 or y < 6 or y > 954:
                st[i] = 1
                continue
            for (x0, y0, x1, y1) in WALLS:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    st[i] = 1
            if st[i] == 0 and np.hypot(x - GOAL[0], y - GOAL[1]) < GR:
                st[i] = 2
                t_reach[i] = t
        P.append(p.copy())
        ST.append(st.copy())
    P, ST = np.array(P), np.array(ST)
    d = np.linalg.norm(P[-1] - GOAL, axis=1)
    # best-ever closeness along path as progress signal
    dm = np.linalg.norm(P - GOAL, axis=2).min(0)
    fit = 1.0 / (dm + 10) ** 2
    fit = np.where(st == 2, 1.0 / 100.0 ** 2 * 4 + (L - t_reach) / L * 0.01, fit)
    fit = np.where(st == 1, fit * 0.5, fit)
    return P, ST, fit


def evolve(seed=3, gens=80, mut=0.05):
    rng = np.random.default_rng(seed)

    def rv(n):
        a = rng.uniform(0, 2 * np.pi, n)
        return np.stack([np.cos(a), np.sin(a)], -1)
    pop = rv(NR * NG).reshape(NR, NG, 2)
    hist = []
    for g in range(gens):
        P, ST, fit = rocket_run(pop)
        hist.append(dict(P=P, ST=ST, reached=int((ST[-1] == 2).sum()), crashed=int((ST[-1] == 1).sum()), best=float(np.linalg.norm(P[-1] - GOAL, axis=1).min())))
        if g == gens - 1:
            break
        order = np.argsort(-fit)
        newp = [pop[i].copy() for i in order[:4]]
        probs = fit / fit.sum()
        while len(newp) < NR:
            a, b = rng.choice(NR, 2, p=probs)
            cut = rng.integers(1, NG)
            child = np.concatenate([pop[a][:cut], pop[b][cut:]])
            m = rng.random(NG) < mut
            child[m] = rv(int(m.sum()))
            newp.append(child)
        pop = np.array(newp)
    return hist


# ------------------------------------------------------------------ ants (double bridge)
def ant_paths():
    nest, food = np.array([120.0, 480.0]), np.array([840.0, 480.0])
    short = [tuple(nest), (300.0, 380.0), (480.0, 340.0), (660.0, 380.0), tuple(food)]
    long_ = [tuple(nest), (220.0, 640.0), (360.0, 800.0), (600.0, 820.0), (760.0, 700.0), tuple(food)]
    return short, long_


def poly_len(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def ants_sim(seed=1, T=34.0, fps=30, speed=330.0, spawn=8.0, k=8.0, n=2.0, evap=0.03):
    rng = np.random.default_rng(seed)
    short, long_ = ant_paths()
    Ls, Ll = poly_len(short), poly_len(long_)
    tau = [0.0, 0.0]
    ants = []   # dict(path, dir(+1 out, -1 back), s)
    frames = []
    dt = 1.0 / fps
    acc = 0.0
    hist_short = []
    trips = [0, 0]
    for f in range(int(T * fps)):
        acc += spawn * dt
        while acc >= 1.0 and len(ants) < 100:
            acc -= 1.0
            ants.append(dict(path=None, dir=1, s=0.0, id=len(ants)))
        for a in ants:
            if a["path"] is None:
                ps = (k + tau[0]) ** n
                pl = (k + tau[1]) ** n
                a["path"] = 0 if rng.random() < ps / (ps + pl) else 1
                a["s"] = 0.0
            Lp = Ls if a["path"] == 0 else Ll
            a["s"] += speed * dt * (1 if a["dir"] == 1 else 1)
            if a["s"] >= Lp:
                tau[a["path"]] += 1.0
                trips[a["path"]] += 1
                a["dir"] = -a["dir"]
                a["path"] = None   # choose again on the way back / out
        tau = [t * (1 - evap * dt) for t in tau]
        snap = []
        for a in ants:
            if a["path"] is None:
                continue
            pts = short if a["path"] == 0 else long_
            Lp = Ls if a["path"] == 0 else Ll
            s = a["s"] if a["dir"] == 1 else Lp - a["s"]
            snap.append((a["path"], a["dir"], s))
        frames.append(dict(tau=tuple(tau), ants=snap, trips=tuple(trips)))
        usingS = sum(1 for x in snap if x[0] == 0)
        hist_short.append(usingS / max(1, len(snap)))
    return frames, hist_short, (Ls, Ll)


def point_on(pts, s):
    for a, b in zip(pts, pts[1:]):
        d = math.dist(a, b)
        if s <= d:
            u = s / d
            return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
        s -= d
    return pts[-1]
