"""Evolving race cars (tiny neural net + sensors) and evolving cart-pole balancers."""
import numpy as np
from scipy import ndimage

S = 960
CX, CY = 480, 480
HW = 40  # track half-width px


def centerline(n=1400):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 290 * (1 + 0.16 * np.sin(2 * th + 0.6) + 0.13 * np.cos(3 * th) + 0.07 * np.sin(5 * th + 1.0))
    return np.stack([CX + r * np.cos(th), CY + r * np.sin(th)], 1)


def build_sdf():
    c = centerline()
    m = np.ones((S, S), bool)
    xi = np.clip(c[:, 0].round().astype(int), 0, S - 1); yi = np.clip(c[:, 1].round().astype(int), 0, S - 1)
    m[yi, xi] = False
    # connect gaps
    m = ndimage.binary_erosion(~m, iterations=0) if False else m
    D = ndimage.distance_transform_edt(m)
    return D.astype(np.float32), c


D, CL = build_sdf()
ANG0 = np.arctan2(CL[0, 1] - CY, CL[0, 0] - CX)
START = CL[0].copy()
_h = CL[3] - CL[0]
HEAD0 = np.arctan2(_h[1], _h[0])
RAYA = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
NIN, NH = 5, 6
NW = NIN * NH + NH + NH * 2 + 2


def unpack(W):
    n = W.shape[0]
    i = 0
    w1 = W[:, i:i + NIN * NH].reshape(n, NIN, NH); i += NIN * NH
    b1 = W[:, i:i + NH]; i += NH
    w2 = W[:, i:i + NH * 2].reshape(n, NH, 2); i += NH * 2
    b2 = W[:, i:i + 2]
    return w1, b1, w2, b2


def sense(x, y, h):
    out = np.zeros((len(x), 5), np.float32)
    for k, a in enumerate(RAYA):
        ang = h + a
        dist = np.zeros(len(x), np.float32)
        hit = np.zeros(len(x), bool)
        for s in range(1, 21):
            d = s * 8.0
            px = np.clip((x + d * np.cos(ang)).astype(int), 0, S - 1)
            py = np.clip((y + d * np.sin(ang)).astype(int), 0, S - 1)
            off = D[py, px] > HW
            new = off & ~hit
            dist[new] = d
            hit |= off
        dist[~hit] = 168.0
        out[:, k] = dist / 168.0
    return out


def run_cars(W, steps=1000, speed=4.0, turn=0.09):
    n = W.shape[0]
    x = np.full(n, START[0], float); y = np.full(n, START[1], float); h = np.full(n, HEAD0)
    alive = np.ones(n, bool)
    ang_prev = np.full(n, ANG0)
    prog = np.zeros(n)
    w1, b1, w2, b2 = unpack(W)
    P = np.zeros((steps, n, 3), np.float32)
    A = np.zeros((steps, n), bool)
    for t in range(steps):
        s = sense(x, y, h)
        hdn = np.tanh(np.einsum("ni,nij->nj", s, w1) + b1)
        o = np.tanh(np.einsum("nj,njk->nk", hdn, w2) + b2)
        h = np.where(alive, h + o[:, 0] * turn, h)
        sp = speed * (0.7 + 0.3 * (o[:, 1] + 1) / 2)
        x = np.where(alive, x + sp * np.cos(h), x); y = np.where(alive, y + sp * np.sin(h), y)
        d = D[np.clip(y.astype(int), 0, S - 1), np.clip(x.astype(int), 0, S - 1)]
        alive &= d <= HW
        ang = np.arctan2(y - CY, x - CX)
        da = (ang - ang_prev + np.pi) % (2 * np.pi) - np.pi
        prog += np.where(alive, da, 0)
        ang_prev = ang
        P[t, :, 0], P[t, :, 1], P[t, :, 2] = x, y, h
        A[t] = alive
    return P, A, prog / (2 * np.pi)


def evolve_cars(seed=1, gens=30, n=40):
    rng = np.random.default_rng(seed)
    pop = rng.normal(0, 0.25, (n, NW))
    hist = []
    for g in range(gens):
        P, A, prog = run_cars(pop)
        hist.append(dict(P=P, A=A, prog=prog.copy()))
        order = np.argsort(-prog)
        elite = pop[order[:6]]
        new = [elite[0].copy(), elite[1].copy()]
        while len(new) < n:
            p = elite[rng.integers(0, 6)]
            new.append(p + rng.normal(0, 0.25, NW) * (rng.random(NW) < 0.3))
        pop = np.array(new)
    return hist


# ---------------------------------------------------------------- cartpole
def cart_run(Wt, steps=500, seed=0):
    n = Wt.shape[0]
    rng = np.random.default_rng(seed)
    s = rng.uniform(-0.05, 0.05, (n, 4))
    alive = np.ones(n, bool)
    traj = np.zeros((steps, n, 2), np.float32)  # x, theta
    A = np.zeros((steps, n), bool)
    life = np.zeros(n, int)
    g, mc, mp, l, fm, tau = 9.8, 1.0, 0.1, 0.5, 5.0, 0.02
    for t in range(steps):
        a = (np.einsum("ni,ni->n", Wt, s) > 0)
        f = np.where(a, fm, -fm)
        x, xd, th, thd = s.T
        ct, st = np.cos(th), np.sin(th)
        tmp = (f + mp * l * thd ** 2 * st) / (mc + mp)
        tha = (g * st - ct * tmp) / (l * (4.0 / 3.0 - mp * ct ** 2 / (mc + mp)))
        xa = tmp - mp * l * tha * ct / (mc + mp)
        x = x + tau * xd; xd = xd + tau * xa; th = th + tau * thd; thd = thd + tau * tha
        s = np.stack([x, xd, th, thd], 1)
        alive &= (np.abs(x) < 2.4) & (np.abs(th) < 12 * np.pi / 180)
        life += alive
        traj[t, :, 0], traj[t, :, 1] = x, th
        A[t] = alive
    return traj, A, life


def evolve_cart(seed=2, gens=14, n=30):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(-1, 1, (n, 4))
    hist = []
    for g in range(gens):
        traj, A, life = cart_run(pop, seed=100 + g)
        hist.append(dict(traj=traj, A=A, life=life.copy()))
        order = np.argsort(-life)
        elite = pop[order[:5]]
        new = [elite[0].copy()]
        while len(new) < n:
            new.append(elite[rng.integers(0, 5)] + rng.normal(0, 0.4, 4))
        pop = np.array(new)
    return hist


if __name__ == "__main__":
    import time
    t0 = time.time()
    H = evolve_cars()
    print("cars", time.time() - t0)
    print([round(float(h["prog"].max()), 2) for h in H])
    print([int(h["A"][-1].sum()) for h in H])
    Hc = evolve_cart()
    print("cart best life", [int(h["life"].max()) for h in Hc], "mean", [int(h["life"].mean()) for h in Hc], "n500", [int((h["life"] >= 500).sum()) for h in Hc])
