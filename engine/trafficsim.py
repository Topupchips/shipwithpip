"""Ring-road car following (Intelligent Driver Model, Treiber et al. 2000)."""
import numpy as np

def veq(N, L, v0, T, s0, ln):
    from scipy.optimize import brentq
    g = L / N - ln
    return brentq(lambda v: 1 - (v / v0) ** 4 - ((s0 + v * T) / g) ** 2, 0, v0)


def idm_ring(N=22, L=260.0, v0=30/3.6, T=1.0, s0=2.0, a=1.0, b=2.0, ln=4.5, dt=0.05, tmax=150.0,
             brake_t=8.0, brake_d=2.0, seed=1, noise=0.0):
    rng = np.random.default_rng(seed)
    x = np.arange(N) * (L / N)
    v = np.full(N, veq(N, L, v0, T, s0, ln))
    out_x, out_v = [], []
    steps = int(tmax / dt)
    for k in range(steps):
        t = k * dt
        lead = np.roll(np.arange(N), -1)           # car i follows car i+1 (ahead)
        s = (x[lead] - x - ln) % L
        dv = v - v[lead]
        sstar = s0 + v * T + v * dv / (2 * np.sqrt(a * b))
        acc = a * (1 - (v / v0) ** 4 - (sstar / np.maximum(s, 0.1)) ** 2)
        if noise:
            acc += rng.normal(0, noise, N)
        if brake_t <= t < brake_t + brake_d:
            acc[0] = -4.0
        v = np.maximum(v + acc * dt, 0)
        x = (x + v * dt) % L
        out_x.append(x.copy()); out_v.append(v.copy())
    return np.array(out_x), np.array(out_v), dt

if __name__ == "__main__":
    import sys
    for N, L in ((22, 230.0), (22, 260.0), (26, 230.0)):
        X, V, dt = idm_ring(N, L)
        vm = V.min(1) * 3.6
        sp = [round(vm[int(t / dt)], 1) for t in (5, 12, 20, 40, 80, 140)]
        stopped = [(V[int(t / dt)] < 0.5).sum() for t in (20, 40, 80, 140)]
        print(N, L, "min kmh", sp, "stopped", stopped, "mean", round(V[-1].mean() * 3.6, 1))
