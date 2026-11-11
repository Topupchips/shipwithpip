"""Real numpy training helpers for the learning reels (MLP + Adam, train/test split, heatmap snapshots)."""
import numpy as np

GN = 96
_gx, _gy = np.meshgrid(np.linspace(-1.15, 1.15, GN), np.linspace(-1.15, 1.15, GN))
GRID = np.stack([_gx.ravel(), _gy.ravel()], 1)


def spiral_data(n=250, seed=0, noise=0.03, test_frac=0.4):
    r = np.random.default_rng(seed)
    X, y = [], []
    for c in (0, 1):
        t = np.sort(r.uniform(0.25, 1, n))
        ang = t * 3.6 * np.pi + c * np.pi
        X.append(np.stack([t * np.cos(ang), t * np.sin(ang)], 1) + r.normal(0, noise, (n, 2)))
        y += [c] * n
    X, y = np.concatenate(X), np.array(y, float)
    idx = r.permutation(len(y))
    nt = int(len(y) * test_frac)
    te, tr = idx[:nt], idx[nt:]
    return X[tr], y[tr], X[te], y[te]


class MLP:
    def __init__(self, sizes, seed=1, act=np.tanh, out="sigmoid", scale=None):
        r = np.random.default_rng(seed)
        self.sizes, self.out = sizes, out
        self.P = []
        for a, b in zip(sizes, sizes[1:]):
            s = (scale or np.sqrt(1.0 / a)) * (1.6 if a == sizes[0] and scale is None else 1.0)
            self.P += [r.normal(0, s, (a, b)), np.zeros(b)]
        self.m = [np.zeros_like(p) for p in self.P]
        self.v = [np.zeros_like(p) for p in self.P]
        self.step = 0

    def forward(self, X):
        hs = [X]
        n = len(self.P) // 2
        for i in range(n):
            z = hs[-1] @ self.P[2 * i] + self.P[2 * i + 1]
            hs.append(np.tanh(z) if i < n - 1 else z)
        z = hs[-1]
        return hs, (1 / (1 + np.exp(-z)) if self.out == "sigmoid" else z)

    def predict(self, X):
        return self.forward(X)[1]

    def train_step(self, X, y, lr):
        hs, p = self.forward(X)
        if self.out == "sigmoid":
            d = (p - y.reshape(p.shape)) / len(X)
        else:
            d = 2 * (p - y) / p.size
        n = len(self.P) // 2
        g = [None] * len(self.P)
        for i in range(n - 1, -1, -1):
            g[2 * i] = hs[i].T @ d
            g[2 * i + 1] = d.sum(0)
            if i > 0:
                d = (d @ self.P[2 * i].T) * (1 - hs[i] ** 2)
        self.step += 1
        for i in range(len(self.P)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g[i]
            self.v[i] = 0.999 * self.v[i] + 0.001 * g[i] ** 2
            mh = self.m[i] / (1 - 0.9 ** self.step)
            vh = self.v[i] / (1 - 0.999 ** self.step)
            self.P[i] -= lr * mh / (np.sqrt(vh) + 1e-8)


def acc(net, X, y):
    return float(np.mean((net.predict(X).ravel() > 0.5) == (y > 0.5)))


def loss_bce(net, X, y):
    p = net.predict(X).ravel()
    return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))


def run_spiral(widths, seed=1, epochs=600, lr=0.01, data_seed=0, keep_grid=True):
    Xtr, ytr, Xte, yte = spiral_data(seed=data_seed)
    net = MLP([2] + list(widths) + [1], seed=seed)
    H = dict(epoch=[], loss=[], tr=[], te=[], grid=[])
    for e in range(epochs + 1):
        H["epoch"].append(e)
        H["loss"].append(loss_bce(net, Xtr, ytr))
        H["tr"].append(acc(net, Xtr, ytr))
        H["te"].append(acc(net, Xte, yte))
        if keep_grid:
            H["grid"].append(net.predict(GRID).reshape(GN, GN).astype(np.float32))
        net.train_step(Xtr, ytr, lr)
    return (Xtr, ytr, Xte, yte), H
