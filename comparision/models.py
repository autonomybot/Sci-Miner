"""Surrogate learners compared in the study.  Every learner exposes  fit(X, y) -> self,  predict(X),  expr().

  PathwayForm     physics form of the mined equation  t = a + b ln|G|  with G built from the pathway variables
  FEX             finite expression method, univariate or multivariate (tree: unary-affine -> binary -> unary)
  SINDy           sparse regression over a candidate library (STLSQ)
  PySR            symbolic regression (only when `pysr` + Julia are installed; otherwise skipped, never faked)
  MLP             neural network baseline with a stated architecture and validation-based tuning
"""
import itertools
import numpy as np
import torch

torch.set_num_threads(2)


class Scaler:
    """x -> (x - mu)/sd on features, y likewise.  Magnitudes in E3SM output span 1e-15 .. 1e3."""
    def fit(self, X, y):
        self.mx, self.sx = X.mean(0), X.std(0)
        self.sx = np.where(self.sx == 0, 1.0, self.sx)
        self.my, self.sy = y.mean(), (y.std() or 1.0)
        return self

    def x(self, X): return (X - self.mx) / self.sx
    def y(self, y): return (y - self.my) / self.sy
    def inv(self, p): return p * self.sy + self.my


# ------------------------------------------------------------------ ln-form from the pathway equation
class PathwayForm:
    """t = a + b ln(G).  For the methane case G = eps*dC/dt + |D (C2-C1)| + L, the numerator of
    Eq. (methane-temperature-pathway) evaluated from data; only a, b are fitted.  For other cases G = feature 0
    (the quasi-steady-state form used in the paper)."""
    name = "Pathway ln-form"

    def __init__(self, names): self.names = list(names)

    def _g(self, X):
        n = {k: i for i, k in enumerate(self.names)}
        if {"eps_gas", "dcdt", "conc2", "conc1", "loss", "diffus"} <= set(n):
            g = X[:, n["eps_gas"]] * X[:, n["dcdt"]] + np.abs(X[:, n["diffus"]] * (X[:, n["conc2"]] - X[:, n["conc1"]])) + X[:, n["loss"]]
        else:
            g = X[:, 0]
        return np.log(np.abs(g) + 1e-30)

    def fit(self, X, y):
        self.w = np.linalg.lstsq(np.c_[np.ones(len(X)), self._g(X)], y, rcond=None)[0]
        return self

    def predict(self, X): return self.w[0] + self.w[1] * self._g(X)
    def expr(self): return f"t = {self.w[0]:.5g} + {self.w[1]:.5g} * ln|G|"


# ------------------------------------------------------------------------------ FEX
UN = {"0": (lambda v: v * 0.0, lambda s: "0"), "1": (lambda v: v * 0.0 + 1.0, lambda s: "1"),
      "id": (lambda v: v, lambda s: s), "sq": (lambda v: v ** 2, lambda s: f"({s})^2"),
      "tanh": (torch.tanh, lambda s: f"tanh({s})"),
      "exp": (lambda v: torch.exp(torch.clamp(v, -20, 20)), lambda s: f"exp({s})"),
      "log": (lambda v: torch.log(torch.abs(v) + 1e-12), lambda s: f"ln|{s}|"),
      "sin": (torch.sin, lambda s: f"sin({s})")}
BI = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
UNK, BIK = list(UN), list(BI)


class _Tree(torch.nn.Module):
    """y = c0 + c1 * U3( U1(w1.x+b1) (+|-|*) U2(w2.x+b2) + b3 )   (FEX depth-2 tree with affine leaves)."""
    def __init__(self, d, gen):
        super().__init__()
        w = torch.randn(3, d, generator=gen) * 0.5 / np.sqrt(d)
        w[:, 0] += 0.5
        self.w = torch.nn.Parameter(w)
        self.b = torch.nn.Parameter(torch.randn(3, generator=gen) * 0.3)
        self.o = torch.nn.Parameter(torch.tensor([0.0, 0.5]))

    def forward(self, x, un, bi):
        a = UN[UNK[un[0]]][0](x @ self.w[0] + self.b[0])
        b = UN[UNK[un[1]]][0](x @ self.w[1] + self.b[1])
        z = UN[UNK[un[2]]][0](BI[BIK[bi]](a, b) + self.b[2])
        return self.o[0] + self.o[1] * z

    def expr(self, un, bi, names):
        aff = lambda i: " + ".join(f"{self.w[i, j].item():.3g}*{n}" for j, n in enumerate(names)) + f" + {self.b[i].item():.3g}"
        inner = f"[{UN[UNK[un[0]]][1](aff(0))}] {BIK[bi]} [{UN[UNK[un[1]]][1](aff(1))}] + {self.b[2].item():.3g}"
        return f"{self.o[0].item():.4g} + {self.o[1].item():.4g} * " + UN[UNK[un[2]]][1](inner)


def _fit_tree(t, x, y, un, bi, steps, lr):
    opt = torch.optim.Adam(t.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean((t(x, un, bi) - y) ** 2)
        if not torch.isfinite(loss):
            return float("inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(t.parameters(), 10.0)
        opt.step()
    with torch.no_grad():
        l = torch.mean((t(x, un, bi) - y) ** 2).item()
    return l if np.isfinite(l) else float("inf")


class FEX:
    """Search over operator choices with a policy-gradient controller (Liang & Yang), continuous parameters by Adam.
    Univariate FEX is the same code on a one-column input; the operator-search budget is identical."""
    def __init__(self, names, iters=15, batch=6, steps=80, finetune=400, seed=0, topk=3, search_rows=1500):
        self.search_rows = search_rows
        self.names, self.iters, self.batch, self.steps, self.finetune, self.seed, self.topk = names, iters, batch, steps, finetune, seed, topk
        self.name = "FEX" if len(names) > 1 else "FEX (1-var)"

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        self.sc = Scaler().fit(X, y)
        xf, yf, d = torch.tensor(self.sc.x(X)), torch.tensor(self.sc.y(y)), X.shape[1]
        sub = np.sort(rng.choice(len(xf), min(len(xf), self.search_rows), replace=False))   # operator search on a subsample
        x, yt = xf[sub], yf[sub]
        nu = 3
        lu = torch.zeros(nu, len(UNK), requires_grad=True)
        lb = torch.zeros(1, len(BIK), requires_grad=True)
        copt = torch.optim.Adam([lu, lb], lr=0.1)
        pool = []
        for _ in range(self.iters):
            pu, pb = torch.softmax(lu, 1), torch.softmax(lb, 1)
            cands, rw = [], []
            for _ in range(self.batch):
                un = [int(torch.multinomial(pu[i], 1)) if rng.random() > .1 else int(rng.integers(len(UNK))) for i in range(nu)]
                bi = int(torch.multinomial(pb[0], 1)) if rng.random() > .1 else int(rng.integers(len(BIK)))
                t = _Tree(d, gen).double()
                l = _fit_tree(t, x, yt, un, bi, self.steps, 0.05)
                cands.append((un, bi))
                rw.append(1 / (1 + np.sqrt(l)) if np.isfinite(l) else 0.0)
                if np.isfinite(l):
                    pool.append((l, un, bi, t))
                    pool.sort(key=lambda q: q[0])
                    del pool[self.topk:]
            r = torch.tensor(rw)
            adv, loss = r - r.mean(), 0.0
            for (un, bi), ad in zip(cands, adv):
                lp = sum(torch.log(pu[i][un[i]] + 1e-9) for i in range(nu)) + torch.log(pb[0][bi] + 1e-9)
                loss = loss - ad * lp / len(cands)
            copt.zero_grad(); loss.backward(); copt.step()
        fin = [(_fit_tree(t, xf, yf, un, bi, self.finetune, 0.01), un, bi, t) for _, un, bi, t in pool]
        _, self.un, self.bi, self.t = min(fin, key=lambda q: q[0])
        return self

    def predict(self, X):
        with torch.no_grad():
            return self.sc.inv(self.t(torch.tensor(self.sc.x(X)), self.un, self.bi).numpy())

    def expr(self): return self.t.expr(self.un, self.bi, self.names) + "   (inputs standardised)"


# ----------------------------------------------------------------------------- SINDy
class SINDy:
    """Sparse identification: library Theta(x) = {1, x, x^2, x^3, ln(x+s), 1/(x+s), x_i x_j} on standardised
    inputs, sequentially thresholded least squares (the STLSQ optimiser of SINDy).  The target is a static map, so
    no time-derivative step is involved.  Threshold chosen on the last 20 % of the training data."""
    name = "SINDy (STLSQ)"

    def __init__(self, names, thresholds=(0.003, 0.01, 0.03, 0.1, 0.3), alpha=1e-8):
        self.names, self.thr, self.alpha = names, thresholds, alpha

    def _lib(self, Z):
        cols, lab = [np.ones(len(Z))], ["1"]
        for i, n in enumerate(self.names):
            z = Z[:, i]
            zp = np.maximum(z - self.lo[i], 0) + 1.0          # shifted positive copy for ln and 1/x
            for f, l in ((z, n), (z ** 2, f"{n}^2"), (z ** 3, f"{n}^3"), (np.log(zp), f"ln({n}+s)"),
                         (1 / zp, f"1/({n}+s)")):
                cols.append(f); lab.append(l)
        for i, j in itertools.combinations(range(len(self.names)), 2):
            cols.append(Z[:, i] * Z[:, j]); lab.append(f"{self.names[i]}*{self.names[j]}")
        return np.nan_to_num(np.column_stack(cols)), lab

    def fit(self, X, y):
        self.sc = Scaler().fit(X, y)
        Z = self.sc.x(X)
        self.lo = Z.min(0)
        T, self.lab = self._lib(Z)
        t = self.sc.y(y)
        self.cn = np.linalg.norm(T, axis=0) / np.sqrt(len(T)) + 1e-12
        Tn = T / self.cn
        k = int(.8 * len(t))
        best = min(((np.mean((Tn[k:] @ self._stlsq(Tn[:k], t[:k], thr) - t[k:]) ** 2), thr) for thr in self.thr))
        self.thr_sel = best[1]
        self.w = self._stlsq(Tn, t, self.thr_sel)
        return self

    def _stlsq(self, T, t, thr):
        ls = lambda m: np.linalg.solve(T[:, m].T @ T[:, m] + self.alpha * len(T) * np.eye(m.sum()), T[:, m].T @ t)
        keep = np.ones(T.shape[1], bool)
        w = np.zeros(T.shape[1])
        w[keep] = ls(keep)
        for _ in range(10):
            keep = np.abs(w) >= thr
            if not keep.any():
                return np.zeros_like(w)
            w[:] = 0
            w[keep] = ls(keep)
        return w

    def predict(self, X):
        T, _ = self._lib(self.sc.x(X))
        return self.sc.inv(T / self.cn @ self.w)

    def expr(self):
        w = self.w / self.cn
        nz = np.flatnonzero(self.w)
        return " + ".join(f"{w[i]:.3g}*{self.lab[i]}" for i in nz) + f"   [{len(nz)} terms, standardised inputs]"


# ------------------------------------------------------------------------------ PySR
class PySR:
    name = "PySR"

    def __init__(self, names, niterations=40, seed=0): self.names, self.nit, self.seed = names, niterations, seed

    @staticmethod
    def available():
        import shutil
        if shutil.which("julia") is None:          # importing pysr would otherwise try to download Julia
            return False
        try:
            import pysr  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X, y):
        from pysr import PySRRegressor
        self.sc = Scaler().fit(X, y)
        self.m = PySRRegressor(niterations=self.nit, binary_operators=["+", "-", "*", "/"],
                               unary_operators=["exp", "log", "sqrt", "tanh", "square"], maxsize=25,
                               random_state=self.seed, deterministic=True, parallelism="serial", progress=False, verbosity=0)
        self.m.fit(self.sc.x(X), self.sc.y(y), variable_names=[f"x{i}" for i in range(X.shape[1])])
        return self

    def predict(self, X): return self.sc.inv(self.m.predict(self.sc.x(X)))
    def expr(self): return str(self.m.sympy()) + "   (standardised inputs: " + ", ".join(f"x{i}={n}" for i, n in enumerate(self.names)) + ")"


# ------------------------------------------------------------------------------- MLP
MLP_GRID = [dict(hidden=h, lr=lr, wd=wd) for h in ((32, 32), (64, 64)) for lr in (1e-3, 3e-3) for wd in (0.0, 1e-4)]


class MLP:
    """Fully connected net, tanh, Adam, MSE, mini-batch 512, early stopping (patience 15, <=150 epochs).  Inputs and
    target standardised on the training set.  Hidden sizes, learning rate and weight decay (MLP_GRID, 8 settings) are
    chosen on the last 20 % (chronological) of the training data, never on the test set; the net is then refit on all
    training data for the early-stopped epoch count."""
    def __init__(self, seed=0, max_epochs=150, patience=15):
        self.seed, self.max_epochs, self.patience, self.name = seed, max_epochs, patience, "MLP"

    @staticmethod
    def _net(d, hidden):
        L, p = [], d
        for h in hidden:
            L += [torch.nn.Linear(p, h), torch.nn.Tanh()]
            p = h
        return torch.nn.Sequential(*L, torch.nn.Linear(p, 1)).float()

    @staticmethod
    def _train(net, xt, yt, xv, yv, lr, wd, epochs, patience):
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        best, be, bs, bad = float("inf"), epochs, None, 0
        for ep in range(epochs):
            p = torch.randperm(len(xt))
            for i in range(0, len(xt), 512):
                ix = p[i:i + 512]
                opt.zero_grad()
                torch.mean((net(xt[ix]).squeeze(-1) - yt[ix]) ** 2).backward()
                opt.step()
            if xv is not None:
                with torch.no_grad():
                    v = torch.mean((net(xv).squeeze(-1) - yv) ** 2).item()
                if v < best - 1e-7:
                    best, be, bs, bad = v, ep + 1, {k: q.clone() for k, q in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if bs is not None:
            net.load_state_dict(bs)
        return best, be

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.sc = Scaler().fit(X, y)
        x, t, d = torch.tensor(self.sc.x(X)).float(), torch.tensor(self.sc.y(y)).float(), X.shape[1]
        k, res = int(.8 * len(x)), []
        for g in MLP_GRID:
            net = self._net(d, g["hidden"])
            v, ep = self._train(net, x[:k], t[:k], x[k:], t[k:], g["lr"], g["wd"], self.max_epochs, self.patience)
            res.append((v, ep, g))
        _, ep, self.cfg = min(res, key=lambda r: r[0])
        self.net = self._net(d, self.cfg["hidden"])
        self._train(self.net, x, t, None, None, self.cfg["lr"], self.cfg["wd"], max(ep, 5), 0)
        return self

    def predict(self, X):
        with torch.no_grad():
            return self.sc.inv(self.net(torch.tensor(self.sc.x(X)).float()).squeeze(-1).numpy().astype(float))

    def expr(self): return f"MLP hidden={self.cfg['hidden']} tanh, lr={self.cfg['lr']}, wd={self.cfg['wd']} (no closed form)"
