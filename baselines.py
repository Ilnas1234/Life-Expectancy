import math
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class NPLSTM:
    def __init__(self, n_in, units=64, lr=0.005, batch=16, epochs=500, patience=30, seed=0, val_frac=0.1):
        self.u, self.lr, self.b, self.ep, self.pat, self.seed, self.vf = units, lr, batch, epochs, patience, seed, val_frac
        rng = np.random.default_rng(seed); g = math.sqrt(6 / (n_in + units))
        self.W = {k: rng.uniform(-g, g, (n_in, units)) for k in "ifgo"}
        self.bias = {k: np.zeros(units) for k in "ifgo"}; self.bias["f"][:] = 1.0
        self.wo = rng.uniform(-math.sqrt(6/(units+1)), math.sqrt(6/(units+1)), units); self.bo = 0.0
        self.n_in = n_in
    def _fwd(self, X):
        sig = lambda z: 1/(1+np.exp(-z))
        i = sig(X@self.W["i"]+self.bias["i"]); g = np.tanh(X@self.W["g"]+self.bias["g"]); o = sig(X@self.W["o"]+self.bias["o"])
        c = i*g; tc = np.tanh(c); h = o*tc; y = h@self.wo + self.bo
        return y, (i, g, o, c, tc, h)
    def _grads(self, X, yt):
        y, (i, g, o, c, tc, h) = self._fwd(X); n = len(X)
        dy = 2*(y-yt)/n
        gwo = h.T@dy; gbo = dy.sum()
        dh = np.outer(dy, self.wo)
        do = dh*tc; dc = dh*o*(1-tc**2); di = dc*g; dg = dc*i
        dzo = do*o*(1-o); dzi = di*i*(1-i); dzg = dg*(1-g**2)
        G = {"i": (X.T@dzi, dzi.sum(0)), "g": (X.T@dzg, dzg.sum(0)), "o": (X.T@dzo, dzo.sum(0))}
        return G, gwo, gbo, np.mean((y-yt)**2)
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed); n = len(X); idx = rng.permutation(n); nv = max(1, int(self.vf*n))
        vi, ti = idx[:nv], idx[nv:]; Xt, yt, Xv, yv = X[ti], y[ti], X[vi], y[vi]
        params = [self.W["i"], self.W["g"], self.W["o"], self.bias["i"], self.bias["g"], self.bias["o"], self.wo]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]; mb = vb = 0.0
        b1, b2, eps, k = 0.9, 0.999, 1e-8, 0; best = np.inf; bad = 0; snap = None
        for e in range(self.ep):
            perm = rng.permutation(len(Xt))
            for s in range(0, len(Xt), self.b):
                bi = perm[s:s+self.b]; G, gwo, gbo, _ = self._grads(Xt[bi], yt[bi]); k += 1
                grads = [G["i"][0], G["g"][0], G["o"][0], G["i"][1], G["g"][1], G["o"][1], gwo]
                for j, (p, gr) in enumerate(zip(params, grads)):
                    m[j] = b1*m[j]+(1-b1)*gr; v[j] = b2*v[j]+(1-b2)*gr**2
                    p -= self.lr*(m[j]/(1-b1**k))/(np.sqrt(v[j]/(1-b2**k))+eps)
                mb = b1*mb+(1-b1)*gbo; vb = b2*vb+(1-b2)*gbo**2
                self.bo -= self.lr*(mb/(1-b1**k))/(math.sqrt(vb/(1-b2**k))+eps)
            vl = np.mean((self._fwd(Xv)[0]-yv)**2)
            if vl < best-1e-7: best, bad, snap = vl, 0, ([p.copy() for p in params], self.bo)
            else:
                bad += 1
                if bad >= self.pat: break
        if snap: 
            for p, q in zip(params, snap[0]): p[...] = q
            self.bo = snap[1]
        self.epochs_run = e+1; return self
    def predict(self, X): return self._fwd(X)[0]

class LSTMWrap:
    def __init__(self, seed): self.seed = seed
    def fit(self, X, y):
        self.sc = StandardScaler().fit(y.reshape(-1,1)); ys = self.sc.transform(y.reshape(-1,1)).ravel()
        self.m = NPLSTM(X.shape[1], seed=self.seed).fit(X, ys); return self
    def predict(self, X): return self.sc.inverse_transform(self.m.predict(X).reshape(-1,1)).ravel()

class ANNWrap:
    def __init__(self, seed, hl=(32,16), lr=0.01): self.m = MLPRegressor(hidden_layer_sizes=hl, max_iter=3000, learning_rate_init=lr, random_state=seed)
    def fit(self, X, y):
        self.sc = StandardScaler().fit(y.reshape(-1,1)); self.m.fit(X, self.sc.transform(y.reshape(-1,1)).ravel()); return self
    def predict(self, X): return self.sc.inverse_transform(self.m.predict(X).reshape(-1,1)).ravel()


DETERMINISTIC = {"KELM", "SVR"}


def make_baselines(seed):
    return {
        "LSBoost (default)": GradientBoostingRegressor(loss="squared_error", n_estimators=200, max_depth=3,
                                                       learning_rate=0.01, subsample=1.0, min_samples_leaf=1,
                                                       random_state=seed),
        "RF": RandomForestRegressor(n_estimators=100, random_state=seed),
        "KELM": KernelRidge(alpha=0.01, kernel="rbf", gamma=0.5),
        "SVR": SVR(C=10, gamma="scale", epsilon=0.1),
        "ANN": ANNWrap(seed),
        "LSTM": LSTMWrap(seed),
    }


def make_hybrid(params, seed):
    n_est, mdp, lr, msl, ss = params
    return GradientBoostingRegressor(loss="squared_error", n_estimators=int(n_est), max_depth=int(mdp),
                                     learning_rate=float(lr), min_samples_leaf=int(msl), subsample=float(ss),
                                     random_state=seed)
