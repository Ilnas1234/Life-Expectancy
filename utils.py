import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
DATA_PATH = os.path.join(REPO, "data", "monthly_interpolated.xlsx")
FIG_DIR = os.path.join(REPO, "results", "figures")
FIG_INDIV_DIR = os.path.join(FIG_DIR, "individual")
TAB_DIR = os.path.join(REPO, "results", "tables")
CACHE_DIR = os.path.join(REPO, "results", "cache")
for d in (FIG_DIR, FIG_INDIV_DIR, TAB_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)

FEATURES = ["CO2", "EA", "GDP", "HE", "NM", "UE", "UP"]
TARGET = "LE"

LB_LSB = np.array([50,   2,   0.01,  1,    0.5])
UB_LSB = np.array([400,  15,   0.50,  25,   1.0])
DIM_LSB = 5

SPLIT_SEED = 42



def apply_plot_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "font.weight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    })

apply_plot_style()


def _bold_ax(ax):
    ax.set_title(ax.get_title(), fontweight="bold")
    ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    if ax.get_legend() is not None:
        for txt in ax.get_legend().get_texts():
            txt.set_fontweight("bold")


def bold_figure(fig):
    for ax in fig.get_axes():
        _bold_ax(ax)


def save_fig(fig, name, also_individual=True):
    """Save figure to figures/<n>.png (and optionally to figures/individual/<n>.png)."""
    bold_figure(fig)
    out_main = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(out_main, dpi=300, bbox_inches="tight")
    if also_individual:
        out_indiv = os.path.join(FIG_INDIV_DIR, f"{name}.png")
        fig.savefig(out_indiv, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_main


def save_individual(fig, name):
    """Save a single-panel figure ONLY to figures/individual/."""
    bold_figure(fig)
    out = os.path.join(FIG_INDIV_DIR, f"{name}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def load_data():
    df = pd.read_excel(DATA_PATH).sort_values("Year").reset_index(drop=True)
    return df


def get_xy(df=None, features=None):
    if df is None:
        df = load_data()
    if features is None:
        features = FEATURES
    return df[features].values.astype(float), df[TARGET].values.astype(float)


def train_test_split_random(X, y, test_frac=0.2, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(round((1 - test_frac) * len(X)))
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], X[te], y[tr], y[te], tr, te


def scaled_split(features=None, seed=SPLIT_SEED):
    df = load_data()
    X, y = get_xy(df, features)
    Xtr_raw, Xte_raw, ytr, yte, tr_idx, te_idx = train_test_split_random(X, y, seed=seed)
    scaler = MinMaxScaler().fit(Xtr_raw)
    return scaler.transform(Xtr_raw), scaler.transform(Xte_raw), ytr, yte, scaler, tr_idx, te_idx


def metrics_all(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    vaf = 100.0 * (1.0 - np.var(y_true - y_pred) / max(np.var(y_true), 1e-12))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "VAF": vaf}


def decode_lsboost(pos):
    n_est = int(round(np.clip(pos[0], LB_LSB[0], UB_LSB[0])))
    mdp   = int(round(np.clip(pos[1], LB_LSB[1], UB_LSB[1])))
    lr    = float(np.clip(pos[2], LB_LSB[2], UB_LSB[2]))
    msl   = max(1, int(round(np.clip(pos[3], LB_LSB[3], UB_LSB[3]))))
    ss    = float(np.clip(pos[4], 0.1, 1.0))
    return n_est, mdp, lr, msl, ss


def make_cv_mae_fitness(X_tr, y_tr, k=5, cv_seed=0, random_state=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=cv_seed)
    folds = list(kf.split(X_tr))
    cache = {}

    def fitness(params):
        n_est, mdp, lr, msl, ss = decode_lsboost(params)
        key = (n_est, mdp, round(lr, 4), msl, round(ss, 3))
        if key in cache:
            return cache[key]
        maes = []
        for tr_i, va_i in folds:
            m = GradientBoostingRegressor(
                loss="squared_error",
                n_estimators=n_est,
                max_depth=mdp,
                learning_rate=lr,
                min_samples_leaf=msl,
                subsample=ss,
                random_state=random_state,
            )
            m.fit(X_tr[tr_i], y_tr[tr_i])
            maes.append(mean_absolute_error(y_tr[va_i], m.predict(X_tr[va_i])))
        val = float(np.mean(maes))
        cache[key] = val
        return val

    return fitness


class OptResult:
    def __init__(self, name):
        self.name = name
        self.best_fit = float("inf")
        self.best_pos = None
        self.curve = []
        self.wall_time = 0.0

    def __repr__(self):
        return f"<{self.name}: best_fit={self.best_fit:.5f}>"


def run_SCA(fitness, pop=20, max_iter=30, seed=0, lb=None, ub=None, dim=None):
    if lb is None: lb = LB_LSB
    if ub is None: ub = UB_LSB
    if dim is None: dim = DIM_LSB
    rng = np.random.default_rng(seed)
    pos = rng.uniform(lb, ub, size=(pop, dim))
    fit = np.array([fitness(p) for p in pos])
    best_idx = int(np.argmin(fit))
    best_pos = pos[best_idx].copy()
    best_fit = float(fit[best_idx])
    curve = np.zeros(max_iter)
    t0 = time.time()
    for l in range(max_iter):
        r1 = 2.0 - l * (2.0 / max_iter)
        for i in range(pop):
            for j in range(dim):
                r2 = 2.0 * np.pi * rng.random()
                r3 = 2.0 * rng.random()
                r4 = rng.random()
                if r4 < 0.5:
                    pos[i, j] = pos[i, j] + r1 * np.sin(r2) * abs(r3 * best_pos[j] - pos[i, j])
                else:
                    pos[i, j] = pos[i, j] + r1 * np.cos(r2) * abs(r3 * best_pos[j] - pos[i, j])
            pos[i] = np.clip(pos[i], lb, ub)
            f_i = fitness(pos[i])
            if f_i < best_fit:
                best_fit = f_i
                best_pos = pos[i].copy()
        curve[l] = best_fit
    res = OptResult("SCA")
    res.best_fit = best_fit; res.best_pos = best_pos
    res.curve = curve; res.wall_time = time.time() - t0
    return res


def run_POA(fitness, pop=20, max_iter=30, seed=0, lb=None, ub=None, dim=None):
    if lb is None: lb = LB_LSB
    if ub is None: ub = UB_LSB
    if dim is None: dim = DIM_LSB
    rng = np.random.default_rng(seed)
    X = rng.uniform(lb, ub, size=(pop, dim))
    fit = np.array([fitness(p) for p in X])
    best_idx = int(np.argmin(fit))
    best_fit = float(fit[best_idx])
    best_pos = X[best_idx].copy()
    curve = np.zeros(max_iter)
    t0 = time.time()
    for t in range(max_iter):
        k = rng.integers(0, pop)
        X_FOOD = X[k].copy(); F_FOOD = fit[k]
        for i in range(pop):
            I = 1 + int(rng.random() < 0.5)
            if fit[i] > F_FOOD:
                X_new = X[i] + rng.random() * (X_FOOD - I * X[i])
            else:
                X_new = X[i] + rng.random() * (X[i] - X_FOOD)
            X_new = np.clip(X_new, lb, ub)
            f_new = fitness(X_new)
            if f_new <= fit[i]:
                X[i] = X_new; fit[i] = f_new
            X_new = X[i] + 0.2 * (1 - t / max_iter) * (2 * rng.random(dim) - 1) * X[i]
            X_new = np.clip(X_new, lb, ub)
            f_new = fitness(X_new)
            if f_new <= fit[i]:
                X[i] = X_new; fit[i] = f_new
            if fit[i] < best_fit:
                best_fit = float(fit[i])
                best_pos = X[i].copy()
        curve[t] = best_fit
    res = OptResult("POA")
    res.best_fit = best_fit; res.best_pos = best_pos
    res.curve = curve; res.wall_time = time.time() - t0
    return res


def _levy(d, rng, beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.standard_normal(d) * sigma
    v = rng.standard_normal(d)
    return u / (np.abs(v) ** (1 / beta))


def run_PO(fitness, pop=20, max_iter=30, seed=0, lb=None, ub=None, dim=None):
    if lb is None: lb = LB_LSB
    if ub is None: ub = UB_LSB
    if dim is None: dim = DIM_LSB
    rng = np.random.default_rng(seed)
    X = rng.uniform(lb, ub, size=(pop, dim))
    fit = np.array([fitness(p) for p in X])
    order = np.argsort(fit); fit = fit[order]; X = X[order]
    best_fit = float(fit[0]); best_pos = X[0].copy()
    curve = np.zeros(max_iter)
    t0 = time.time()
    for i in range(max_iter):
        alpha = rng.random() / 5.0
        sita = rng.random() * math.pi
        X_new = np.zeros_like(X); mean_X = X.mean(axis=0)
        for j in range(pop):
            St = rng.integers(1, 5)
            if St == 1:
                X_new[j] = (X[j] - best_pos) * _levy(dim, rng) + rng.random() * mean_X * (1 - i / max_iter) ** (2 * i / max_iter)
            elif St == 2:
                X_new[j] = X[j] + best_pos * _levy(dim, rng) + rng.standard_normal() * (1 - i / max_iter) * np.ones(dim)
            elif St == 3:
                H = rng.random()
                if H < 0.5:
                    X_new[j] = X[j] + alpha * (1 - i / max_iter) * (X[j] - mean_X)
                else:
                    X_new[j] = X[j] + alpha * (1 - i / max_iter) * math.exp(-j / (rng.random() * max_iter + 1e-9))
            else:
                X_new[j] = (X[j] + rng.random() * math.cos(math.pi * i / (2 * max_iter)) *
                            (best_pos - X[j]) - math.cos(sita) * (i / max_iter) ** (2 / max_iter) *
                            (X[j] - best_pos))
            X_new[j] = np.clip(X_new[j], lb, ub)
        fit_new = np.array([fitness(p) for p in X_new])
        for j in range(pop):
            if fit_new[j] < best_fit:
                best_fit = float(fit_new[j])
                best_pos = X_new[j].copy()
        X = X_new; fit = fit_new
        order = np.argsort(fit); fit = fit[order]; X = X[order]
        curve[i] = best_fit
    res = OptResult("PO")
    res.best_fit = best_fit; res.best_pos = best_pos
    res.curve = curve; res.wall_time = time.time() - t0
    return res


OPTIMIZERS = {"PO": run_PO, "POA": run_POA, "SCA": run_SCA}


def save_and_show_table(df, name, float_fmt=None):
    path_csv = os.path.join(TAB_DIR, f"{name}.csv")
    df.to_csv(path_csv, index=False)
    with pd.option_context("display.max_columns", None,
                           "display.width", 220,
                           "display.float_format", float_fmt):
        print(f"\n[{name}] saved to {path_csv}\n")
        try:
            print(df.to_markdown(index=False))
        except Exception:
            print(df.to_string(index=False))


if __name__ == "__main__":
    df = load_data()
    print(f"Data shape: {df.shape}")
    Xtr, Xte, ytr, yte, sc, _, _ = scaled_split()
    fit = make_cv_mae_fitness(Xtr, ytr, k=3)
    print(f"Fitness at (100,3,0.1,5,0.8): {fit([100, 3, 0.1, 5, 0.8]):.5f}")


def year_grouped_folds(years, k=5, seed=0):
    """Assign calendar years to k folds at random; all months of a year share a fold.
    Returns an integer fold id per row."""
    yrs = np.unique(years)
    perm = np.random.default_rng(seed).permutation(yrs)
    fold_of_year = {y: i % k for i, y in enumerate(perm)}
    return np.array([fold_of_year[y] for y in years])


def annual_records(df=None):
    """The 23 observed annual anchors (January rows) - the non-interpolated dataset."""
    if df is None:
        df = load_data()
    ann = df[df["Year"].dt.month == 1].reset_index(drop=True)
    return ann[FEATURES].values.astype(float), ann[TARGET].values.astype(float), ann["Year"].dt.year.values


def summarize_ci(values, digits=4, deterministic=False):
    """mean +/- 1.96*SE string used throughout the result tables."""
    v = np.asarray(values, dtype=float)
    if deterministic or len(v) < 2:
        return f"{v.mean():.{digits}f} (—)"
    return f"{v.mean():.{digits}f} ± {1.96 * v.std(ddof=1) / np.sqrt(len(v)):.{digits}f}"


def best_hybrid_params():
    """Representative hyperparameter vectors of the best configuration of each optimizer
    (selected by training-only cross-validated fitness; see section_04_optimization.py)."""
    import json
    with open(os.path.join(CACHE_DIR, "best_hybrid_params.json")) as f:
        d = json.load(f)
    return {k: tuple(v["params"]) for k, v in d.items()}, {k: v["pop"] for k, v in d.items()}
