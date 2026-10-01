import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from baselines import LSTMWrap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

from utils import (
    scaled_split, metrics_all, FIG_DIR, FIG_INDIV_DIR, TAB_DIR, CACHE_DIR,
    save_and_show_table, save_fig, save_individual, apply_plot_style,
    OPTIMIZERS, FEATURES, TARGET,
)
apply_plot_style()


N_REPS = 10 
DETERMINISTIC = {"KELM", "SVR"}


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------
def make_baselines():
    def _lsb(s):  return GradientBoostingRegressor(loss="squared_error",
                                                   n_estimators=200, max_depth=3,
                                                   learning_rate=0.01, random_state=s)
    def _rf(s):   return RandomForestRegressor(n_estimators=100, random_state=s, n_jobs=-1)
    def _kelm(s): return KernelRidge(alpha=0.01, kernel="rbf", gamma=0.5)
    def _svr(s):  return SVR(C=10, gamma="scale")
    def _ann(s):  return MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000,
                                      learning_rate_init=0.01, random_state=s)
    def _lstm(s): return LSTMWrap(s)
    return {"LSBoost (default)": _lsb, "RF": _rf, "KELM": _kelm,
            "SVR": _svr, "ANN": _ann, "LSTM": _lstm}


def fit_and_predict(name, model, X_tr, y_tr, X_te, y_te):
    if name == "ANN":
        sc_y = StandardScaler().fit(y_tr.reshape(-1, 1))
        y_tr_s = sc_y.transform(y_tr.reshape(-1, 1)).ravel()
        model.fit(X_tr, y_tr_s)
        pred_tr = sc_y.inverse_transform(model.predict(X_tr).reshape(-1, 1)).ravel()
        pred_te = sc_y.inverse_transform(model.predict(X_te).reshape(-1, 1)).ravel()
    else:
        model.fit(X_tr, y_tr)
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
    return pred_tr, pred_te


# ---------------------------------------------------------------------------
# Repeated runs for stochastic models
# ---------------------------------------------------------------------------
def run_reps(name, ctor, X_tr, y_tr, X_te, y_te, n_reps):
    tr_list, te_list = [], []
    preds_tr_first, preds_te_first = None, None
    reps = 1 if name in DETERMINISTIC else n_reps
    for s in range(reps):
        m = ctor(s)
        pt, pe = fit_and_predict(name, m, X_tr, y_tr, X_te, y_te)
        tr_list.append(metrics_all(y_tr, pt))
        te_list.append(metrics_all(y_te, pe))
        if s == 0:
            preds_tr_first, preds_te_first = pt, pe
    return tr_list, te_list, preds_tr_first, preds_te_first


def run_hybrid_reps(params, X_tr, y_tr, X_te, y_te, n_reps):
    n_est, mdp, lr, msl, ss = params
    tr_list, te_list = [], []
    preds_tr_first, preds_te_first = None, None
    for s in range(n_reps):
        m = GradientBoostingRegressor(loss="squared_error",
                                      n_estimators=n_est, max_depth=mdp,
                                      learning_rate=lr, min_samples_leaf=msl,
                                      subsample=ss, random_state=s)
        m.fit(X_tr, y_tr)
        pt, pe = m.predict(X_tr), m.predict(X_te)
        tr_list.append(metrics_all(y_tr, pt))
        te_list.append(metrics_all(y_te, pe))
        if s == 0:
            preds_tr_first, preds_te_first = pt, pe
    return tr_list, te_list, preds_tr_first, preds_te_first


def summarize(metric_list, key, digits, deterministic=False):
    vals = np.array([m[key] for m in metric_list])
    mean = vals.mean()
    if deterministic or len(vals) < 2:
        return f"{mean:.{digits}f} (—)"
    ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def table_9(X_tr, y_tr, X_te, y_te, best_hybrids):
    rows = []
    predictions = {}

    for opt_name, info in best_hybrids.items():
        pop = info["pop"]; params = tuple(info["params"])
        tr_l, te_l, pt, pe = run_hybrid_reps(params, X_tr, y_tr, X_te, y_te, N_REPS)
        row = {"Model": f"{opt_name}-LSBoost (pop={pop})"}
        for k in ("R2", "RMSE", "MAE"): row[f"{k} train"] = summarize(tr_l, k, 4)
        row["VAF train"] = summarize(tr_l, "VAF", 3)
        for k in ("R2", "RMSE", "MAE"): row[f"{k} test"] = summarize(te_l, k, 4)
        row["VAF test"] = summarize(te_l, "VAF", 3)
        rows.append(row)
        predictions[f"{opt_name}-LSBoost"] = (pt, pe)

    baselines = make_baselines()
    for name, ctor in baselines.items():
        det = name in DETERMINISTIC
        tr_l, te_l, pt, pe = run_reps(name, ctor, X_tr, y_tr, X_te, y_te, N_REPS)
        row = {"Model": name}
        for k in ("R2", "RMSE", "MAE"): row[f"{k} train"] = summarize(tr_l, k, 4, det)
        row["VAF train"] = summarize(tr_l, "VAF", 3, det)
        for k in ("R2", "RMSE", "MAE"): row[f"{k} test"]  = summarize(te_l, k, 4, det)
        row["VAF test"] = summarize(te_l, "VAF", 3, det)
        rows.append(row)
        predictions[name] = (pt, pe)

    save_and_show_table(pd.DataFrame(rows), "table_12_13_hybrids_vs_baselines")
    return predictions


def figure_11(predictions, y_tr, y_te):
    def _panel(ax, pred_tr, pred_te, name):
        ax.scatter(y_tr, pred_tr, s=22, alpha=0.55, color="#118AB2",
                   edgecolors="none", label="Train")
        ax.scatter(y_te, pred_te, s=30, alpha=0.85, color="#FF7F50",
                   edgecolors="none", label="Test")
        lo = min(y_tr.min(), y_te.min()) - 0.2
        hi = max(y_tr.max(), y_te.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.8, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Observed LE"); ax.set_ylabel("Predicted LE")

        r2_tr = r2_score(y_tr, pred_tr); r2_te = r2_score(y_te, pred_te)
        rmse_te = np.sqrt(mean_squared_error(y_te, pred_te))
        mae_te = mean_absolute_error(y_te, pred_te)
        ax.text(0.03, 0.97,
                f"Train R²={r2_tr:.4f}\nTest R²={r2_te:.4f}\n"
                f"Test RMSE={rmse_te:.3f}\nTest MAE={mae_te:.3f}",
                transform=ax.transAxes, va="top", fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", linewidth=0.5))
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="lower right")


    for name, (pt, pe) in predictions.items():
        fig, ax = plt.subplots(figsize=(6, 5.5))
        _panel(ax, pt, pe, name)
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        save_individual(fig, f"figure_11_scatter_{safe}")

    names = list(predictions.keys())
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.7 * nrow))
    axes = axes.flatten()
    for i, name in enumerate(names):
        pt, pe = predictions[name]
        _panel(axes[i], pt, pe, name)
    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Figure 11. Predicted vs. observed for all models.")
    plt.tight_layout()
    save_fig(fig, "supp_scatter_multipanel", also_individual=False)


def _taylor_coords(y_obs, y_pred):
    std_pred = np.std(y_pred, ddof=1)
    corr = np.corrcoef(y_obs, y_pred)[0, 1]
    return std_pred, corr


def _draw_taylor(ax, predictions, y_obs, which, colors):
    std_obs = np.std(y_obs, ddof=1)
    ax.plot(0, std_obs, "k*", markersize=20, label="Observation")

    max_sp = std_obs
    for (name, (pt, pe)), c in zip(predictions.items(), colors):
        y_pred = pt if which == "train" else pe
        std_p, corr = _taylor_coords(y_obs, y_pred)
        theta = np.arccos(np.clip(corr, -1, 1))
        ax.plot(theta, std_p, "o", color=c, markersize=12, label=name,
                markeredgecolor="black", markeredgewidth=0.8)
        max_sp = max(max_sp, std_p)

    corr_ticks = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                           0.8, 0.9, 0.95, 0.99, 1.0])
    theta_ticks = np.arccos(corr_ticks)
    ax.set_xticks(theta_ticks)
    ax.set_xticklabels([f"{c:.2f}" for c in corr_ticks],
                       fontweight="bold", color="#1F4E96")

    ax.set_thetamin(0); ax.set_thetamax(90)
    rmax = max_sp * 1.15
    ax.set_rlim(0, rmax)

    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_color("#A93226")
        

def figure_12(predictions, y_tr, y_te):
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))

    def _add_axis_labels(fig, ax):
        bbox = ax.get_position()
        x0, x1 = bbox.x0, bbox.x1
        y0, y1 = bbox.y0, bbox.y1

        fig.text(x0 - 0.02, (y0 + y1) / 2, "Standard Deviation",
                 ha="center", va="center", rotation=90,
                 fontweight="bold", color="#A93226", fontsize=13)

        fig.text((x0 + x1) / 2, y0 - 0.04, "Observed",
                 ha="center", va="center",
                 fontweight="bold", color="#1B7A50", fontsize=13)

        fig.text(x1 - 0.10, y1 - 0.20, "Correlation",
                 ha="center", va="center", rotation=-45,
                 fontweight="bold", color="#1F4E96", fontsize=13)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")
    _draw_taylor(ax, predictions, y_tr, "train", colors)
    fig.text(0.06, 0.93, "Training Result Taylor Diagram", fontweight="bold", fontsize=18)
    ax.legend(loc="center left", bbox_to_anchor=(1.20, 0.5),
              frameon=True, fancybox=False, edgecolor="black", fontsize=10)
    plt.subplots_adjust(left=0.15, right=0.70, top=0.88, bottom=0.12)
    _add_axis_labels(fig, ax)
    save_individual(fig, "figure_12a_taylor_train")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")
    _draw_taylor(ax, predictions, y_te, "test", colors)
    fig.text(0.06, 0.93, "Test Result Taylor Diagram", fontweight="bold", fontsize=18)
    ax.legend(loc="center left", bbox_to_anchor=(1.20, 0.5),
              frameon=True, fancybox=False, edgecolor="black", fontsize=10)
    plt.subplots_adjust(left=0.15, right=0.70, top=0.88, bottom=0.12)
    _add_axis_labels(fig, ax)
    save_individual(fig, "figure_12b_taylor_test")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             subplot_kw={"projection": "polar"})
    _draw_taylor(axes[0], predictions, y_tr, "train", colors)
    _draw_taylor(axes[1], predictions, y_te, "test", colors)
    fig.text(0.04, 0.92, "(a) Train set", fontweight="bold", fontsize=14)
    fig.text(0.46, 0.92, "(b) Test set",  fontweight="bold", fontsize=14)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.20, 0.5),
                   frameon=True, fancybox=False, edgecolor="black", fontsize=10)
    fig.suptitle("Figure 12. Taylor diagram.", y=0.99)
    plt.subplots_adjust(left=0.07, right=0.78, top=0.86, bottom=0.10, wspace=0.35)
    _add_axis_labels(fig, axes[0])
    _add_axis_labels(fig, axes[1])
    save_fig(fig, "supp_taylor_diagrams", also_individual=False)
    plt.close("all")

def _decode_rf(pos, lb, ub):
    n_est = int(np.clip(pos[0], lb[0], ub[0]))
    max_d = int(np.clip(pos[1], lb[1], ub[1]))
    min_s = max(2, int(np.clip(pos[2], lb[2], ub[2])))
    return n_est, max_d, min_s


def _decode_svr(pos, lb, ub):
    C = float(np.clip(pos[0], lb[0], ub[0]))
    gamma = float(np.clip(pos[1], lb[1], ub[1]))
    epsilon = float(np.clip(pos[2], lb[2], ub[2]))
    return C, gamma, epsilon


def table_10(X_tr, y_tr, X_te, y_te):
    plt.close("all")
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    folds = list(kf.split(X_tr))

    rf_lb = np.array([50, 3, 2]); rf_ub = np.array([300, 20, 15])
    rf_cache = {}
    def rf_fit(p):
        n_est, max_d, min_s = _decode_rf(p, rf_lb, rf_ub)
        key = (n_est, max_d, min_s)
        if key in rf_cache: return rf_cache[key]
        maes = []
        for tr_i, va_i in folds:
            m = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                      min_samples_split=min_s, random_state=0, n_jobs=-1)
            m.fit(X_tr[tr_i], y_tr[tr_i])
            maes.append(mean_absolute_error(y_tr[va_i], m.predict(X_tr[va_i])))
        rf_cache[key] = float(np.mean(maes))
        return rf_cache[key]

    svr_lb = np.array([0.1, 1e-3, 1e-3]); svr_ub = np.array([100.0, 10.0, 0.5])
    svr_cache = {}
    def svr_fit(p):
        C, gamma, eps = _decode_svr(p, svr_lb, svr_ub)
        key = (round(C, 2), round(gamma, 4), round(eps, 4))
        if key in svr_cache: return svr_cache[key]
        maes = []
        for tr_i, va_i in folds:
            m = SVR(C=C, gamma=gamma, epsilon=eps)
            m.fit(X_tr[tr_i], y_tr[tr_i])
            maes.append(mean_absolute_error(y_tr[va_i], m.predict(X_tr[va_i])))
        svr_cache[key] = float(np.mean(maes))
        return svr_cache[key]

    rows = []
    for opt_name, func in OPTIMIZERS.items():
        # RF
        res = func(rf_fit, pop=5, max_iter=4, seed=0, lb=rf_lb, ub=rf_ub, dim=3)
        n_est, max_d, min_s = _decode_rf(res.best_pos, rf_lb, rf_ub)
        m = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                  min_samples_split=min_s, random_state=0, n_jobs=-1)
        m.fit(X_tr, y_tr)
        tr = metrics_all(y_tr, m.predict(X_tr))
        te = metrics_all(y_te, m.predict(X_te))
        rows.append({
            "Optimized model": f"{opt_name}-RF",
            "R² (train)":  round(tr["R2"], 4), "RMSE (train)": round(tr["RMSE"], 4),
            "MAE (train)": round(tr["MAE"], 4),
            "R² (test)":   round(te["R2"], 4), "RMSE (test)": round(te["RMSE"], 4),
            "MAE (test)":  round(te["MAE"], 4),
            "hyperparams": f"n_est={n_est}, max_depth={max_d}, min_samples_split={min_s}",
            "CV MAE (train)": round(float(res.best_fit), 5),
        })
        print(f"  {opt_name}-RF: Test R²={te['R2']:.4f} MAE={te['MAE']:.4f}")

        res = func(svr_fit, pop=5, max_iter=4, seed=0, lb=svr_lb, ub=svr_ub, dim=3)
        C, gamma, eps = _decode_svr(res.best_pos, svr_lb, svr_ub)
        m = SVR(C=C, gamma=gamma, epsilon=eps)
        m.fit(X_tr, y_tr)
        tr = metrics_all(y_tr, m.predict(X_tr))
        te = metrics_all(y_te, m.predict(X_te))
        rows.append({
            "Optimized model": f"{opt_name}-SVR",
            "R² (train)":  round(tr["R2"], 4), "RMSE (train)": round(tr["RMSE"], 4),
            "MAE (train)": round(tr["MAE"], 4),
            "R² (test)":   round(te["R2"], 4), "RMSE (test)": round(te["RMSE"], 4),
            "MAE (test)":  round(te["MAE"], 4),
            "hyperparams": f"C={C:.2f}, gamma={gamma:.4f}, epsilon={eps:.4f}",
            "CV MAE (train)": round(float(res.best_fit), 5),
        })
        print(f"  {opt_name}-SVR: Test R²={te['R2']:.4f} MAE={te['MAE']:.4f}")

    save_and_show_table(pd.DataFrame(rows), "supp_optimized_rf_svr")
    return rows

def figure_13(X_tr, y_tr, X_te, y_te, best_hybrids):
    # Default LSBoost
    m = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  random_state=0).fit(X_tr, y_tr)
    lsb_def = {"train": metrics_all(y_tr, m.predict(X_tr)),
               "test":  metrics_all(y_te, m.predict(X_te))}

    best_opt = "SCA"
    params = best_hybrids[best_opt]["params"]
    m = GradientBoostingRegressor(n_estimators=params[0], max_depth=params[1],
                                  learning_rate=params[2], min_samples_leaf=params[3],
                                  subsample=params[4], random_state=0).fit(X_tr, y_tr)
    lsb_opt = {"train": metrics_all(y_tr, m.predict(X_tr)),
               "test":  metrics_all(y_te, m.predict(X_te))}

    m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    rf_def = {"train": metrics_all(y_tr, m.predict(X_tr)),
              "test":  metrics_all(y_te, m.predict(X_te))}

    tab10_path = os.path.join(TAB_DIR, "supp_optimized_rf_svr.csv")
    tab10 = pd.read_csv(tab10_path)
    rf_rows = tab10[tab10["Optimized model"].str.endswith("-RF")]
    best_rf_row = rf_rows.loc[rf_rows["CV MAE (train)"].idxmin()]
    hp = best_rf_row["hyperparams"]
    parts = {p.split("=")[0].strip(): p.split("=")[1].strip() for p in hp.split(",")}
    m = RandomForestRegressor(n_estimators=int(parts["n_est"]),
                              max_depth=int(parts["max_depth"]),
                              min_samples_split=int(parts["min_samples_split"]),
                              random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    rf_opt = {"train": metrics_all(y_tr, m.predict(X_tr)),
              "test":  metrics_all(y_te, m.predict(X_te))}

    data = {
        "LSBoost": {"orig_tr": lsb_def["train"], "orig_te": lsb_def["test"],
                    "opt_tr":  lsb_opt["train"], "opt_te":  lsb_opt["test"]},
        "RF":      {"orig_tr": rf_def["train"],  "orig_te": rf_def["test"],
                    "opt_tr":  rf_opt["train"],  "opt_te":  rf_opt["test"]},
    }
    models = ["LSBoost", "RF"]
    metric_keys = ["R2", "RMSE", "MAE", "VAF"]

    def _panel(ax, mk):
        x = np.arange(len(models)); w = 0.2
        orig_tr = [data[m]["orig_tr"][mk] for m in models]
        orig_te = [data[m]["orig_te"][mk] for m in models]
        opt_tr  = [data[m]["opt_tr"][mk]  for m in models]
        opt_te  = [data[m]["opt_te"][mk]  for m in models]
        ax.bar(x - 1.5*w, orig_tr, w, label="Original train", color="#9DB4C0",
               edgecolor="black", linewidth=0.6)
        ax.bar(x - 0.5*w, orig_te, w, label="Original test",  color="#5C6B73",
               edgecolor="black", linewidth=0.6)
        ax.bar(x + 0.5*w, opt_tr,  w, label="Optimized train", color="#FFADAD",
               edgecolor="black", linewidth=0.6)
        ax.bar(x + 1.5*w, opt_te,  w, label="Optimized test",  color="#D90429",
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(models, fontweight="bold")
        ax.set_title(mk, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    for mk in metric_keys:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        _panel(ax, mk)
        ax.legend(fontsize=9, loc="best")
        plt.tight_layout()
        save_individual(fig, f"figure_13_{mk}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for mi, mk in enumerate(metric_keys):
        _panel(axes[mi], mk)
        if mi == 0:
            axes[mi].legend(fontsize=9, loc="lower right")
    fig.suptitle("Figure 13. Metrics before and after metaheuristic optimization.")
    plt.tight_layout()
    save_fig(fig, "supp_before_after_tuning", also_individual=False)

    print(f"Using {best_opt}-LSBoost as the optimized variant in Figure 13.")


def figure_14(X_tr, y_tr, X_te, y_te, best_hybrids):
    best_opt = "SCA"

    p = best_hybrids[best_opt]["params"]
    m = GradientBoostingRegressor(loss="squared_error", n_estimators=p[0],
                                  max_depth=p[1], learning_rate=p[2],
                                  min_samples_leaf=p[3], subsample=p[4],
                                  random_state=0)
    m.fit(X_tr, y_tr)
    train_losses, test_losses = [], []
    for pred_tr, pred_te in zip(m.staged_predict(X_tr), m.staged_predict(X_te)):
        train_losses.append(mean_absolute_error(y_tr, pred_tr))
        test_losses.append(mean_absolute_error(y_te, pred_te))

    xs = np.arange(1, p[0] + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, train_losses, "-",  color="#118AB2", lw=2.2, label="Train MAE")
    ax.plot(xs, test_losses,  "--", color="#EF476F", lw=2.2, label="Test MAE")
    ax.set_xlabel("Boosting iteration"); ax.set_ylabel("MAE (years)")
    ax.set_title(f"Figure 14. Loss trajectories of {best_opt}-LSBoost "
                 f"(n_est={p[0]}, depth={p[1]}, lr={p[2]:.3f}, "
                 f"msl={p[3]}, subsample={p[4]:.2f}).")
    ax.legend(frameon=False); ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    save_fig(fig, "supp_loss_curves")


def main():
    X_tr, X_te, y_tr, y_te, *_ = scaled_split()
    with open(os.path.join(CACHE_DIR, "best_hybrid_params.json")) as f:
        best_hybrids = json.load(f)

    predictions = table_9(X_tr, y_tr, X_te, y_te, best_hybrids)
    figure_11(predictions, y_tr, y_te)    
    figure_12(predictions, y_tr, y_te)                         
    if "--extras" in sys.argv:
        table_10(X_tr, y_tr, X_te, y_te)
        figure_13(X_tr, y_tr, X_te, y_te, best_hybrids)
        figure_14(X_tr, y_tr, X_te, y_te, best_hybrids)


if __name__ == "__main__":
    main()
