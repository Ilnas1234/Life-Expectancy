import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

from utils import (
    scaled_split, make_cv_mae_fitness, metrics_all,
    OPTIMIZERS, FIG_DIR, TAB_DIR, CACHE_DIR,
    save_and_show_table, save_fig, save_individual,
    decode_lsboost, LB_LSB, UB_LSB, DIM_LSB,
)


POP_SIZES  = [10, 20, 30, 40]
MAX_ITER   = 100
N_SEEDS    = 10
import sys
if "--quick" in sys.argv:
    POP_SIZES, MAX_ITER, N_SEEDS = [5], 3, 1
CV_FOLDS   = 5


def run_config(opt_name, pop, max_iter, n_seeds, X_tr, y_tr):
    func = OPTIMIZERS[opt_name]
    all_results, curves, times = [], [], []
    for s in range(n_seeds):
        fit_fn = make_cv_mae_fitness(X_tr, y_tr, k=CV_FOLDS, cv_seed=s)
        res = func(fit_fn, pop=pop, max_iter=max_iter, seed=s)
        all_results.append(res); curves.append(res.curve); times.append(res.wall_time)
    mean_curve = np.mean(np.vstack(curves), axis=0)
    std_curve  = np.std(np.vstack(curves), axis=0)
    best_i = int(np.argmin([r.best_fit for r in all_results]))
    best_res = all_results[best_i]
    return {
        "opt": opt_name, "pop": pop,
        "mean_curve": mean_curve, "std_curve": std_curve,
        "times": times,
        "best_fit":  best_res.best_fit,
        "best_pos":  best_res.best_pos.tolist() if hasattr(best_res.best_pos, "tolist") else list(best_res.best_pos),
        "best_params": decode_lsboost(best_res.best_pos),
    }


def evaluate_params(params, X_tr, y_tr, X_te, y_te, seed=42):
    n_est, mdp, lr, msl, ss = params
    m = GradientBoostingRegressor(loss="squared_error",
                                  n_estimators=n_est, max_depth=mdp,
                                  learning_rate=lr, min_samples_leaf=msl,
                                  subsample=ss, random_state=seed)
    m.fit(X_tr, y_tr)
    return metrics_all(y_tr, m.predict(X_tr)), metrics_all(y_te, m.predict(X_te)), m


def main():
    X_tr, X_te, y_tr, y_te, *_ = scaled_split()
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    suffix = "_quick" if "--quick" in sys.argv else ""
    cache_file = os.path.join(CACHE_DIR, f"optimization_results{suffix}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as fh:
            all_cfgs = pickle.load(fh)
        print(f"Resuming from cache ({len(all_cfgs)} configs already done).")
    else:
        all_cfgs = {}

    total = len(OPTIMIZERS) * len(POP_SIZES); count = 0
    for opt_name in OPTIMIZERS:
        for pop in POP_SIZES:
            count += 1
            if (opt_name, pop) in all_cfgs:
                print(f"  [{count}/{total}] {opt_name} pop={pop}: cached.")
                continue
            t0 = time.time()
            print(f"  [{count}/{total}] {opt_name} pop={pop} ...", end=" ", flush=True)
            cfg = run_config(opt_name, pop, MAX_ITER, N_SEEDS, X_tr, y_tr)
            all_cfgs[(opt_name, pop)] = cfg
            with open(cache_file, "wb") as fh:
                pickle.dump(all_cfgs, fh)
            print(f"best_fit={cfg['best_fit']:.5f} ({time.time()-t0:.0f}s)")

    if len(all_cfgs) < total:
        print(f"\nOnly {len(all_cfgs)}/{total} configs done. Re-run to finish.")
        return
    print(f"\nAll {total} configs complete.")

    for opt_name, t_idx in zip(OPTIMIZERS.keys(), [4, 5, 6]):
        rows = []
        for pop in POP_SIZES:
            cfg = all_cfgs[(opt_name, pop)]
            n_est, mdp, lr, msl, ss = cfg["best_params"]
            tr, te, _ = evaluate_params(cfg["best_params"], X_tr, y_tr, X_te, y_te)
            rows.append({
                "Pop. size": pop,
                "R² (train)": round(tr["R2"], 4),
                "RMSE (train)": round(tr["RMSE"], 4),
                "MAE (train)": round(tr["MAE"], 4),
                "VAF (train)": round(tr["VAF"], 3),
                "R² (test)": round(te["R2"], 4),
                "RMSE (test)": round(te["RMSE"], 4),
                "MAE (test)": round(te["MAE"], 4),
                "VAF (test)": round(te["VAF"], 3),
                "n_est": n_est, "max_depth": mdp,
                "learning_rate": round(lr, 4),
                "min_samples_leaf": msl, "subsample": round(ss, 3),
            })
        save_and_show_table(pd.DataFrame(rows), f"table_08_09_source_{opt_name}_by_pop" + suffix)

    best_cfg = {}
    for opt_name in OPTIMIZERS:
        best_pop = min(POP_SIZES, key=lambda p: all_cfgs[(opt_name, p)]["best_fit"])
        best_cfg[opt_name] = (best_pop, all_cfgs[(opt_name, best_pop)])

    rows = []
    for opt_name in OPTIMIZERS:
        pop, cfg = best_cfg[opt_name]
        n_est, mdp, lr, msl, ss = cfg["best_params"]
        tr, te, _ = evaluate_params(cfg["best_params"], X_tr, y_tr, X_te, y_te)
        rows.append({
            "Hybrid model": f"{opt_name}-LSBoost (pop={pop})",
            "R² (train)": round(tr["R2"], 4),
            "RMSE (train)": round(tr["RMSE"], 4),
            "MAE (train)": round(tr["MAE"], 4),
            "VAF (train)": round(tr["VAF"], 3),
            "R² (test)": round(te["R2"], 4),
            "RMSE (test)": round(te["RMSE"], 4),
            "MAE (test)": round(te["MAE"], 4),
            "VAF (test)": round(te["VAF"], 3),
            "n_est": n_est, "max_depth": mdp,
            "learning_rate": round(lr, 4),
            "min_samples_leaf": msl, "subsample": round(ss, 3),
        })
    save_and_show_table(pd.DataFrame(rows), "table_08_09_best_configurations" + suffix)

    # Table 8
    rows = []
    for opt_name in OPTIMIZERS:
        for pop in POP_SIZES:
            cfg = all_cfgs[(opt_name, pop)]
            total_s = float(np.mean(cfg["times"]))
            rows.append({
                "Model": f"{opt_name}-LSBoost",
                "Pop size": pop,
                "Avg time/iter (s)": round(total_s / MAX_ITER, 2),
                "Total runtime (s)": round(total_s, 1),
                "Std across seeds (s)": round(float(np.std(cfg["times"])), 2),
            })
    save_and_show_table(pd.DataFrame(rows), "table_14_computational_cost" + suffix)

    colors = plt.cm.viridis(np.linspace(0, 0.85, len(POP_SIZES)))


    for opt_name in OPTIMIZERS:
        fig, ax = plt.subplots(figsize=(7, 5))
        for pop, c in zip(POP_SIZES, colors):
            cfg = all_cfgs[(opt_name, pop)]
            curve = cfg["mean_curve"]; std = cfg["std_curve"]
            xs = np.arange(1, len(curve) + 1)
            ax.plot(xs, curve, lw=2, color=c, label=f"Pop={pop}")
            ax.fill_between(xs, curve - std, curve + std, color=c, alpha=0.15)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Fitness (CV MAE)")
        ax.set_title(f"{opt_name}-LSBoost fitness evolution")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(frameon=False)
        plt.tight_layout()
        save_individual(fig, f"figure_09_fitness_{opt_name}" + suffix)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, opt_name in zip(axes, OPTIMIZERS.keys()):
        for pop, c in zip(POP_SIZES, colors):
            cfg = all_cfgs[(opt_name, pop)]
            curve = cfg["mean_curve"]; std = cfg["std_curve"]
            xs = np.arange(1, len(curve) + 1)
            ax.plot(xs, curve, lw=1.8, color=c, label=f"Pop={pop}")
            ax.fill_between(xs, curve - std, curve + std, color=c, alpha=0.15)
        ax.set_title(f"({chr(ord('a') + list(OPTIMIZERS).index(opt_name))}) {opt_name}-LSBoost")
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, frameon=False)
    axes[0].set_ylabel("Fitness (CV MAE)")
    fig.suptitle("Fitness evolution curves of the three hybrid models")
    plt.tight_layout()
    save_fig(fig, "figure_04_convergence_curves" + suffix, also_individual=False)

    test_metrics = {}
    for opt_name in OPTIMIZERS:
        for pop in POP_SIZES:
            cfg = all_cfgs[(opt_name, pop)]
            _, te, _ = evaluate_params(cfg["best_params"], X_tr, y_tr, X_te, y_te)
            test_metrics[(opt_name, pop)] = te

    def _norm(values, key):
        vals = np.array(values, dtype=float)
        if key in ("RMSE", "MAE"):
            v = (vals.max() - vals + 1e-12) / (vals.max() - vals.min() + 1e-12)
        else:
            v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        return v

    metric_colors = ["#EF476F", "#118AB2", "#FFD166", "#06D6A0"]
    metric_keys = ["R2", "RMSE", "MAE", "VAF"]

    def _draw_panel(ax, names, sector_vals_getter, title):
            theta = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
            width = 2 * np.pi / len(names) * 0.85
            stacked = np.zeros(len(names))
            for mk, mc in zip(metric_keys, metric_colors):
                vals = sector_vals_getter(mk)
                heights = _norm(vals, mk)
                ax.bar(theta, heights, bottom=stacked, width=width,
                       color=mc, edgecolor="white", alpha=0.85, label=mk)
                stacked += heights
            ax.set_xticks(theta)
            ax.set_xticklabels([])
            rmax = stacked.max() * 1.15
            ax.set_ylim(0, rmax)
            for ang, name in zip(theta, names):
                ax.text(ang, rmax * 1.08, name,
                        ha="center", va="center", fontweight="bold", fontsize=10)
            ax.set_yticks([])
            ax.set_title(title, pad=20)

    for opt_name in OPTIMIZERS:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")
        names = [f"Pop={p}" for p in POP_SIZES]
        _draw_panel(ax, names,
                    lambda k, o=opt_name: [test_metrics[(o, p)][k] for p in POP_SIZES],
                    f"({opt_name}-LSBoost)")
        best_pop_opt = min(POP_SIZES, key=lambda p: all_cfgs[(opt_name, p)]["best_fit"])
        i_best = POP_SIZES.index(best_pop_opt)
        theta_b = np.linspace(0, 2 * np.pi, len(POP_SIZES), endpoint=False)[i_best]
        rmax = ax.get_ylim()[1]
        ax.plot(theta_b, rmax * 0.92, marker="*", color="red",
                markersize=22, zorder=5,
                markeredgecolor="black", markeredgewidth=0.8)
        ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.08), frameon=False)
        save_individual(fig, f"figure_10_radar_{opt_name}" + suffix)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11),
                             subplot_kw={"projection": "polar"})
    axes = axes.flatten()
    subtitles = [f"(a) {list(OPTIMIZERS)[0]}-LSBoost", f"(b) {list(OPTIMIZERS)[1]}-LSBoost",
                 f"(c) {list(OPTIMIZERS)[2]}-LSBoost", "(d) Best hybrids"]

    for ax_idx, opt_name in enumerate(list(OPTIMIZERS.keys())):
        ax = axes[ax_idx]
        names = [f"Pop={p}" for p in POP_SIZES]
        _draw_panel(ax, names,
                    lambda k, o=opt_name: [test_metrics[(o, p)][k] for p in POP_SIZES],
                    subtitles[ax_idx])

    ax = axes[3]
    names = list(OPTIMIZERS.keys())
    def _best_metric_by_opt(k):
        vals = []
        for opt_name in names:
            pop, cfg = best_cfg[opt_name]
            _, te, _ = evaluate_params(cfg["best_params"], X_tr, y_tr, X_te, y_te)
            vals.append(te[k])
        return vals
    _draw_panel(ax, [f"{n}\n(pop={best_cfg[n][0]})" for n in names],
                _best_metric_by_opt, subtitles[3])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), frameon=False)
    fig.suptitle("Figure 10. Performance stacking chart across population sizes.")
    plt.tight_layout()
    save_fig(fig, "supp_radar_hyperparameters" + suffix, also_individual=False)

    best_summary = {
        opt: {"pop": best_cfg[opt][0],
              "params": list(best_cfg[opt][1]["best_params"]),
              "cv_fitness": round(float(best_cfg[opt][1]["best_fit"]), 6),
              "best_pos": best_cfg[opt][1]["best_pos"]}
        for opt in OPTIMIZERS
    }
    with open(os.path.join(CACHE_DIR, f"best_hybrid_params{suffix}.json"), "w") as fh:
        json.dump(best_summary, fh, indent=2)
    rep = {opt: {str(pop): {"params": list(all_cfgs[(opt, pop)]["best_params"]),
                            "cv_fitness": round(float(all_cfgs[(opt, pop)]["best_fit"]), 6)}
                 for pop in POP_SIZES} for opt in OPTIMIZERS}
    with open(os.path.join(CACHE_DIR, f"representative_vectors{suffix}.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print("\nBest hybrid hyperparameters:")
    for k, v in best_summary.items():
        print(f"  {k}-LSBoost: pop={v['pop']}, params={v['params']}")


if __name__ == "__main__":
    main()
