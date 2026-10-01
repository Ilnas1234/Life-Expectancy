import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import r2_score
import shap

from utils import (
    scaled_split, FEATURES, TARGET, FIG_DIR, FIG_INDIV_DIR, TAB_DIR, CACHE_DIR,
    save_fig, save_individual, apply_plot_style,
)
apply_plot_style()


REPRESENTATIVE = "SCA"


def load_representative_model(X_tr, y_tr):
    import sys
    with open(os.path.join(CACHE_DIR, "best_hybrid_params.json")) as f:
        best_hybrids = json.load(f)
    opt = REPRESENTATIVE
    if "--model" in sys.argv:
        opt = sys.argv[sys.argv.index("--model") + 1]
    elif "--select-by-cv" in sys.argv:
        opt = min(best_hybrids, key=lambda k: best_hybrids[k].get("cv_fitness", np.inf))
    p = best_hybrids[opt]["params"]
    m = GradientBoostingRegressor(loss="squared_error",
                                  n_estimators=p[0], max_depth=p[1],
                                  learning_rate=p[2], min_samples_leaf=p[3],
                                  subsample=p[4], random_state=0)
    m.fit(X_tr, y_tr)
    print(f"Representative hybrid interpreted: {opt}-LSBoost (pop={best_hybrids[opt]['pop']})")
    print(f"  params = n_est={p[0]}, depth={p[1]}, lr={p[2]:.4f}, msl={p[3]}, subsample={p[4]:.3f}")
    return m, opt, p


def figure_15(model, X_te, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    gain_imp = model.feature_importances_

    order = np.argsort(mean_abs_shap)[::-1]
    fnames = [feature_names[i] for i in order]
    mas = mean_abs_shap[order]
    gim = gain_imp[order] / gain_imp.sum()

    fig = plt.figure(figsize=(7.5, 5))
    shap.summary_plot(shap_values, X_te, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title("SHAP summary (beeswarm)", fontweight="bold")
    plt.tight_layout()
    save_individual(plt.gcf(), "figure_15a_shap_beeswarm")

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(fnames))
    ax.barh(y_pos - 0.2, mas, 0.38, color="#118AB2", label="Mean |SHAP|",
            edgecolor="black", linewidth=0.6)
    ax.barh(y_pos + 0.2, gim, 0.38, color="#FFD166", label="Gain importance (norm.)",
            edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos); ax.set_yticklabels(fnames, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance ranking")
    ax.legend(fontsize=10); ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    save_individual(fig, "figure_15b_importance_bars")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_te, feature_names=feature_names,
                      show=False, plot_size=None)
    axes[0].set_title("SHAP summary (beeswarm)")

    axes[1].barh(y_pos - 0.2, mas, 0.38, color="#118AB2", label="Mean |SHAP|",
                 edgecolor="black", linewidth=0.6)
    axes[1].barh(y_pos + 0.2, gim, 0.38, color="#FFD166", label="Gain importance (norm.)",
                 edgecolor="black", linewidth=0.6)
    axes[1].set_yticks(y_pos); axes[1].set_yticklabels(fnames, fontweight="bold")
    axes[1].invert_yaxis(); axes[1].set_xlabel("Importance")
    axes[1].set_title("Feature importance ranking")
    axes[1].legend(fontsize=10); axes[1].grid(axis="x", alpha=0.3, linestyle="--")
    fig.suptitle("SHAP-based feature importance for the representative hybrid model.")
    plt.tight_layout()
    save_fig(fig, "figure_07_08_shap_beeswarm_and_bar", also_individual=False)

    pd.DataFrame({
        "Feature":         fnames,
        "Mean |SHAP|":     np.round(mas, 4),
        "Gain (norm.)":    np.round(gim, 4),
    }).to_csv(os.path.join(TAB_DIR, "supp_shap_importance_ranking.csv"), index=False)

    return shap_values, mean_abs_shap


def figure_16(model, X_tr, feature_names, top_features):
    for i, fname in enumerate(top_features):
        fidx = feature_names.index(fname)
        pdp = partial_dependence(model, X_tr, [fidx], grid_resolution=50)
        grid = pdp["grid_values"][0]; avg = pdp["average"][0]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(grid, avg, lw=2.5, color="#118AB2")
        ax.fill_between(grid, avg.min(), avg, color="#118AB2", alpha=0.12)
        ax.set_xlabel(f"{fname}"); ax.set_ylabel("Partial dependence on LE (years)")
        ax.set_title(f"{fname}")
        ax.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        save_individual(fig, f"figure_16_pdp_{fname}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for i, fname in enumerate(top_features):
        fidx = feature_names.index(fname)
        pdp = partial_dependence(model, X_tr, [fidx], grid_resolution=50)
        grid = pdp["grid_values"][0]; avg = pdp["average"][0]
        axes[i].plot(grid, avg, lw=2.2, color="#118AB2")
        axes[i].fill_between(grid, avg.min(), avg, color="#118AB2", alpha=0.12)
        axes[i].set_xlabel(f"{fname} (scaled)"); axes[i].set_ylabel("Partial dependence on LE")
        axes[i].set_title(f"({chr(ord('a') + i)}) {fname}")
        axes[i].grid(alpha=0.3, linestyle="--")
    fig.suptitle("1D partial dependence plots for top-4 features.")
    plt.tight_layout()
    save_fig(fig, "figure_09_pdp", also_individual=False)

def figure_17(model, X_tr, feature_names, top_features):
    f1_name, f2_name = top_features[0], top_features[1]
    f1, f2 = feature_names.index(f1_name), feature_names.index(f2_name)

    pdp = partial_dependence(model, X_tr, [(f1, f2)], grid_resolution=20)
    grid1 = pdp["grid_values"][0]; grid2 = pdp["grid_values"][1]
    avg = pdp["average"][0]
    G1, G2 = np.meshgrid(grid1, grid2, indexing="ij")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(G1, G2, avg, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel(f1_name, fontweight="bold")
    ax.set_ylabel(f2_name, fontweight="bold")
    ax.set_zlabel("PDP(LE)", fontweight="bold")
    ax.set_title(f"3D PDP ({f1_name} × {f2_name})")
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        lbl.set_fontweight("bold")
    plt.tight_layout()
    save_individual(fig, "figure_17a_3d_pdp")

    explainer = shap.TreeExplainer(model)
    sub = np.random.default_rng(0).choice(len(X_tr),
                                           size=min(150, len(X_tr)), replace=False)
    inter_vals = explainer.shap_interaction_values(X_tr[sub])
    inter_12 = inter_vals[:, f1, f2]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X_tr[sub, f1], X_tr[sub, f2], inter_12,
                    c=inter_12, cmap="coolwarm", s=45, alpha=0.85,
                    edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f1_name, fontweight="bold")
    ax.set_ylabel(f2_name, fontweight="bold")
    ax.set_zlabel("SHAP interaction", fontweight="bold")
    ax.set_title(f"SHAP interaction ({f1_name} × {f2_name})")
    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.1)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        lbl.set_fontweight("bold")
    plt.tight_layout()
    save_individual(fig, "figure_17b_shap_interaction")

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax1.plot_surface(G1, G2, avg, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.set_xlabel(f1_name, fontweight="bold"); ax1.set_ylabel(f2_name, fontweight="bold")
    ax1.set_zlabel("PDP(LE)", fontweight="bold")
    ax1.set_title(f"3D PDP ({f1_name} × {f2_name})")
    fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.1)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc = ax2.scatter(X_tr[sub, f1], X_tr[sub, f2], inter_12,
                     c=inter_12, cmap="coolwarm", s=42, alpha=0.85,
                     edgecolor="black", linewidth=0.3)
    ax2.set_xlabel(f1_name, fontweight="bold"); ax2.set_ylabel(f2_name, fontweight="bold")
    ax2.set_zlabel("SHAP interaction", fontweight="bold")
    ax2.set_title(f"SHAP interaction ({f1_name} × {f2_name})")
    fig.colorbar(sc, ax=ax2, shrink=0.5, pad=0.1)

    for ax in (ax1, ax2):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            lbl.set_fontweight("bold")

    fig.suptitle("Joint partial dependence and SHAP interaction "
                 "for the two most influential features")
    plt.tight_layout()
    save_fig(fig, "supp_3d_pdp_shap", also_individual=False)


# ---------------------------------------------------------------------------
def main():
    X_tr, X_te, y_tr, y_te, *_ = scaled_split()
    model, opt_name, params = load_representative_model(X_tr, y_tr)

    shap_values, mean_abs_shap = figure_15(model, X_te, FEATURES)

    top_idx = np.argsort(mean_abs_shap)[::-1][:4]
    top_features = [FEATURES[i] for i in top_idx]
    print(f"Top-4 features by |SHAP|: {top_features}")

    figure_16(model, X_tr, FEATURES, top_features)
    figure_17(model, X_tr, FEATURES, top_features)


if __name__ == "__main__":
    main()
