import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

from utils import (
    load_data, FEATURES, TARGET, FIG_DIR, TAB_DIR,
    save_and_show_table, save_fig, save_individual,
    train_test_split_random, SPLIT_SEED,
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except Exception:
    HAS_LGB = False


def make_models(seed=42):
    models = {
        "LSBoost": GradientBoostingRegressor(loss="squared_error",
                                             n_estimators=200, max_depth=3,
                                             learning_rate=0.01, random_state=seed),
        "RF":      RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        "KELM":    KernelRidge(alpha=0.01, kernel="rbf", gamma=0.5),
        "SVR":     SVR(C=1, gamma="scale"),
        "ANN":     MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000,
                                learning_rate_init=0.01, random_state=seed),
        "LSTM":    MLPRegressor(hidden_layer_sizes=(64,), max_iter=3000,
                                learning_rate_init=0.005, random_state=seed),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=3,
                                         learning_rate=0.01, random_state=seed, verbosity=0)
    if HAS_LGB:
        models["LightGBM"] = LGBMRegressor(n_estimators=200, max_depth=3,
                                           learning_rate=0.03, random_state=seed, verbose=-1)
    return models


def fit_predict(name, model, Xtr, ytr, Xte, yte):
    if name in ("ANN", "LSTM"):
        sc_y = StandardScaler().fit(ytr.reshape(-1, 1))
        ytr_s = sc_y.transform(ytr.reshape(-1, 1)).ravel()
        model.fit(Xtr, ytr_s)
        pred_tr = sc_y.inverse_transform(model.predict(Xtr).reshape(-1, 1)).ravel()
        pred_te = sc_y.inverse_transform(model.predict(Xte).reshape(-1, 1)).ravel()
    else:
        model.fit(Xtr, ytr)
        pred_tr = model.predict(Xtr)
        pred_te = model.predict(Xte)
    return (r2_score(ytr, pred_tr),
            r2_score(yte, pred_te),
            mean_absolute_error(yte, pred_te))


def table_3_vif(df):
    X = df[FEATURES].values.astype(float)
    X_std = StandardScaler().fit_transform(X)
    vif = [variance_inflation_factor(X_std, i) for i in range(X_std.shape[1])]
    tab = pd.DataFrame({"Feature": FEATURES, "VIF": np.round(vif, 3)})
    return tab.sort_values("VIF", ascending=False).reset_index(drop=True)


def ranked_features_by_abs_corr(df):
    corr = df[FEATURES + [TARGET]].corr().loc[FEATURES, TARGET]
    return corr.abs().sort_values(ascending=False).index.tolist()


def feature_accumulation(df):
    features_ranked = ranked_features_by_abs_corr(df)
    print("Features ranked by |corr(LE)|:", features_ranked)

    X_full = df[features_ranked].values.astype(float)
    y_full = df[TARGET].values.astype(float)
    Xtr_raw, Xte_raw, ytr, yte, _, _ = train_test_split_random(X_full, y_full, seed=SPLIT_SEED)

    results = {name: {"R2": [], "MAE": []} for name in make_models().keys()}
    for k in range(1, len(features_ranked) + 1):
        sc = MinMaxScaler().fit(Xtr_raw[:, :k])
        Xtr = sc.transform(Xtr_raw[:, :k])
        Xte = sc.transform(Xte_raw[:, :k])
        models = make_models()
        for name, mdl in models.items():
            _, r2_te, mae_te = fit_predict(name, mdl, Xtr, ytr, Xte, yte)
            results[name]["R2"].append(r2_te)
            results[name]["MAE"].append(mae_te)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    ks = np.arange(1, len(features_ranked) + 1)


    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, vals), c in zip(results.items(), colors):
        ax.plot(ks, vals["R2"], "-o", label=name, color=c, lw=2, ms=5)
    ax.set_xlabel("Number of features added"); ax.set_ylabel("Test R²")
    ax.set_title("Test R² vs. number of features")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(ncol=3, fontsize=9)
    ax.set_xticks(ks); ax.set_xticklabels(features_ranked, fontweight="bold")
    plt.tight_layout()
    save_individual(fig, "figure_07a_accumulation_R2")


    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, vals), c in zip(results.items(), colors):
        ax.plot(ks, vals["MAE"], "-o", label=name, color=c, lw=2, ms=5)
    ax.set_xlabel("Number of features added"); ax.set_ylabel("Test MAE")
    ax.set_title("Test MAE vs. number of features")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(ncol=3, fontsize=9)
    ax.set_xticks(ks); ax.set_xticklabels(features_ranked, fontweight="bold")
    plt.tight_layout()
    save_individual(fig, "figure_07b_accumulation_MAE")


    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    for (name, vals), c in zip(results.items(), colors):
        axes[0].plot(ks, vals["R2"], "-o", label=name, color=c, lw=1.8, ms=4)
        axes[1].plot(ks, vals["MAE"], "-o", label=name, color=c, lw=1.8, ms=4)
    axes[0].set_ylabel("Test R²")
    axes[0].set_title("(a) Test R² vs. number of features")
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].legend(ncol=4, fontsize=9, loc="lower right")
    axes[0].set_xticks(ks); axes[0].set_xticklabels(features_ranked, fontweight="bold")
    axes[1].set_xlabel("Number of features"); axes[1].set_ylabel("Test MAE")
    axes[1].set_title("(b) Test MAE vs. number of features")
    axes[1].grid(alpha=0.3, linestyle="--")
    axes[1].set_xticks(ks); axes[1].set_xticklabels(features_ranked, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "supp_feature_accumulation", also_individual=False)

    rows = []
    for name, vals in results.items():
        for i, k in enumerate(ks):
            rows.append({"Model": name, "k_features": k,
                         "R2_test": round(vals["R2"][i], 4),
                         "MAE_test": round(vals["MAE"][i], 4)})
    pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "supp_feature_accumulation_data.csv"), index=False)
    return features_ranked


def feature_ablation(df):
    X_raw = df[FEATURES].values.astype(float)
    y = df[TARGET].values.astype(float)
    Xtr_raw, Xte_raw, ytr, yte, _, _ = train_test_split_random(X_raw, y, seed=SPLIT_SEED)
    sc = MinMaxScaler().fit(Xtr_raw)
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)

    base = GradientBoostingRegressor(loss="squared_error", n_estimators=200,
                                     max_depth=3, learning_rate=0.1, random_state=42)
    base.fit(Xtr, ytr)
    imp = pd.Series(base.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importance (descending):"); print(imp.to_string(float_format=lambda v: f"{v:.4f}"))

    kept = list(imp.index)
    r2_list, mae_list, labels = [], [], []
    while kept:
        X_sub = df[kept].values.astype(float)
        Xtr_r, Xte_r, ytr_s, yte_s, _, _ = train_test_split_random(X_sub, y, seed=SPLIT_SEED)
        sc2 = MinMaxScaler().fit(Xtr_r)
        Xtr_k, Xte_k = sc2.transform(Xtr_r), sc2.transform(Xte_r)
        m = GradientBoostingRegressor(loss="squared_error", n_estimators=200,
                                      max_depth=3, learning_rate=0.1, random_state=42)
        m.fit(Xtr_k, ytr_s)
        pred = m.predict(Xte_k)
        r2_list.append(r2_score(yte_s, pred))
        mae_list.append(mean_absolute_error(yte_s, pred))
        labels.append(kept[-1])
        kept = kept[:-1]

    r2_list = r2_list[::-1]; mae_list = mae_list[::-1]; labels = labels[::-1]

    xs = np.arange(len(labels))
    imp_ordered = [imp[l] for l in labels]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.bar(xs, imp_ordered, color="#FFD166", alpha=0.85, edgecolor="black",
            label="Feature importance")
    ax1.set_ylabel("Feature importance", color="#8B6F2A", fontweight="bold")
    ax1.set_xticks(xs); ax1.set_xticklabels(labels, fontweight="bold")
    ax1.set_ylim(0, max(imp_ordered) * 1.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, r2_list, "-o", color="#118AB2", lw=2, label="Test R²")
    ax2.plot(xs, mae_list, "-s", color="#EF476F", lw=2, label="Test MAE")
    ax2.set_ylabel("Test R² / MAE", fontweight="bold")

    for lbl in ax2.get_yticklabels(): lbl.set_fontweight("bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    ax1.set_xlabel("Features retained (added in importance order)")
    ax1.set_title("LSBoost performance during feature ablation.")
    plt.tight_layout()
    save_fig(fig, "supp_feature_ablation")

    pd.DataFrame({
        "Feature_added": labels,
        "R2_test": np.round(r2_list, 4),
        "MAE_test": np.round(mae_list, 4),
    }).to_csv(os.path.join(TAB_DIR, "supp_feature_ablation_data.csv"), index=False)


def main():
    df = load_data()
    tab3 = table_3_vif(df)
    save_and_show_table(tab3, "supp_vif")
    print(f"Max VIF: {tab3['VIF'].max():.2f} "
          f"(values > 10 indicate multicollinearity; tree models are robust to it).")
    feature_accumulation(df)
    feature_ablation(df)


if __name__ == "__main__":
    main()
