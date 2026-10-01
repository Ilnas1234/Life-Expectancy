import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from utils import load_data, scaled_split, best_hybrid_params, apply_plot_style, TAB_DIR, FIG_DIR
from baselines import make_hybrid

def main():
    apply_plot_style(); df = load_data()
    X_tr, X_te, y_tr, y_te, _, tr_i, te_i = scaled_split()
    hyb, _ = best_hybrid_params()
    pe = make_hybrid(hyb["SCA"], 0).fit(X_tr, y_tr).predict(X_te); resid = y_te - pe; order = np.argsort(te_i)
    gap = np.array([min(abs(i - j) for j in tr_i) for i in te_i])
    print(f"Test rows with a training row 1 month away: {np.mean(gap <= 1):.3f}; within 2 months: {np.mean(gap <= 2):.3f}")
    dw = np.sum(np.diff(resid[order]) ** 2) / np.sum(resid ** 2); lag1 = pd.Series(resid[order]).autocorr(1)
    print(f"Residual mean {resid.mean():.4f}, SD {resid.std(ddof=1):.4f}, Shapiro p {stats.shapiro(resid).pvalue:.2e}, DW {dw:.2f}, lag-1 {lag1:.2f}")
    big = pd.DataFrame({"Date": df["Year"].dt.strftime("%Y-%m").values[te_i], "Observed": y_te, "Predicted": pe, "Residual": resid})
    big = big.reindex(np.argsort(-np.abs(resid))[:5]); big.round(4).to_csv(os.path.join(TAB_DIR, "largest_errors_random_split.csv"), index=False); print(big.round(4))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(df["Year"].values[te_i][order], resid[order], "o-", ms=4, color="#118AB2"); ax[0].axhline(0, color="k", lw=1)
    ax[0].set_title("(a) Test residuals vs time (random split)"); ax[0].set_ylabel("Residual (years)")
    ax[1].hist(resid, bins=12, color="#118AB2", edgecolor="white"); ax[1].set_title("(b) Residual distribution"); ax[1].set_xlabel("Residual (years)")
    lags = range(1, 11); acf = [pd.Series(resid[order]).autocorr(l) for l in lags]
    ax[2].bar(list(lags), acf, color="#118AB2"); ax[2].axhline(1.96 / np.sqrt(len(resid)), ls="--", color="gray"); ax[2].axhline(-1.96 / np.sqrt(len(resid)), ls="--", color="gray")
    ax[2].set_title("(c) Residual autocorrelation (test, time-ordered)"); ax[2].set_xlabel("Lag (months)")
    plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "figure_05_residual_diagnostics.png"), dpi=300); plt.close(fig)

if __name__ == "__main__":
    main()
