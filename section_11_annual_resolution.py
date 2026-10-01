import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from utils import annual_records, metrics_all, summarize_ci, best_hybrid_params, apply_plot_style, TAB_DIR, FIG_DIR
from baselines import make_hybrid, make_baselines, DETERMINISTIC

N_REFITS = 5

def main():
    apply_plot_style()
    Xa, ya, ta = annual_records(); n = len(ya)
    hyb, _ = best_hybrid_params()
    models = {f"{k}-LSBoost": (lambda s, p=p: make_hybrid(p, s)) for k, p in hyb.items()}
    for name in make_baselines(0): models[name] = (lambda s, nm=name: make_baselines(s)[nm])
    models["Linear regression (OLS)"] = lambda s: LinearRegression()
    rows, preds = [], {}
    for name, ctor in models.items():
        det = name in DETERMINISTIC or name.startswith("Linear"); mets = []
        for s in range(1 if det else N_REFITS):
            yp = np.zeros(n)
            for i in range(n):
                tr = np.arange(n) != i; sc = MinMaxScaler().fit(Xa[tr]); m = ctor(s); m.fit(sc.transform(Xa[tr]), ya[tr]); yp[i] = m.predict(sc.transform(Xa[[i]]))[0]
            mets.append(metrics_all(ya, yp))
            if s == 0: preds[name] = yp
        rows.append({"Model": name, **{k: summarize_ci([q[k] for q in mets], 4, det) for k in ("R2", "RMSE", "MAE")}}); print(rows[-1], flush=True)
    yp = np.array([LinearRegression().fit(np.delete(ta, i).reshape(-1, 1), np.delete(ya, i)).predict([[ta[i]]])[0] for i in range(n)]); preds["Linear time-trend"] = yp
    mm = metrics_all(ya, yp); rows.append({"Model": "Linear time-trend", **{k: f"{mm[k]:.4f} (—)" for k in ("R2", "RMSE", "MAE")}})
    yp = np.r_[ya[0], ya[:-1]]; mm = metrics_all(ya[1:], yp[1:]); rows.append({"Model": "Persistence (previous year)", **{k: f"{mm[k]:.4f} (—)" for k in ("R2", "RMSE", "MAE")}})
    out = pd.DataFrame(rows); out.to_csv(os.path.join(TAB_DIR, "table_17_annual_LOYO.csv"), index=False); print(out.to_string(index=False))
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.plot(ta, ya, "ko-", lw=2, label="Observed (annual)")
    for name, c in [("SCA-LSBoost", "#D62828"), ("RF", "#2A9D8F"), ("Linear regression (OLS)", "#3A86FF"), ("Linear time-trend", "#8338EC")]:
        ax.plot(ta, preds[name], "s--", ms=4, lw=1.3, color=c, label=f"{name} (LOYO)")
    ax.set_xlabel("Year"); ax.set_ylabel("Life expectancy (years)"); ax.set_title("Leave-one-year-out predictions at native annual resolution (n = 23)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, ls="--"); plt.tight_layout(); os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "figure_06_annual_LOYO.png"), dpi=300); plt.close(fig)

if __name__ == "__main__":
    main()
