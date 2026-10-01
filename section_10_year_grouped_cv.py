import os
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from utils import load_data, get_xy, metrics_all, summarize_ci, best_hybrid_params, year_grouped_folds, TAB_DIR
from baselines import make_hybrid, make_baselines, DETERMINISTIC

N_REFITS = 5

def main():
    df = load_data(); X, y = get_xy(df); years = df["Year"].dt.year.values; t = np.arange(len(df))
    fid = year_grouped_folds(years, k=5, seed=0)
    hyb, _ = best_hybrid_params()
    models = {f"{k}-LSBoost": (lambda s, p=p: make_hybrid(p, s)) for k, p in hyb.items()}
    for name in make_baselines(0): models[name] = (lambda s, n=name: make_baselines(s)[n])
    rows = []
    for name, ctor in models.items():
        det = name in DETERMINISTIC; mets = []
        for s in range(1 if det else N_REFITS):
            yp = np.zeros_like(y)
            for f in range(5):
                tr, te = fid != f, fid == f
                sc = MinMaxScaler().fit(X[tr]); m = ctor(s); m.fit(sc.transform(X[tr]), y[tr]); yp[te] = m.predict(sc.transform(X[te]))
            mets.append(metrics_all(y, yp))
        rows.append({"Model": name, **{k: summarize_ci([q[k] for q in mets], 4, det) for k in ("R2", "RMSE", "MAE")}})
        print(rows[-1], flush=True)
    for nm in ("Linear time-trend", "Persistence (nearest train month)"):
        yp = np.zeros_like(y)
        for f in range(5):
            tr, te = np.where(fid != f)[0], np.where(fid == f)[0]
            if nm.startswith("Linear"): yp[te] = LinearRegression().fit(t[tr].reshape(-1, 1), y[tr]).predict(t[te].reshape(-1, 1))
            else: yp[te] = [y[tr[np.argmin(np.abs(tr - i))]] for i in te]
        mm = metrics_all(y, yp); rows.append({"Model": nm, **{k: f"{mm[k]:.4f} (—)" for k in ("R2", "RMSE", "MAE")}})
    out = pd.DataFrame(rows); out.to_csv(os.path.join(TAB_DIR, "table_16_year_grouped_cv.csv"), index=False); print(out.to_string(index=False))

if __name__ == "__main__":
    main()
