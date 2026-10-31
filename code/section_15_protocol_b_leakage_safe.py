"""
section_15_protocol_b_leakage_safe.py - Leakage-safe Protocol B (manuscript Table 16, third revision)

Why this script exists (Reviewer 3, round 3, Comment 1)
-------------------------------------------------------
Every annual value, including LE, sits in the January row of its year and months 2-12 are linear interpolations
toward the next January.  Month m (m = 2..12) of year t therefore equals (1 - (m-1)/12) * A_t + ((m-1)/12) * A_{t+1}.
If year t is a training year and year t+1 is held out, months 2-12 of year t contain information from the held-out
anchor A_{t+1} (target AND inputs).  Grouping the twelve months of a year in one fold does NOT remove this.

Leakage-safe rule used here
---------------------------
A row enters the TRAINING set of a fold only if every annual anchor it was computed from belongs to a training year:
  * January rows of training years (observed anchors) are always kept;
  * months 2-12 of a training year t are kept only if year t+1 is also a training year
    (2022 is carried forward from its own anchor and is always kept);
  * everything else is dropped from training (it is never moved to the test set).
Scaler, model and the persistence reference are fitted on these rows only.  Predictions are made for all twelve
months of the held-out years and pooled over the 276 rows, as before; in addition the metrics are reported on the
23 held-out January anchors only, which are the observed annual values and contain no interpolated information.

Hyperparameters of the hybrids are the vectors selected under Protocol A ("transferred"); the nested, fold-wise
tuning requested in Comment 2 is in section_16_nested_tuning.py.
"""
import os
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from utils import load_data, get_xy, metrics_all, summarize_ci, best_hybrid_params, year_grouped_folds, TAB_DIR
from baselines import make_hybrid, make_baselines, DETERMINISTIC

N_REFITS = 5
K = 5
FOLD_SEED = 0


def leakage_safe_train_mask(years, months, test_mask):
    """True for rows that may be used for training without touching a held-out anchor."""
    held = set(np.unique(years[test_mask]))
    last = years.max()
    ok = ~test_mask.copy()
    for i in range(len(years)):
        if ok[i] and months[i] >= 2 and years[i] != last and (years[i] + 1) in held:
            ok[i] = False
    return ok


def audit(years, months, fid):
    rows = []
    for f in range(K):
        te = fid == f
        naive = ~te
        safe = leakage_safe_train_mask(years, months, te)
        rows.append({"Fold": f, "Held-out years": ", ".join(str(y) for y in sorted(set(years[te]))),
                     "Training rows (year grouping only)": int(naive.sum()),
                     "Training rows using a held-out anchor": int(naive.sum() - safe.sum()),
                     "Training rows after leakage-safe rule": int(safe.sum())})
    return pd.DataFrame(rows)


def main():
    df = load_data(); X, y = get_xy(df)
    years = df["Year"].dt.year.values; months = df["Year"].dt.month.values; t = np.arange(len(df))
    fid = year_grouped_folds(years, k=K, seed=FOLD_SEED)
    aud = audit(years, months, fid)
    aud.to_csv(os.path.join(TAB_DIR, "protocol_B_leakage_audit.csv"), index=False); print(aud.to_string(index=False))
    jan = months == 1

    hyb, _ = best_hybrid_params()
    models = {f"{k}-LSBoost": (lambda s, p=p: make_hybrid(p, s)) for k, p in hyb.items()}
    for name in make_baselines(0): models[name] = (lambda s, n=name: make_baselines(s)[n])

    rows = []
    for name, ctor in models.items():
        det = name in DETERMINISTIC; m_all, m_jan = [], []
        for s in range(1 if det else N_REFITS):
            yp = np.zeros_like(y)
            for f in range(K):
                te = fid == f; tr = leakage_safe_train_mask(years, months, te)
                sc = MinMaxScaler().fit(X[tr]); m = ctor(s); m.fit(sc.transform(X[tr]), y[tr]); yp[te] = m.predict(sc.transform(X[te]))
            m_all.append(metrics_all(y, yp)); m_jan.append(metrics_all(y[jan], yp[jan]))
        rows.append({"Model": name,
                     **{f"{k} (276 rows)": summarize_ci([q[k] for q in m_all], 4, det) for k in ("R2", "RMSE", "MAE")},
                     "MAE (23 January anchors)": summarize_ci([q["MAE"] for q in m_jan], 4, det)})
        print(rows[-1], flush=True)

    for nm in ("Linear time-trend", "Persistence (nearest training month)"):
        yp = np.zeros_like(y)
        for f in range(K):
            te = fid == f; tr = leakage_safe_train_mask(years, months, te)
            tr_i, te_i = np.where(tr)[0], np.where(te)[0]
            if nm.startswith("Linear"):
                yp[te_i] = LinearRegression().fit(t[tr_i].reshape(-1, 1), y[tr_i]).predict(t[te_i].reshape(-1, 1))
            else:
                yp[te_i] = [y[tr_i[np.argmin(np.abs(tr_i - i))]] for i in te_i]
        ma, mj = metrics_all(y, yp), metrics_all(y[jan], yp[jan])
        rows.append({"Model": nm, **{f"{k} (276 rows)": f"{ma[k]:.4f} (—)" for k in ("R2", "RMSE", "MAE")},
                     "MAE (23 January anchors)": f"{mj['MAE']:.4f} (—)"})
    out = pd.DataFrame(rows); out.to_csv(os.path.join(TAB_DIR, "table_16_year_grouped_cv_leakage_safe.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
