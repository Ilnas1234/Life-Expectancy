import os, json, pickle
import numpy as np, pandas as pd
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import partial_dependence
from utils import scaled_split, annual_records, best_hybrid_params, FEATURES, CACHE_DIR, TAB_DIR
from baselines import make_hybrid

REVERSAL_TOL = 0.20


def shap_rank(model, Xs, feats):
    sv = shap.TreeExplainer(model).shap_values(Xs); return pd.Series(np.abs(sv).mean(0), index=feats)


def representative_vectors():
    """All twelve representative vectors: from the optimization cache if present, else from the shipped
    JSON that reproduces Table 8."""
    pkl = os.path.join(CACHE_DIR, "optimization_results.pkl")
    if os.path.exists(pkl):
        cfg = pickle.load(open(pkl, "rb"))
        return {(o, p): tuple(c["best_params"]) for (o, p), c in cfg.items()}
    d = json.load(open(os.path.join(CACHE_DIR, "representative_vectors.json")))
    return {(o, int(p)): tuple(v["params"]) for o, pp in d.items() if not o.startswith("_") for p, v in pp.items()}


def pdp_transition(grid, v, raw):
    lo, hi = v[:5].mean(), v[-5:].mean(); change = hi - lo; d = np.sign(change) if change != 0 else 1.0
    steps = np.diff(v); reversal = np.abs(steps[np.sign(steps) == -d]).sum()
    monotone = abs(change) > 0 and reversal <= REVERSAL_TOL * abs(change)
    row = {"PDP low plateau": round(lo, 3), "PDP high plateau": round(hi, 3)}
    if monotone:
        def first(q): return raw(grid[np.argmax(d * v >= d * (lo + q * change))])
        row.update({"Shape": "monotone step", "Range (years)": round(abs(change), 3),
                    "Midpoint (raw units)": round(first(0.5), 2),
                    "10-90% band (raw)": f"{first(0.1):.3f} – {first(0.9):.3f}"})
    else:
        row.update({"Shape": "non-monotonic", "Range (years)": round(v.max() - v.min(), 3),
                    "Midpoint (raw units)": "not defined",
                    "10-90% band (raw)": f"not defined; min {v.min():.3f} at {raw(grid[v.argmin()]):.0f}, "
                                          f"max {v.max():.3f} at {raw(grid[v.argmax()]):.0f}"})
    return row


def main():
    X_tr, X_te, y_tr, y_te, scaler, tr_i, te_i = scaled_split()
    hyb, pops = best_hybrid_params(); rk = {}
    for k, p in hyb.items(): rk[f"{k}-LSBoost (pop={pops[k]})"] = shap_rank(make_hybrid(p, 0).fit(X_tr, y_tr), X_te, FEATURES)
    reps = representative_vectors()
    for pop in (10, 30, 40):
        if ("SCA", pop) in reps: rk[f"SCA-LSBoost (pop={pop})"] = shap_rank(make_hybrid(reps[("SCA", pop)], 0).fit(X_tr, y_tr), X_te, FEATURES)
    n = len(tr_i) + len(te_i); Xt_tr = np.c_[X_tr, tr_i / (n - 1)]; Xt_te = np.c_[X_te, te_i / (n - 1)]
    rk["SCA-LSBoost + time index"] = shap_rank(make_hybrid(hyb["SCA"], 0).fit(Xt_tr, y_tr), Xt_te, FEATURES + ["TIME"])
    Xa, ya, _ = annual_records(); scA = MinMaxScaler().fit(Xa)
    rk["SCA-LSBoost (annual, n=23)"] = shap_rank(make_hybrid(hyb["SCA"], 0).fit(scA.transform(Xa), ya), scA.transform(Xa), FEATURES)
    RK = pd.DataFrame(rk).fillna(0).round(3); RK.to_csv(os.path.join(TAB_DIR, "table_19_shap_stability.csv")); print(RK)

    m = make_hybrid(hyb["SCA"], 0).fit(X_tr, y_tr); rows = []
    for f in ["UP", "EA", "NM", "CO2"]:
        j = FEATURES.index(f); pdp = partial_dependence(m, X_tr, [j], grid_resolution=50)
        g, v = pdp["grid_values"][0], pdp["average"][0]
        raw = lambda x, j=j: scaler.data_min_[j] + x * scaler.data_range_[j]
        rows.append({"Feature": f, **pdp_transition(g, v, raw)})
        pd.DataFrame({"grid_raw": raw(g), "partial_dependence": v}).to_csv(
            os.path.join(TAB_DIR, f"pdp_curve_{f}.csv"), index=False)
    out = pd.DataFrame(rows); out.to_csv(os.path.join(TAB_DIR, "table_20_pdp_thresholds.csv"), index=False); print(out.to_string(index=False))


if __name__ == "__main__":
    main()
