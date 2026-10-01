import os
import numpy as np, pandas as pd
from scipy import stats
from utils import scaled_split, metrics_all, best_hybrid_params, TAB_DIR
from baselines import make_hybrid, make_baselines

N_SEEDS = 10

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((np.sum(a[:, None] < b[None, :]) - np.sum(a[:, None] > b[None, :])) / (len(a) * len(b)))

def main():
    X_tr, X_te, y_tr, y_te, *_ = scaled_split()
    hyb, _ = best_hybrid_params()
    mae = {f"{k}-LSBoost": [metrics_all(y_te, make_hybrid(p, s).fit(X_tr, y_tr).predict(X_te))["MAE"] for s in range(N_SEEDS)] for k, p in hyb.items()}
    mae["RF"] = [metrics_all(y_te, make_baselines(s)["RF"].fit(X_tr, y_tr).predict(X_te))["MAE"] for s in range(N_SEEDS)]
    rows = []
    for a, b in [("SCA-LSBoost", "PO-LSBoost"), ("SCA-LSBoost", "POA-LSBoost"), ("PO-LSBoost", "POA-LSBoost"), ("SCA-LSBoost", "RF")]:
        w = stats.wilcoxon(mae[a], mae[b]); t = stats.ttest_rel(mae[a], mae[b])
        d = (np.mean(mae[b]) - np.mean(mae[a])) / np.sqrt((np.var(mae[a], ddof=1) + np.var(mae[b], ddof=1)) / 2)
        rows.append({"Comparison (A vs B)": f"{a} vs {b}", "Mean MAE A": round(np.mean(mae[a]), 4), "Mean MAE B": round(np.mean(mae[b]), 4),
                     "Wilcoxon p": round(w.pvalue, 4), "Paired t p": round(t.pvalue, 4), "Cohen d": round(d, 2),
                     "Cliff delta": round(cliffs_delta(mae[a], mae[b]), 2), "A better in k/10": int(np.sum(np.array(mae[a]) < np.array(mae[b])))})
    out = pd.DataFrame(rows); out.to_csv(os.path.join(TAB_DIR, "table_15_paired_tests.csv"), index=False); print(out.to_string(index=False))

if __name__ == "__main__":
    main()
