import os
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from utils import scaled_split, metrics_all, summarize_ci, best_hybrid_params, TAB_DIR
from baselines import make_hybrid

N_SEEDS = 10

def main():
    X_tr, X_te, y_tr, y_te, *_ = scaled_split()
    hyb, _ = best_hybrid_params(); p5 = hyb["SCA"]
    folds = list(KFold(5, shuffle=True, random_state=0).split(X_tr))
    def cv_mae(p):
        return float(np.mean([np.mean(np.abs(y_tr[b] - make_hybrid(p, 0).fit(X_tr[a], y_tr[a]).predict(X_tr[b]))) for a, b in folds]))
    def test_stats(p, n=N_SEEDS):
        m = [metrics_all(y_te, make_hybrid(p, s).fit(X_tr, y_tr).predict(X_te)) for s in range(n)]
        return m
    abl = []
    for lab, p in [("5-dim optimum (SCA)", p5), ("3-dim (min_samples_leaf=1, subsample=1.0)", (p5[0], p5[1], p5[2], 1, 1.0))]:
        m = test_stats(p); tr = [metrics_all(y_tr, make_hybrid(p, s).fit(X_tr, y_tr).predict(X_tr))["MAE"] for s in range(3)]
        abl.append({"Configuration": lab, "params": str(tuple(round(x, 4) for x in p)), "CV MAE (train)": round(cv_mae(p), 4),
                    "Train MAE": summarize_ci(tr), "Test MAE": summarize_ci([q["MAE"] for q in m]), "Test RMSE": summarize_ci([q["RMSE"] for q in m])})
    abl = pd.DataFrame(abl); abl.to_csv(os.path.join(TAB_DIR, "table_11a_ablation_3v5.csv"), index=False); print(abl.to_string(index=False))
    sens = []
    for ne in (400, 600, 800, 1000):
        for ss in (0.3, 0.4, 0.5):
            p = (ne, p5[1], p5[2], p5[3], ss)
            sens.append({"n_estimators": ne, "subsample": ss, "CV MAE (train)": round(cv_mae(p), 4),
                         "Test MAE (3 refits)": round(float(np.mean([q["MAE"] for q in test_stats(p, 3)])), 4)})
    sens = pd.DataFrame(sens); sens.to_csv(os.path.join(TAB_DIR, "table_11b_bound_sensitivity.csv"), index=False); print(sens.to_string(index=False))

if __name__ == "__main__":
    main()
