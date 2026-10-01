import os, sys, pickle
import numpy as np, pandas as pd
from utils import CACHE_DIR, TAB_DIR

def main():
    suffix = "_quick" if "--quick" in sys.argv else ""
    path = os.path.join(CACHE_DIR, f"optimization_results{suffix}.pkl")
    if not os.path.exists(path):
        print(f"{path} not found: run section_04_optimization.py first (Table 10 needs the convergence curves).")
        return
    cfg = pickle.load(open(path, "rb"))
    rows = []
    for (opt, pop), c in cfg.items():
        mc = np.array(c["mean_curve"]); T = len(mc)
        imp = np.where(np.diff(mc) < -1e-9)[0]
        last_imp = int(imp.max() + 2) if len(imp) else 1
        evals_per_iter = pop * (2 if opt == "POA" else 1)
        rows.append({"Optimizer": opt, "Pop": pop, "Evals/iter": evals_per_iter, "Total evals": T * evals_per_iter,
                     "Fitness it.30": round(mc[min(29, T-1)], 5), "Fitness it.50": round(mc[min(49, T-1)], 5), "Fitness final": round(mc[-1], 5),
                     "Gain 50->100 (%)": round(100 * (mc[min(49, T-1)] - mc[-1]) / mc[min(49, T-1)], 2),
                     "Gain 80->100 (%)": round(100 * (mc[min(79, T-1)] - mc[-1]) / mc[min(79, T-1)], 2),
                     "Last improvement (iter)": last_imp, "Mean time/seed (s)": round(float(np.mean(c["times"])), 1),
                     "Time per eval (s)": round(float(np.mean(c["times"])) / (T * evals_per_iter), 3)})
    out = pd.DataFrame(rows).sort_values(["Optimizer", "Pop"])
    os.makedirs(TAB_DIR, exist_ok=True); out.to_csv(os.path.join(TAB_DIR, f"table_10_convergence_audit{suffix}.csv"), index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
