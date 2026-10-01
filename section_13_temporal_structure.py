import os
import numpy as np, pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils import load_data, FEATURES, TARGET, TAB_DIR

def main():
    df = load_data(); t = np.arange(len(df)); rows = []
    Xv = df[FEATURES].values.astype(float)
    vif = {f: variance_inflation_factor(Xv, i) for i, f in enumerate(FEATURES)}
    for c in FEATURES + [TARGET]:
        s = pd.Series(df[c].values)
        rows.append({"Variable": c, "corr_with_time": round(np.corrcoef(t, s)[0, 1], 3), "lag1_autocorr": round(s.autocorr(1), 4),
                     "lag12_autocorr": round(s.autocorr(12), 3), "corr_with_LE": round(np.corrcoef(s, df[TARGET])[0, 1], 3),
                     "VIF": round(vif[c], 1) if c in vif else "—"})
    out = pd.DataFrame(rows); out.to_csv(os.path.join(TAB_DIR, "table_18_temporal_structure.csv"), index=False); print(out.to_string(index=False))
    df[FEATURES + [TARGET]].corr().round(3).to_csv(os.path.join(TAB_DIR, "correlation_matrix.csv"))

if __name__ == "__main__":
    main()
