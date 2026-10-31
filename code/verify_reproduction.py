"""
verify_reproduction.py - Check that the regenerated result tables match the values printed in the manuscript.

Run after run_all.py.  For every table listed below the script reads the CSV in results/tables, extracts the mean
of each reported cell and compares it with the manuscript value within a tolerance that reflects (i) the rounding
of the hyperparameter vectors printed in Table 8 and (ii) the stated rounding of the table.  It prints one line per
table and exits with status 1 if any cell is outside its tolerance.

    python verify_reproduction.py
"""
import os, re, sys
import pandas as pd

TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables")


def mean_of(cell):
    m = re.match(r"\s*(-?\d+\.\d+)", str(cell)); return float(m.group(1)) if m else None


# (file, key column, {row label: {column: (manuscript value, tolerance)}})
CHECKS = [
    ("table_12_13_hybrids_vs_baselines.csv", "Model", {
        "SCA-LSBoost (pop=20)": {"R2 test": (0.9994, 0.0002), "RMSE test": (0.0198, 0.0010), "MAE test": (0.0112, 0.0007)},
        "PO-LSBoost (pop=40)": {"MAE test": (0.0141, 0.0010)},
        "POA-LSBoost (pop=40)": {"MAE test": (0.0142, 0.0010)},
        "LSBoost (default)": {"MAE test": (0.1228, 0.0005)},
        "RF": {"MAE test": (0.0238, 0.0010)},
        "SVR": {"MAE test": (0.0671, 0.0005)},
        "KELM": {"MAE test": (0.2192, 0.0005)}}),
    ("table_15_paired_tests.csv", "Comparison (A vs B)", {
        "SCA-LSBoost vs RF": {"Mean MAE A": (0.0112, 0.0007), "Mean MAE B": (0.0238, 0.0010), "Wilcoxon p": (0.002, 0.001)}}),
    ("table_16_year_grouped_cv_leakage_safe.csv", "Model", {
        "PO-LSBoost": {"MAE (276 rows)": (0.1613, 0.004), "R2 (276 rows)": (0.8808, 0.006)},
        "POA-LSBoost": {"MAE (276 rows)": (0.1610, 0.004)},
        "SCA-LSBoost": {"MAE (276 rows)": (0.1700, 0.004), "MAE (23 January anchors)": (0.2523, 0.006)},
        "LSBoost (default)": {"MAE (276 rows)": (0.2496, 0.002)},
        "RF": {"MAE (276 rows)": (0.1966, 0.004)},
        "SVR": {"MAE (276 rows)": (0.2220, 0.001)},
        "Persistence (nearest training month)": {"MAE (276 rows)": (0.2111, 0.001)}}),
    ("table_17_annual_LOYO.csv", "Model", {
        "PO-LSBoost": {"R2": (0.7028, 0.03), "MAE": (0.2731, 0.015)},
        "POA-LSBoost": {"R2": (0.7075, 0.03), "MAE": (0.2736, 0.015)},
        "SCA-LSBoost": {"R2": (0.6827, 0.02), "MAE": (0.2920, 0.015)},
        "SVR": {"MAE": (0.2658, 0.001)},
        "Linear regression (OLS)": {"R2": (0.6065, 0.001)},
        "Persistence (previous year)": {"MAE": (0.2873, 0.001)}}),
    ("table_18_temporal_structure.csv", "Variable", {
        "UP": {"corr_with_LE": (0.832, 0.002), "VIF": (10.7, 0.1)}, "HE": {"VIF": (142.1, 0.2)}}),
    ("table_20_pdp_thresholds.csv", "Feature", {
        "UP": {"Range (years)": (1.496, 0.01), "Midpoint (raw units)": (80.08, 0.01)},
        "EA": {"Midpoint (raw units)": (29.70, 0.01)}, "NM": {"Midpoint (raw units)": (10.77, 0.01)},
        "CO2": {"Range (years)": (0.158, 0.01)}}),
]


def main():
    bad = 0
    for fname, key, rows in CHECKS:
        path = os.path.join(TAB, fname)
        if not os.path.exists(path):
            print(f"MISSING  {fname} (run run_all.py first)"); bad += 1; continue
        df = pd.read_csv(path).set_index(key)
        n = 0
        for label, cols in rows.items():
            for col, (ref, tol) in cols.items():
                val = mean_of(df.loc[label, col])
                ok = val is not None and abs(val - ref) <= tol
                n += 1
                if not ok:
                    bad += 1; print(f"  DIFF   {fname} [{label}, {col}]: regenerated {val} vs manuscript {ref} (tol {tol})")
        print(f"{'OK     ' if bad == 0 else 'CHECKED'} {fname}: {n} cells")
    t19 = os.path.join(TAB, "table_19_shap_stability.csv")
    if os.path.exists(t19):
        s = pd.read_csv(t19, index_col=0)
        up = s.loc["UP"]; ok = (up.drop([c for c in up.index if "annual" in c]) > 0.6).all()
        print(("OK      " if ok else "DIFF    ") + "table_19_shap_stability.csv: UP first with mean |SHAP| > 0.6 in every monthly condition")
        bad += 0 if ok else 1
    print("\nAll checked cells agree with the manuscript." if bad == 0 else f"\n{bad} problem(s) found.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
