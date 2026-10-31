"""
run_all.py - Reproduce the tables and figures of the manuscript in order.

    python run_all.py           # every analysis, using the shipped representative vectors (about 30-60 min)
    python run_all.py --full    # also re-runs the metaheuristic search (section_04; about 10 h on one core)
    python run_all.py --quick   # 2-minute smoke test of the search code (results are NOT the paper values)
    python run_all.py --nested  # also runs the fold-wise (nested) tuning of Protocols B and C (about 6 h on one core)

Outputs are written to ../results/tables and ../results/figures.
"""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("section_01_eda.py", "Figure 1: data distribution"),
    ("section_03_split_and_distribution.py", "Figure 2: Protocol A train/test distributions"),
    ("section_08_ablation_and_bounds.py", "Table 11: ablation and extended bounds"),
    ("section_05_baselines.py", "Tables 12-13: hybrids vs baselines (Protocol A)"),
    ("section_09_paired_tests.py", "Table 15: paired tests"),
    ("section_12_residual_diagnostics.py", "Figure 5: residual diagnostics"),
    ("section_10_year_grouped_cv.py", "Protocol B with year grouping only (superseded; kept for the leakage comparison in Section 4.4)"),
    ("section_11_annual_resolution.py", "Table 17 / Figure 6: Protocol C (transferred vectors)"),
    ("section_15_protocol_b_leakage_safe.py", "Table 16: leakage-safe Protocol B (transferred vectors) and leakage audit"),
    ("section_13_temporal_structure.py", "Table 18: temporal structure and VIF"),
    ("section_06_interpretability.py", "Figures 7-9: SHAP and PDP of the representative model"),
    ("section_14_shap_stability_and_pdp_thresholds.py", "Tables 19-20: SHAP stability and PDP transitions"),
]


def run(script, args=()):
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, script, *args], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"{script} failed with exit code {r.returncode}")


def main():
    if "--quick" in sys.argv:
        run("section_04_optimization.py", ["--quick"]); run("section_07_convergence_audit.py", ["--quick"]); return
    if "--full" in sys.argv:
        run("section_04_optimization.py")      # Tables 8-9, 14, Figure 4; rewrites results/cache/*.json
        run("section_07_convergence_audit.py")  # Table 10
    for script, what in STEPS:
        print(f"[{what}]"); run(script)
    if "--nested" in sys.argv:
        run("section_16_nested_tuning.py")     # nested rows of Tables 16-17 and Table 18 (about 6 h on one core)
    print("\nAll steps finished. Tables: ../results/tables  Figures: ../results/figures")
    run("verify_reproduction.py")


if __name__ == "__main__":
    main()
