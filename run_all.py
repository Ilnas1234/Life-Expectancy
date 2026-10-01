import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("section_01_eda.py", "Figure 1: data distribution"),
    ("section_03_split_and_distribution.py", "Figure 2: Protocol A train/test distributions"),
    ("section_08_ablation_and_bounds.py", "Table 11: ablation and extended bounds"),
    ("section_05_baselines.py", "Tables 12-13: hybrids vs baselines (Protocol A)"),
    ("section_09_paired_tests.py", "Table 15: paired tests"),
    ("section_12_residual_diagnostics.py", "Figure 5: residual diagnostics"),
    ("section_10_year_grouped_cv.py", "Table 16: Protocol B"),
    ("section_11_annual_resolution.py", "Table 17 / Figure 6: Protocol C"),
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
        run("section_04_optimization.py")     
        run("section_07_convergence_audit.py")
    for script, what in STEPS:
        print(f"[{what}]"); run(script)
    print("\nAll steps finished. Tables: ../results/tables  Figures: ../results/figures")


if __name__ == "__main__":
    main()
