# Manuscript map: which script produces which table or figure

Numbers refer to the revised manuscript (revision 3). Table 18 is new in revision 3 (fold-wise selection summary); the former Tables 18-21 are now Tables 19-22. Run the scripts from the `code/` folder; outputs go to
`results/tables/` (CSV) and `results/figures/` (PNG). `run_all.py` runs them in this order.

| Manuscript item | Content | Script | Output file(s) |
|---|---|---|---|
| Tables 2, 3 | Indicator provenance; observed vs interpolated rows | data files (interpolation done in EViews 10) | `data/annual_indicators.csv`, `data/monthly_interpolated.xlsx`, `data/DATA_DICTIONARY.md` |
| Figure 1 | Distribution of each variable | `section_01_eda.py` | `figure_01_data_distribution.png` |
| Figure 2 | Train/test distributions, Protocol A | `section_03_split_and_distribution.py` | `figure_02_train_test_distributions.png` |
| Tables 8, 9, 14; Figure 4 | Twelve optimizer configurations, runtimes, convergence curves | `section_04_optimization.py` (about 10 h) | `table_08_09_*.csv`, `table_14_computational_cost.csv`, `figure_04_convergence_curves.png`, `results/cache/*.json` |
| Table 10 | Convergence and evaluation-budget audit | `section_07_convergence_audit.py` (needs the section_04 cache) | `table_10_convergence_audit.csv` |
| Table 11 | Ablation (a) and extended bounds (b) | `section_08_ablation_and_bounds.py` | `table_11a_ablation_3v5.csv`, `table_11b_bound_sensitivity.csv` |
| Tables 12, 13 | Hybrids vs six baselines, ten refits | `section_05_baselines.py` | `table_12_13_hybrids_vs_baselines.csv` |
| Table 15 | Paired Wilcoxon / t-tests, Cohen's d, Cliff's delta | `section_09_paired_tests.py` | `table_15_paired_tests.csv` |
| Figure 5 | Residual diagnostics, Protocol A | `section_12_residual_diagnostics.py` | `figure_05_residual_diagnostics.png`, `largest_errors_random_split.csv` |
| Table 16 | Protocol B, leakage-safe year-grouped five-fold CV (transferred rows) and the leakage audit quoted in Section 3.1.2 | `section_15_protocol_b_leakage_safe.py` | `table_16_year_grouped_cv_leakage_safe.csv`, `protocol_B_leakage_audit.csv` |
| Tables 16-17 (nested rows), Table 18 | Fold-wise (nested) hyperparameter selection under Protocols B and C | `section_16_nested_tuning.py` (about 6 h) | `nested_B_*.csv`, `nested_C_*.csv`, `nested_*_fold_summary.csv`, `nested_summary.csv` |
| (superseded) | Protocol B with year grouping only, as in the first two versions; kept for the comparison in Section 4.4 | `section_10_year_grouped_cv.py` | `table_16_year_grouped_cv_SUPERSEDED.csv` |
| Table 17 (transferred rows), Figure 6 | Protocol C, 23 annual records, leave-one-year-out | `section_11_annual_resolution.py` | `table_17_annual_LOYO.csv`, `figure_06_annual_LOYO.png` |
| Table 19 | Temporal structure, correlations, VIF | `section_13_temporal_structure.py` | `table_18_temporal_structure.csv`, `correlation_matrix.csv` |
| Figures 7, 8, 9 | SHAP beeswarm, mean abs. SHAP, partial dependence (representative model SCA-LSBoost) | `section_06_interpretability.py` | `figure_07_08_shap_beeswarm_and_bar.png`, `figure_09_pdp.png` |
| Tables 19, 20 | SHAP stability; PDP transition points | `section_14_shap_stability_and_pdp_thresholds.py` | `table_19_shap_stability.csv`, `table_20_pdp_thresholds.csv`, `pdp_curve_*.csv` |
| Table 7 | Computing environment | `env_report.py` | printed to the console |
| Table 22 | Positioning against previous studies | compiled by hand from the tables above and the cited papers | - |
| check | Regenerated tables vs manuscript values | `verify_reproduction.py` | console report, exit status |

Supplementary outputs that are not in the manuscript: `section_02_feature_selection.py` (feature accumulation and
ablation), the `supp_*` figures of `section_05_baselines.py` (predicted-vs-observed scatter and Taylor diagrams;
removed from the manuscript in revision 2 as redundant with Figures 5-6 and Tables 12-13, 16-17), and
`section_05_baselines.py --extras` (metaheuristic-tuned RF and SVR, about 1 h).

## Model selection (training data only)

* Within each run, the hyperparameter vector with the lowest five-fold CV MAE on the training rows is retained.
* Across the ten seeds of a configuration, the vector with the lowest CV fitness is the representative vector (Table 8).
* Across population sizes, the size with the lowest CV fitness is the best configuration of an optimizer.
* Under Protocols B and C the "transferred" rows reuse the Protocol A vectors (selected with months of every year, so
  not nested); the "nested" rows re-select the five hyperparameters inside every outer training fold
  (`section_16_nested_tuning.py`), with the scaler and the inner folds also contained in the fold.
* Protocol B training sets exclude every row computed from a held-out January anchor (`leakage_safe_train_mask`
  in `section_15_protocol_b_leakage_safe.py`).
* No script uses the test rows to choose a model. The interpretation (Figures 7-9, Table 21) uses SCA-LSBoost as a
  representative tuned model; Table 20 repeats the attribution for PO- and POA-LSBoost.
  `python section_06_interpretability.py --select-by-cv` interprets the hybrid with the lowest training-only CV
  fitness (POA-LSBoost) instead, and `--model PO|POA|SCA` interprets any one of them.
