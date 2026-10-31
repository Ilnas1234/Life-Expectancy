Reference outputs produced by the authors with this release (third revision), kept so that the new Protocol B
results and the nested-tuning results can be inspected without re-running them:

* table_16_year_grouped_cv_leakage_safe.csv  - manuscript Table 16 (transferred rows)
* protocol_B_leakage_audit.csv               - fold-level audit quoted in Section 3.1.2
* nested_B_*.csv, nested_C_*.csv             - nested rows of Tables 16 and 17
* nested_*_fold_summary.csv                  - per-fold selected vectors (manuscript Table 18)
* table_15, table_18-20 (file names keep the second-revision numbering; manuscript numbers are 15, 19, 20, 21)

Running `python run_all.py` regenerates all of them except the nested_* files, which need `--nested` (about 6 h).
