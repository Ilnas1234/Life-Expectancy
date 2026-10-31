# Life expectancy prediction for the United States with metaheuristic-tuned LSBoost

Code and processed data for the article *Predicting Life Expectancy at Birth in the United States with metaheuristic
optimized least squares boosting on socioeconomic and environmental indicators* (I. M. K. Ghonsol and W. M. S. Khalifa).

Archived versions: v1.0.0 Zenodo DOI 10.5281/zenodo.23081107 (second revision); v1.1.0 DOI 10.5281/zenodo.23841107 (third revision: leakage-safe Protocol B, nested tuning, verification script)  ·  Repository: https://github.com/Ilnas1234/Life-Expectancy

## Contents

```
data/
  annual_indicators.csv        23 observed annual records, 2000-2022 (the native data; Protocol C)
  monthly_interpolated.xlsx    276 monthly rows obtained by linear interpolation (Protocols A and B)
  DATA_DICTIONARY.md           sources, indicator codes, units, extraction dates
code/
  utils.py                     data loading, Protocol A split, LSBoost fitness, PO / POA / SCA optimizers
  baselines.py                 six baselines (untuned LSBoost, RF, KELM, SVR, ANN, NumPy LSTM)
  section_01 ... section_16    one script per analysis (see docs/MANUSCRIPT_MAP.md)
  verify_reproduction.py       compares the regenerated tables with the manuscript values
  run_all.py                   runs every analysis in order
  env_report.py                prints CPU, RAM, OS and library versions (manuscript Table 7)
results/cache/
  best_hybrid_params.json      best configuration of each optimizer (Tables 12-13)
  representative_vectors.json  representative vectors of all twelve configurations (Table 8)
docs/MANUSCRIPT_MAP.md         script -> table / figure map and the model-selection rules
requirements.txt, environment.yml
```

## Quick start

```bash
pip install -r requirements.txt
cd code
python run_all.py            # all analyses from the shipped vectors (about 30-60 min on one CPU core)
python run_all.py --full     # also re-runs the metaheuristic search (about 10 h on one CPU core)
python run_all.py --quick    # 2-minute smoke test of the optimizer code (not the paper values)
python run_all.py --nested   # also re-selects the hyperparameters inside every fold of Protocols B and C (about 6 h)
python verify_reproduction.py  # checks the regenerated tables against the manuscript
python env_report.py         # hardware / software environment
```

Tables are written to `results/tables/` and figures to `results/figures/`.

## Data

All indicators are public and annual. The annual value of each variable is assigned to January of its year and the
remaining months are filled by linear interpolation toward the next January; 2022 is carried forward. Life expectancy
is interpolated in the same way. The interpolation was carried out in EViews 10 (IHS Global Inc.); the resulting monthly
file is provided in `data/` and is the input of every analysis script. See `data/DATA_DICTIONARY.md`.

## Evaluation protocols

* **A** - random 80:20 split of the 276 monthly rows (seed 42); optimistic because adjacent interpolated months fall on
  both sides of the split.
* **B** - leakage-safe year-grouped five-fold cross-validation: all twelve months of a year are kept in the same fold,
  and a row is used for training only if every annual anchor it was interpolated from belongs to a training year
  (months 2-12 of year t are computed from the January value of year t+1). Metrics on the 276 pooled rows and on the
  23 held-out January anchors.
* **C** - the 23 observed annual records, leave-one-year-out cross-validation, no interpolated value.

For B and C, "transferred" results reuse the Protocol A hyperparameters and "nested" results re-select them inside
every outer training fold (`section_16_nested_tuning.py`).

## Reproducibility notes

* The shipped vectors are those printed in Table 8 (learning rate rounded to four decimals). With them, the SCA-LSBoost
  results reproduce the manuscript values; PO- and POA-LSBoost reproduce them to within about 0.0006 in the
  reported metrics, the effect of the rounding. `run_all.py --full` regenerates the vectors at full precision.
* Seeds: optimizer runs 0-9 (initialization, random draws and CV folds); refits for the +/- values random_state 0-9;
  Protocol A split seed 42; year-grouped fold assignment seed 0.
* Computation is CPU-only and single-process except Random Forest (`n_jobs=-1`).
* The results were produced with Python 3.10.5 (manuscript Table 7); this package was also verified with Python 3.12,
  scikit-learn 1.8, NumPy 2.4, pandas 3.0, SciPy 1.17 and SHAP 0.52.
