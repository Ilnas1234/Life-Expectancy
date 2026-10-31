"""
section_16_nested_tuning.py - Nested (fold-wise) hyperparameter tuning for Protocols B and C
(manuscript Tables 16 and 17, "nested" rows, third revision; Reviewer 3, round 3, Comment 2)

In the second revision the hybrids evaluated under Protocols B and C used the hyperparameter vectors selected
under Protocol A, i.e. with a cross-validated fitness computed on the random 80% training partition of the 276
monthly rows, which contains months of every year.  Those vectors were therefore selected with information from
years that are later held out.  This script repeats Protocols B and C with the whole selection pipeline inside
every outer training fold:

  for each outer fold f:
      training rows  <- outer training rows (leakage-safe rule of section_15 for Protocol B;
                        the 22 remaining annual records for Protocol C)
      scaler         <- MinMaxScaler fitted on the training rows only
      inner CV       <- Protocol B: year-grouped 4-fold CV of the outer training years, same leakage-safe rule
                        Protocol C: 5-fold CV (KFold, shuffle, seed 0) of the 22 training records
      search         <- optimizer (PO / POA / SCA) over the five LSBoost hyperparameters of Table 4,
                        fitness = inner pooled MAE, population N_INNER, T_INNER iterations, seed 0
      final model    <- LSBoost with the selected vector, refitted on all outer training rows
                        (random_state 0..N_REFITS-1 for the +/- values)
      prediction     <- held-out rows of fold f, never used in any step above
  metrics            <- computed once on the pooled out-of-fold predictions

Nothing from a held-out fold enters the scaler, the inner folds, the fitness or the selection.  The budget per
outer fold (N_INNER x T_INNER evaluations) is smaller than the Protocol A search (N up to 40, T = 100, 10 seeds)
because the search is repeated 5 (Protocol B) or 23 (Protocol C) times; Table 10 shows that most of the Protocol A
gain is reached within 30 iterations.  Run time on one CPU core: about 2 h (Protocol B) and 4 h (Protocol C).
    python section_16_nested_tuning.py            # both protocols, all three optimizers
    python section_16_nested_tuning.py B SCA      # one protocol / optimizer
    python section_16_nested_tuning.py B SCA 20,40   # larger inner budget (N = 20, T = 40); outputs get the suffix _N20T40
"""
import os, sys, json, time
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from utils import (load_data, get_xy, metrics_all, summarize_ci, year_grouped_folds, annual_records,
                   decode_lsboost, OPTIMIZERS, TAB_DIR, CACHE_DIR)
from baselines import make_hybrid
from section_15_protocol_b_leakage_safe import leakage_safe_train_mask

N_INNER, T_INNER, SEARCH_SEED = 10, 25, 0
N_INNER_C, T_INNER_C = 8, 20
N_REFITS = 3
K_OUTER_B, K_INNER_B, FOLD_SEED = 5, 4, 0


def make_fitness(X_tr, y_tr, inner_splits):
    cache = {}
    def fitness(pos):
        p = decode_lsboost(pos); key = (p[0], p[1], round(p[2], 4), p[3], round(p[4], 3))
        if key in cache: return cache[key]
        yp = np.zeros_like(y_tr); seen = np.zeros(len(y_tr), bool)
        for a, b in inner_splits:
            sc = MinMaxScaler().fit(X_tr[a]); m = make_hybrid(p, 0).fit(sc.transform(X_tr[a]), y_tr[a])
            yp[b] = m.predict(sc.transform(X_tr[b])); seen[b] = True
        cache[key] = float(np.mean(np.abs(yp[seen] - y_tr[seen]))); return cache[key]
    return fitness


def run_protocol(proto, opt_name, budget=None):
    sfx = f"_N{budget[0]}T{budget[1]}" if budget else ""
    out_csv = os.path.join(TAB_DIR, f"nested_{proto}_{opt_name}{sfx}.csv")
    log_json = os.path.join(CACHE_DIR, f"nested_{proto}_{opt_name}{sfx}_folds.json")
    if proto == "B":
        df = load_data(); X, y = get_xy(df); years = df["Year"].dt.year.values; months = df["Year"].dt.month.values
        fid = year_grouped_folds(years, K_OUTER_B, FOLD_SEED); folds = range(K_OUTER_B)
        n_in, t_in = budget or (N_INNER, T_INNER)
    else:
        X, y, years = annual_records(); months = np.ones(len(y), int); fid = np.arange(len(y)); folds = range(len(y))
        n_in, t_in = budget or (N_INNER_C, T_INNER_C)
    yp = {s: np.zeros_like(y) for s in range(N_REFITS)}; fold_log = []
    if os.path.exists(log_json):
        fold_log = json.load(open(log_json))
        for e in fold_log:
            for s in range(N_REFITS): yp[s][e["test_idx"]] = e["pred"][str(s)]
    done = {e["fold"] for e in fold_log}
    for f in folds:
        if f in done: continue
        te = fid == f
        tr = leakage_safe_train_mask(years, months, te) if proto == "B" else ~te
        tr_i, te_i = np.where(tr)[0], np.where(te)[0]
        X_tr, y_tr = X[tr_i], y[tr_i]
        if proto == "B":
            yrs_tr = years[tr_i]; mo_tr = months[tr_i]; ifid = year_grouped_folds(yrs_tr, K_INNER_B, FOLD_SEED)
            inner = []
            for g in range(K_INNER_B):
                ite = ifid == g; itr = leakage_safe_train_mask(yrs_tr, mo_tr, ite)
                inner.append((np.where(itr)[0], np.where(ite)[0]))
        else:
            inner = list(KFold(5, shuffle=True, random_state=0).split(X_tr))
        t0 = time.time()
        res = OPTIMIZERS[opt_name](make_fitness(X_tr, y_tr, inner), pop=n_in, max_iter=t_in, seed=SEARCH_SEED)
        p = decode_lsboost(res.best_pos)
        sc = MinMaxScaler().fit(X_tr); preds = {}
        for s in range(N_REFITS):
            m = make_hybrid(p, s).fit(sc.transform(X_tr), y_tr); yp[s][te_i] = m.predict(sc.transform(X[te_i])); preds[str(s)] = yp[s][te_i].tolist()
        fold_log.append({"fold": int(f), "held_out_years": [int(v) for v in sorted(set(years[te_i]))], "n_train_rows": int(len(tr_i)),
                         "selected": {"n_estimators": p[0], "max_depth": p[1], "learning_rate": round(p[2], 4), "min_samples_leaf": p[3], "subsample": round(p[4], 3)},
                         "inner_fitness_MAE": round(float(res.best_fit), 4), "evaluations": n_in * t_in * (2 if opt_name == "POA" else 1) + n_in,
                         "search_seconds": round(time.time() - t0, 1), "test_idx": te_i.tolist(), "pred": preds})
        json.dump(fold_log, open(log_json, "w"))
        print(f"[{proto}/{opt_name}] fold {f} years {fold_log[-1]['held_out_years']} selected {p} inner MAE {res.best_fit:.4f} ({time.time()-t0:.0f}s)", flush=True)
    mets = [metrics_all(y, yp[s]) for s in range(N_REFITS)]
    row = {"Protocol": proto, "Model": f"{opt_name}-LSBoost (nested{', N=' + str(n_in) + ', T=' + str(t_in) if budget else ''})", **{k: summarize_ci([q[k] for q in mets], 4) for k in ("R2", "RMSE", "MAE")}}
    if proto == "B":
        jan = months == 1; row["MAE (23 January anchors)"] = summarize_ci([metrics_all(y[jan], yp[s][jan])["MAE"] for s in range(N_REFITS)], 4)
    pd.DataFrame([row]).to_csv(out_csv, index=False); print(row)
    summ = pd.DataFrame([{"Protocol": proto, "Optimizer": opt_name, "Fold": e["fold"], "Held-out years": ", ".join(map(str, e["held_out_years"])),
                          "Training rows": e["n_train_rows"], **e["selected"], "Inner CV MAE": e["inner_fitness_MAE"],
                          "Evaluations": e["evaluations"], "Search time (s)": e["search_seconds"]} for e in fold_log])
    summ.to_csv(os.path.join(TAB_DIR, f"nested_{proto}_{opt_name}{sfx}_fold_summary.csv"), index=False)
    return row


def main():
    args = sys.argv[1:]
    protos = [args[0]] if args else ["B", "C"]; opts = [args[1]] if len(args) > 1 else ["SCA", "PO", "POA"]
    budget = tuple(int(v) for v in args[2].split(",")) if len(args) > 2 else None
    rows = [run_protocol(p, o, budget) for p in protos for o in opts]
    pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "nested_summary.csv"), index=False)


if __name__ == "__main__":
    main()
