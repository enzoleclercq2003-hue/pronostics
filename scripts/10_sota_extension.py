"""Étape 10 (expérience) : les variables de l'état de l'art améliorent-elles le modèle ?

On ajoute les pi-ratings et les forces estimées sur les tirs cadrés, et on mesure l'effet en
walk-forward (2020/21 -> 2024/25) avec les mêmes hyperparamètres. Le test final n'est pas consulté.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as cfg
from src.data import load_processed
from src.extra_features import PI_COLS, SOT_COLS, build_extra_features
from src.features import select_features
from src.folds import load_fold, split
from src.metrics import evaluate, paired_bootstrap, per_match_losses
from src.models import Model, load_params

extra = build_extra_features(load_processed())
extra.to_pickle(cfg.PROCESSED_DIR / "extra_features.pkl")
print(f"Variables supplémentaires : {extra.shape}")
print(extra.tail().to_string(float_format="%.3f"))

MODELS = ["logreg", "lgbm"]
CONFIGS = {"complet (référence)": [], "+ pi-ratings": PI_COLS, "+ forces tirs cadrés": SOT_COLS,
           "+ pi-ratings + tirs cadrés": PI_COLS + SOT_COLS}
preds = {(m, c): [] for m in MODELS for c in CONFIGS}
ys = []
for season in cfg.WALK_FORWARD_SEASONS:
    fold = load_fold(season).merge(extra, on="match_id", how="left")
    train, test = split(fold, season)
    base = select_features(load_fold(season).columns)
    ys.append(test["y"].to_numpy(int))
    for cname, add in CONFIGS.items():
        for m in MODELS:
            preds[(m, cname)].append(Model(m, load_params(m), base + add).fit(train).predict_proba(test))
    print(f"saison {cfg.season_label(season)} terminée", flush=True)

y = np.concatenate(ys)
rows = []
for m in MODELS:
    ref = per_match_losses(y, np.vstack(preds[(m, "complet (référence)")]))["logloss"]
    for cname in CONFIGS:
        p = np.vstack(preds[(m, cname)])
        ev = evaluate(y, p)
        d, lo, hi, pval = paired_bootstrap(per_match_losses(y, p)["logloss"], ref)
        rows.append({"modèle": m, "configuration": cname, "accuracy": ev["accuracy"], "logloss": ev["logloss"],
                     "rps": ev["rps"], "delta_logloss": d, "ci_lo": lo, "ci_hi": hi,
                     "p_bootstrap": pval if cname != "complet (référence)" else np.nan})
res = pd.DataFrame(rows)
res.to_csv(cfg.RESULTS_DIR / "sota_extension.csv", index=False)
pd.set_option("display.width", 220)
print(f"\nEffet des variables « état de l'art » ({len(y)} matchs ; delta < 0 = amélioration) :")
print(res.to_string(index=False, float_format="%.5f"))
