"""Étape 15 (expérience) : l'enjeu sportif (titre, Europe, maintien, plus rien à jouer) améliore-t-il le modèle ?

Walk-forward 2020/21 -> 2024/25, mêmes hyperparamètres, mêmes matchs. On regarde l'ensemble des
matchs, puis le dernier quart de saison, où l'enjeu est le plus marqué. Le test final n'est pas consulté.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as cfg
from src.data import load_processed
from src.features import select_features
from src.folds import load_fold, split
from src.metrics import evaluate, paired_bootstrap, per_match_losses
from src.models import Model, load_params
from src.stakes import STAKE_COLS, build_stakes

stakes = build_stakes(load_processed())
stakes.to_pickle(cfg.PROCESSED_DIR / "stakes.pkl")
x = stakes.merge(load_processed()[["match_id", "season", "FTR"]], on="match_id")
x = x[x.season.between(2010, 2025)]
late = x.season_progress >= .75
print("Part des matchs où une équipe n'a plus rien à jouer (dernier quart de saison) :",
      f"{((x.h_st_dead == 1) | (x.a_st_dead == 1))[late].mean():.1%}")
y = (x.FTR == "H").astype(float)
print("Corrélation avec la victoire à domicile (dernier quart) :")
print(x.loc[late, ["diff_st_any", "diff_st_dead", "h_st_releg", "a_st_releg", "h_st_dead", "a_st_dead"]].corrwith(y[late]).round(3).to_string())

MODELS = ["logreg", "lgbm"]
preds, ys, prog = {(m, c): [] for m in MODELS for c in ("ref", "stakes")}, [], []
for season in cfg.WALK_FORWARD_SEASONS:
    raw = load_fold(season)
    fold = raw.merge(stakes, on="match_id", how="left")
    train, test = split(fold, season)
    base = select_features(raw.columns)
    ys.append(test["y"].to_numpy(int)); prog.append(test["season_progress"].to_numpy())
    for m in MODELS:
        preds[(m, "ref")].append(Model(m, load_params(m), base).fit(train).predict_proba(test))
        preds[(m, "stakes")].append(Model(m, load_params(m), base + STAKE_COLS).fit(train).predict_proba(test))
    print(f"saison {cfg.season_label(season)} terminée", flush=True)

y, prog = np.concatenate(ys), np.concatenate(prog)
rows = []
for m in MODELS:
    for subset, mask in (("tous les matchs", np.ones_like(y, bool)), ("dernier quart de saison", prog >= .75)):
        ref, new = np.vstack(preds[(m, "ref")])[mask], np.vstack(preds[(m, "stakes")])[mask]
        d, lo, hi, pval = paired_bootstrap(per_match_losses(y[mask], new)["logloss"], per_match_losses(y[mask], ref)["logloss"])
        rows.append({"modèle": m, "matchs": subset, "n": int(mask.sum()), "logloss_ref": evaluate(y[mask], ref)["logloss"],
                     "logloss_enjeu": evaluate(y[mask], new)["logloss"], "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pval})
res = pd.DataFrame(rows)
res.to_csv(cfg.RESULTS_DIR / "stakes_experiment.csv", index=False)
pd.set_option("display.width", 220)
print("\nEffet des variables d'enjeu (delta < 0 = amélioration) :")
print(res.to_string(index=False, float_format="%.5f"))
