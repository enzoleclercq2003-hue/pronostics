"""Étape 14 (expérience) : indisponibilités, postes, âge et banc améliorent-ils le modèle ?

Référence : le modèle d'avant-match actuellement en production (variables de base + valeur des
effectifs, valeurs marchandes figées à juin). On ajoute chaque groupe séparément, puis tous.
Walk-forward 2020/21 -> 2024/25, entraînement depuis 2013/14, mêmes hyperparamètres.
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
from src.player_ext import ABS_COLS, EXT_COLS, MISC_COLS, POS_COLS, build_ext_features
from src.player_features import LIVE_PRE_COLS, build_player_features, load_tables

matches, tables = load_processed(), load_tables()
players = build_player_features(matches, tables, freeze_valuations=True)
ext = build_ext_features(matches, tables)
ext.to_pickle(cfg.PROCESSED_DIR / "player_ext.pkl")
x = ext.merge(matches[["match_id", "season", "FTR"]], on="match_id")
x = x[x.season.between(2013, 2025)]
print("Couverture et corrélation avec la victoire à domicile (2013-2025) :")
diag = pd.DataFrame({"couverture": x[EXT_COLS].notna().mean(), "moyenne": x[EXT_COLS].mean(),
                     "corr_victoire_dom": x[EXT_COLS].corrwith((x.FTR == "H").astype(float))})
print(diag[diag.index.str.startswith(("diff_", "mu_"))].to_string(float_format="%.3f"))

START, MODELS = 2013, ["logreg", "lgbm"]
CONFIGS = {"production (base + valeur des effectifs)": [], "+ indisponibilités (absents, suspendus)": ABS_COLS,
           "+ valeur par ligne (attaque, milieu, défense)": POS_COLS, "+ âge, banc, stabilité": MISC_COLS, "+ tout": EXT_COLS}
preds, ys = {(m, c): [] for m in MODELS for c in CONFIGS}, []
for season in cfg.WALK_FORWARD_SEASONS:
    raw = load_fold(season)
    fold = raw.merge(players, on="match_id", how="left").merge(ext, on="match_id", how="left")
    train, test = split(fold, season)
    train = train[train.season >= START]
    base = select_features(raw.columns) + LIVE_PRE_COLS
    ys.append(test["y"].to_numpy(int))
    for cname, add in CONFIGS.items():
        for m in MODELS:
            preds[(m, cname)].append(Model(m, load_params(m), base + add).fit(train).predict_proba(test))
    print(f"saison {cfg.season_label(season)} terminée", flush=True)

y = np.concatenate(ys)
rows = []
for m in MODELS:
    ref = per_match_losses(y, np.vstack(preds[(m, "production (base + valeur des effectifs)")]))["logloss"]
    for cname in CONFIGS:
        p = np.vstack(preds[(m, cname)])
        d, lo, hi, pval = paired_bootstrap(per_match_losses(y, p)["logloss"], ref)
        rows.append({"modèle": m, "configuration": cname, "accuracy": evaluate(y, p)["accuracy"], "logloss": evaluate(y, p)["logloss"],
                     "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pval})
res = pd.DataFrame(rows)
res.to_csv(cfg.RESULTS_DIR / "player_ext_experiment.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 60)
print(f"\n{len(y)} matchs (delta < 0 = amélioration par rapport à la production) :")
print(res.to_string(index=False, float_format="%.5f"))
