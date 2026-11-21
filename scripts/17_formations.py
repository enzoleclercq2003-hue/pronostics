"""Étape 17 (expérience) : la compatibilité des systèmes de jeu améliore-t-elle le modèle ?

Walk-forward 2020/21 -> 2024/25, mêmes hyperparamètres. Les formations n'existant que depuis 2013/14,
les deux variantes comparées sont entraînées sur 2014/15 et après (une saison pour amorcer l'historique).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as cfg
from src.data import load_processed
from src.features import select_features
from src.folds import load_features, load_fold, split
from src.formations import FAMILIES, FORM_COLS, build_formation_features
from src.metrics import evaluate, paired_bootstrap, per_match_losses
from src.models import Model, load_params

matches = load_processed()
form = build_formation_features(matches, load_features())
form.to_pickle(cfg.PROCESSED_DIR / "formations.pkl")
x = form.merge(matches[["match_id", "season"]], on="match_id")
x = x[x.season.between(2014, 2025)]
print(f"Couverture 2014-2025 : {x.pair_resid.notna().mean():.1%}")
print("Familles (équipe à domicile) :", {FAMILIES[k]: f"{x[f'h_fam_{k}'].mean():.0%}" for k in range(len(FAMILIES))})
print("Écart-type de la compatibilité des affiches :", round(x.pair_resid.std(), 4), "| par équipe :", round(x.h_vs_style.std(), 4))

START, MODELS = 2014, ["logreg", "lgbm"]
SETS = {"référence (depuis 2014/15)": [], "+ familles de formation": [c for c in FORM_COLS if "fam_" in c or c in ("h_back3", "a_back3", "h_fwd", "a_fwd", "back_mismatch")],
        "+ compatibilité des affiches": ["back_mismatch", "pair_resid"], "+ réussite par équipe selon le système adverse": ["h_vs_style", "a_vs_style", "style_diff"],
        "+ tout": FORM_COLS}
preds, ys = {(m, c): [] for m in MODELS for c in SETS}, []
for season in cfg.WALK_FORWARD_SEASONS:
    raw = load_fold(season)
    fold = raw.merge(form, on="match_id", how="left")
    train, test = split(fold, season)
    train = train[train.season >= START]
    base = select_features(raw.columns)
    ys.append(test["y"].to_numpy(int))
    for cname, add in SETS.items():
        for m in MODELS:
            preds[(m, cname)].append(Model(m, load_params(m), base + add).fit(train).predict_proba(test))
    print(f"saison {cfg.season_label(season)} terminée", flush=True)

y = np.concatenate(ys)
rows = []
for m in MODELS:
    ref = per_match_losses(y, np.vstack(preds[(m, "référence (depuis 2014/15)")]))["logloss"]
    for cname in SETS:
        p = np.vstack(preds[(m, cname)])
        d, lo, hi, pv = paired_bootstrap(per_match_losses(y, p)["logloss"], ref)
        rows.append({"modèle": m, "variables": cname, "accuracy": evaluate(y, p)["accuracy"], "logloss": evaluate(y, p)["logloss"],
                     "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pv})
res = pd.DataFrame(rows)
res.to_csv(cfg.RESULTS_DIR / "formations_experiment.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 60)
print(f"\n{len(y)} matchs (delta < 0 = amélioration) :")
print(res.to_string(index=False, float_format="%.5f"))
