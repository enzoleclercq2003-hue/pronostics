"""Étape 12 (expérience) : le gain des variables joueurs survit-il aux contraintes de la production ?

En direct, on ne peut maintenir que (1) les compositions des matchs de championnat et (2) des valeurs
marchandes figées à la mi-juin précédant la saison (le jeu de données n'est plus mis à jour).
On mesure donc en walk-forward un jeu réduit « valeur marchande du onze », avec valeurs à jour puis figées.
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
from src.player_features import LIVE_LINEUP_COLS, LIVE_PRE_COLS, PRE_COLS, build_player_features, load_tables

matches, tables = load_processed(), load_tables()
fresh = build_player_features(matches, tables)
frozen = build_player_features(matches, tables, freeze_valuations=True)
frozen.to_pickle(cfg.PROCESSED_DIR / "player_features_frozen.pkl")
START = 2013
CONFIGS = {   # nom : (table, variables ajoutées)
    "référence actuelle": (None, []),
    "+ toutes variables joueurs avant-match, valeurs à jour": (fresh, PRE_COLS),
    "+ valeur du onze avant-match, valeurs à jour": (fresh, LIVE_PRE_COLS),
    "+ valeur du onze avant-match, valeurs figées en juin": (frozen, LIVE_PRE_COLS),
    "+ valeur du onze + composition officielle, valeurs figées en juin": (frozen, LIVE_PRE_COLS + LIVE_LINEUP_COLS),
}
preds, ys = {c: [] for c in CONFIGS}, []
for season in cfg.WALK_FORWARD_SEASONS:
    raw = load_fold(season)
    base = select_features(raw.columns)
    for cname, (table, add) in CONFIGS.items():
        fold = raw if table is None else raw.merge(table, on="match_id", how="left")
        train, test = split(fold, season)
        tr = train if table is None else train[train.season >= START]
        preds[cname].append(Model("logreg", load_params("logreg"), base + add).fit(tr).predict_proba(test))
    ys.append(test["y"].to_numpy(int))
y = np.concatenate(ys)
ref = per_match_losses(y, np.vstack(preds["référence actuelle"]))["logloss"]
rows = []
for cname in CONFIGS:
    p = np.vstack(preds[cname])
    d, lo, hi, pval = paired_bootstrap(per_match_losses(y, p)["logloss"], ref)
    rows.append({"configuration": cname, "accuracy": evaluate(y, p)["accuracy"], "logloss": evaluate(y, p)["logloss"],
                 "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pval})
res = pd.DataFrame(rows)
res.to_csv(cfg.RESULTS_DIR / "player_features_live_experiment.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 70)
print(f"Régression logistique, walk-forward, {len(y)} matchs (delta < 0 = amélioration) :")
print(res.to_string(index=False, float_format="%.5f"))
