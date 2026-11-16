"""Étape 6 : ablations en walk-forward (sans/avec clustering, Elo, cotes ; K-means vs GMM vs Ward).

Mêmes hyperparamètres, mêmes matchs ; seule la liste de variables change. Différences testées
par bootstrap apparié sur la log loss par match.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as cfg
from src.features import select_features
from src.folds import load_fold, split
from src.metrics import evaluate, paired_bootstrap, per_match_losses
from src.models import Model, load_params

MODELS = ["logreg", "lgbm"]
CONFIGS = {
    # nom : (méthode de clustering, options de sélection des variables)
    "complet (K-means)": ("kmeans", dict()),
    "sans clustering": ("kmeans", dict(with_cluster=False)),
    "clustering GMM": ("gmm", dict()),
    "clustering hiérarchique (Ward)": ("ward", dict()),
    "sans Elo": ("kmeans", dict(with_elo=False)),
    "sans Elo ni clustering": ("kmeans", dict(with_elo=False, with_cluster=False)),
    "avec cotes": ("kmeans", dict(with_odds=True)),
    "avec cotes, sans clustering": ("kmeans", dict(with_odds=True, with_cluster=False)),
}

preds = {(m, c): [] for m in MODELS for c in CONFIGS}
ys = []
for season in cfg.WALK_FORWARD_SEASONS:
    folds = {method: load_fold(season, method) for method in ("kmeans", "gmm", "ward")}
    ys.append(split(folds["kmeans"], season)[1]["y"].to_numpy(int))
    for cname, (method, opts) in CONFIGS.items():
        train, test = split(folds[method], season)
        feats = select_features(folds[method].columns, **opts)
        for m in MODELS:
            variant = "odds" if opts.get("with_odds") else "no_odds"
            preds[(m, cname)].append(Model(m, load_params(m, variant), feats).fit(train).predict_proba(test))
    print(f"saison {cfg.season_label(season)} terminée", flush=True)

y = np.concatenate(ys)
rows = []
for m in MODELS:
    ref = per_match_losses(y, np.vstack(preds[(m, "complet (K-means)")]))["logloss"]
    for cname in CONFIGS:
        p = np.vstack(preds[(m, cname)])
        ev = evaluate(y, p)
        d, lo, hi, pval = paired_bootstrap(per_match_losses(y, p)["logloss"], ref)
        rows.append({"modèle": m, "configuration": cname, "accuracy": ev["accuracy"], "logloss": ev["logloss"],
                     "brier": ev["brier"], "rps": ev["rps"], "delta_logloss_vs_complet": d,
                     "ci_lo": lo, "ci_hi": hi, "p_bootstrap": pval if cname != "complet (K-means)" else np.nan})
abl = pd.DataFrame(rows)
abl.to_csv(cfg.RESULTS_DIR / "ablations.csv", index=False)
pd.set_option("display.width", 220)
print(f"\nAblations walk-forward ({len(y)} matchs, log loss brute ; delta < 0 = meilleur que le modèle complet) :")
print(abl.to_string(index=False, float_format="%.5f"))
