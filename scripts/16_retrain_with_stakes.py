"""Étape 16 : modèle de production avec les variables d'enjeu (retenues à l'étape 15).

1. Prédictions hors échantillon de la régression logistique avec enjeu, saison par saison (2017/18 -> 2025/26),
   chaque saison prédite par un modèle entraîné sur les saisons antérieures uniquement.
2. Calibration (température) apprise sur ces prédictions, comme pour le modèle d'origine.
3. Comparaison avec le modèle d'origine : walk-forward 2020/21 -> 2024/25 (critère de décision), puis
   saison 2025/26, déjà utilisée comme test final : simple contrôle, signalé comme tel.
4. Sauvegarde de models/best_pipeline.joblib (l'ancien est conservé sous best_pipeline_v1.joblib).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

import config as cfg
from src.data import load_processed
from src.elo import load_elo_params
from src.evaluation import get, oos_for_season
from src.features import select_features
from src.folds import load_features, load_fold, split
from src.metrics import evaluate, paired_bootstrap, per_match_losses
from src.models import Model, TemperatureScaler, load_params
from src.pipeline import MatchPredictor
from src.stakes import build_stakes

stakes = build_stakes(load_processed())
seasons = cfg.TUNING_SEASONS + cfg.WALK_FORWARD_SEASONS + [cfg.FINAL_TEST_SEASON]
parts = []
for season in seasons:
    raw = load_fold(season)
    train, test = split(raw.merge(stakes, on="match_id", how="left"), season)
    feats = select_features(raw.columns) + [c for c in stakes.columns if c != "match_id"]
    p = Model("logreg", load_params("logreg"), feats).fit(train).predict_proba(test)
    old = oos_for_season(season)[["match_id", "no_odds|logreg|raw|H", "no_odds|logreg|raw|D", "no_odds|logreg|raw|A"]]
    part = test[["match_id", "season", "y"]].copy()
    part[["no_odds|logreg|raw|H", "no_odds|logreg|raw|D", "no_odds|logreg|raw|A"]] = p
    parts.append(part.merge(old.rename(columns=lambda c: c.replace("no_odds", "old")), on="match_id"))
    print(f"saison {cfg.season_label(season)} : {len(test)} matchs", flush=True)
oos = pd.concat(parts, ignore_index=True)

rows = []
for label, sel in (("walk-forward 2020/21-2024/25 (décision)", cfg.WALK_FORWARD_SEASONS), (f"{cfg.season_label(cfg.FINAL_TEST_SEASON)} (contrôle)", [cfg.FINAL_TEST_SEASON])):
    ys, pn, po = [], [], []
    for s in sel:                              # calibration apprise sur les saisons antérieures, comme à l'étape 5
        prior, cur = oos[oos.season < s], oos[oos.season == s]
        yp = prior["y"].to_numpy(int)
        pn.append(TemperatureScaler().fit(get(prior, "no_odds|logreg|raw"), yp).transform(get(cur, "no_odds|logreg|raw")))
        po.append(TemperatureScaler().fit(get(prior, "old|logreg|raw"), yp).transform(get(cur, "old|logreg|raw")))
        ys.append(cur["y"].to_numpy(int))
    y, pn, po = np.concatenate(ys), np.vstack(pn), np.vstack(po)
    d, lo, hi, pv = paired_bootstrap(per_match_losses(y, pn)["logloss"], per_match_losses(y, po)["logloss"])
    rows.append({"période": label, "n": len(y), "acc_ancien": evaluate(y, po)["accuracy"], "acc_enjeu": evaluate(y, pn)["accuracy"],
                 "logloss_ancien": evaluate(y, po)["logloss"], "logloss_enjeu": evaluate(y, pn)["logloss"], "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pv})
res = pd.DataFrame(rows)
res.to_csv(cfg.RESULTS_DIR / "stakes_production.csv", index=False)
pd.set_option("display.width", 220)
print(res.to_string(index=False, float_format="%.4f"))

old_path, new_path = cfg.MODELS_DIR / "best_pipeline.joblib", cfg.MODELS_DIR / "best_pipeline_v1.joblib"
if not new_path.exists():
    shutil.copy(old_path, new_path)
predictor = MatchPredictor("no_odds|logreg|temp").fit(load_features(), oos, load_elo_params(), stakes=stakes)
joblib.dump(predictor, old_path, compress=3)
print(f"Modèle sauvegardé : {old_path.name} ({len(predictor.features)} variables, dont enjeu ; ancien conservé : {new_path.name})")
