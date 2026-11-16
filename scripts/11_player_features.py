"""Étape 11 (expérience) : les données au niveau JOUEUR améliorent-elles le modèle ?

Compositions, valeurs marchandes, forme individuelle et fatigue (src/player_features.py), mesurées
en walk-forward (2020/21 -> 2024/25) avec les hyperparamètres existants. Le test final 2025/26
n'est pas consulté. Les données joueurs commençant en 2012/13, on compare aussi des modèles
entraînés uniquement depuis 2013/14 pour ne pas confondre l'effet des variables et celui de l'imputation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as cfg
from src.data import load_processed
from src.evaluation import get
from src.features import select_features
from src.folds import load_fold, split
from src.metrics import evaluate, paired_bootstrap, per_match_losses
from src.models import Model, load_params
from src.player_features import LINEUP_COLS, PRE_COLS, build_player_features

path = cfg.PROCESSED_DIR / "player_features.pkl"
players = pd.read_pickle(path) if path.exists() else build_player_features(load_processed())
players.to_pickle(path)
START = 2013
MODELS = ["logreg", "lgbm"]
# nom : (variables ajoutées, avec cotes, première saison d'entraînement)
CONFIGS = {
    "référence actuelle (tout l'historique)": ([], False, None),
    "référence, entraînée depuis 2013/14": ([], False, START),
    "+ joueurs avant-match (depuis 2013/14)": (PRE_COLS, False, START),
    "+ joueurs avant-match (tout l'historique)": (PRE_COLS, False, None),
    "+ joueurs avant-match + composition officielle": (PRE_COLS + LINEUP_COLS, False, START),
    "avec cotes (depuis 2013/14)": ([], True, START),
    "avec cotes + joueurs avant-match": (PRE_COLS, True, START),
    "avec cotes + joueurs + composition officielle": (PRE_COLS + LINEUP_COLS, True, START),
}
preds = {(m, c): [] for m in MODELS for c in CONFIGS}
ys, ids = [], []
for season in cfg.WALK_FORWARD_SEASONS:
    raw = load_fold(season)
    fold = raw.merge(players, on="match_id", how="left")
    train, test = split(fold, season)
    ys.append(test["y"].to_numpy(int)); ids.append(test["match_id"].to_numpy())
    for cname, (add, with_odds, start) in CONFIGS.items():
        feats = select_features(raw.columns, with_odds=with_odds) + add
        tr = train if start is None else train[train.season >= start]
        for m in MODELS:
            params = load_params(m, "odds" if with_odds else "no_odds")
            preds[(m, cname)].append(Model(m, params, feats).fit(tr).predict_proba(test))
    print(f"saison {cfg.season_label(season)} terminée", flush=True)

y, ids = np.concatenate(ys), np.concatenate(ids)
wf = pd.read_pickle(cfg.PROCESSED_DIR / "walkforward_predictions.pkl").set_index("match_id").loc[ids]
book = per_match_losses(y, get(wf, "book|shin"))["logloss"]
print(f"Bookmakers (Bet365, Shin) sur ces {len(y)} matchs : log loss {book.mean():.5f}")
rows = []
for m in MODELS:
    ref = per_match_losses(y, np.vstack(preds[(m, "référence actuelle (tout l'historique)")]))["logloss"]
    for cname in CONFIGS:
        p = np.vstack(preds[(m, cname)])
        ev, ll = evaluate(y, p), per_match_losses(y, p)["logloss"]
        d, lo, hi, pval = paired_bootstrap(ll, ref)
        db, blo, bhi, bp = paired_bootstrap(ll, book)
        rows.append({"modèle": m, "configuration": cname, "accuracy": ev["accuracy"], "logloss": ev["logloss"], "rps": ev["rps"],
                     "delta_vs_ref": d, "ci_lo": lo, "ci_hi": hi, "p": pval,
                     "delta_vs_book": db, "book_ci_lo": blo, "book_ci_hi": bhi, "p_book": bp})
        wf[f"exp|{m}|{cname}|H"], wf[f"exp|{m}|{cname}|D"], wf[f"exp|{m}|{cname}|A"] = p[:, 0], p[:, 1], p[:, 2]
res = pd.DataFrame(rows)
res.to_csv(cfg.RESULTS_DIR / "player_features_experiment.csv", index=False)
wf[[c for c in wf.columns if c.startswith("exp|")]].to_pickle(cfg.PROCESSED_DIR / "player_experiment_predictions.pkl")
pd.set_option("display.width", 250, "display.max_colwidth", 60)
print(f"\nEffet des variables joueurs ({len(y)} matchs ; delta < 0 = amélioration) :")
print(res.to_string(index=False, float_format="%.5f"))
