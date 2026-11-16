"""Étape 13 : contrôle sur des données jamais utilisées, les matchs déjà joués de 2026/27.

Vérifie la chaîne de production complète (compositions ESPN + valeurs Transfermarkt figées à juin
2026) : modèles entraînés sur les saisons <= 2025/26, évalués sur la saison en cours. L'échantillon
est petit (environ 500 matchs) : les intervalles sont larges, c'est un contrôle de cohérence.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

import config as cfg
from src.data import load_processed
from src.metrics import evaluate, paired_bootstrap, per_match_losses
from src.models import Model, load_params
from src.odds import implied_shin
from src.pipeline_players import fit_player_models, player_fold, predict_levels

base = joblib.load(cfg.MODELS_DIR / "best_pipeline.joblib")
fold, _, report = player_fold(base, None, load_processed(), verbose=True)
print(f"Compositions ESPN : {report['events']} matchs sur {report['matches_current']} ; "
      f"titulaires identifiés : {report['matched_share']:.1%} ({report['how']})")
played = fold["y"].notna().to_numpy()
train_mask = played & (fold.season < cfg.CURRENT_SEASON).to_numpy()
test = fold[played & (fold.season == cfg.CURRENT_SEASON).to_numpy()].reset_index(drop=True)
y = test["y"].to_numpy(int)

ref = Model("logreg", load_params("logreg"), base.features).fit(fold[train_mask & (fold.season > cfg.FIRST_SEASON)])
models = fit_player_models(base, fold, train_mask)


class _Ref:                                   # modèle de base réentraîné sans la saison en cours
    predict_from_features = staticmethod(ref.predict_proba)


p_ref = ref.predict_proba(test)
p_live, level = predict_levels(_Ref, models, test)
p_pre = p_ref.copy()
has_pre = test[["diff_pxi_logmv", "diff_top11_logmv"]].notna().all(axis=1).to_numpy()
p_pre[has_pre] = models["effectifs"].predict_proba(test[has_pre])
book = implied_shin(test[["odds_h", "odds_d", "odds_a"]].to_numpy(float))
print(f"\n{len(test)} matchs de {cfg.season_label(cfg.CURRENT_SEASON)} ; niveaux utilisés : {pd.Series(level).value_counts().to_dict()}")
rows = []
l_ref = per_match_losses(y, p_ref)["logloss"]
for name, p in (("modèle de base (sans joueurs)", p_ref), ("+ valeur des effectifs (avant-match)", p_pre),
                ("+ composition officielle", p_live), ("bookmakers (Bet365, Shin)", book)):
    ev, ll = evaluate(y, p), per_match_losses(y, p)["logloss"]
    d, lo, hi, pval = paired_bootstrap(ll, l_ref)
    rows.append({"prédiction": name, "accuracy": ev["accuracy"], "logloss": ev["logloss"], "rps": ev["rps"],
                 "delta_vs_base": d, "ci_lo": lo, "ci_hi": hi, "p": pval})
res = pd.DataFrame(rows)
res.to_csv(cfg.RESULTS_DIR / "live_check_2026.csv", index=False)
pd.set_option("display.width", 220)
print(res.to_string(index=False, float_format="%.4f"))
