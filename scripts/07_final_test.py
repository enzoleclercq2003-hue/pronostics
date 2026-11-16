"""Étape 7 : test final sur la dernière saison complète (2025/26), puis modèle de production.

Le test final n'intervient dans AUCUNE décision : hyperparamètres (Optuna sur 2017-2019),
choix du modèle (walk-forward 2020-2024), calibration et ensembles (prédictions hors
échantillon antérieures) sont tous figés avant. Les prédictions sont calculées une seule fois
puis mises en cache ; relancer ce script ne fait que régénérer les tableaux.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

import config as cfg
from src.elo import load_elo_params
from src.evaluation import breakdown, comparison_table, get, oos_for_season, post_process
from src.folds import load_features
from src.metrics import confusion, evaluate
from src.pipeline import MatchPredictor

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 45)
selected = json.loads((cfg.RESULTS_DIR / "selected_model.json").read_text(encoding="utf-8"))
S = cfg.FINAL_TEST_SEASON

# 1. Prédictions du test final (modèles entraînés sur les saisons <= 2024/25)
seasons = cfg.TUNING_SEASONS + cfg.WALK_FORWARD_SEASONS + [S]
oos = pd.concat([oos_for_season(s) for s in seasons], ignore_index=True)
final = post_process(oos, [S])
final.to_pickle(cfg.PROCESSED_DIR / "final_test_predictions.pkl")
print(f"Test final {cfg.season_label(S)} : {final.shape}, {final.Date.min().date()} -> {final.Date.max().date()}")

# 2. Tableau comparatif
table = comparison_table(final)
table.to_csv(cfg.RESULTS_DIR / "model_comparison_final_test.csv", index=False)
print("\nComparaison sur le test final (triée par log loss) :")
print(table[["modèle", "famille", "accuracy", "logloss", "brier", "rps", "f1_H", "f1_D", "f1_A",
             "dlogloss_vs_book", "p_bootstrap", "p_dm", "n"]].sort_values("logloss").to_string(index=False, float_format="%.4f"))

# 3. Modèles retenus (choisis AVANT, sur le walk-forward)
y = final["y"].to_numpy(int)
for variant, sel in selected.items():
    row = table[table["clé"] == sel["key"]].iloc[0]
    print(f"\nModèle retenu [{variant}] : {sel['label']}")
    print(f"  précision {row.accuracy:.4f} [IC95 {row.accuracy_ci_lo:.4f}-{row.accuracy_ci_hi:.4f}] | "
          f"log loss {row.logloss:.4f} [{row.logloss_ci_lo:.4f}-{row.logloss_ci_hi:.4f}] | "
          f"Brier {row.brier:.4f} | RPS {row.rps:.4f}")
    print(f"  écart de log loss vs bookmakers (Shin) : {row.dlogloss_vs_book:+.4f} "
          f"[IC95 {row.dlogloss_ci_lo:+.4f} ; {row.dlogloss_ci_hi:+.4f}], p bootstrap = {row.p_bootstrap:.3f}, p DM = {row.p_dm:.3f}")
    print("  matrice de confusion (lignes = réel H/D/A, colonnes = prédit H/D/A) :")
    print(confusion(y, get(final, sel["key"])))

keys = [selected["no_odds"]["key"], selected["odds"]["key"], "book|shin", "book|closing_shin", "base|elo"]
b = breakdown(final, keys, "league")
b.to_csv(cfg.RESULTS_DIR / "final_test_by_league.csv", index=False)
print("\nTest final par championnat - log loss :")
print(b.pivot(index="league", columns="clé", values="logloss")[keys].to_string(float_format="%.4f"))
print("Test final par championnat - précision :")
print(b.pivot(index="league", columns="clé", values="accuracy")[keys].to_string(float_format="%.4f"))

# 4. Modèle de production : réentraîné sur tout l'historique disponible, mêmes hyperparamètres ;
#    calibration / combinaison apprise sur toutes les prédictions hors échantillon (2017 -> 2025).
feats = load_features()
for variant, fname in (("no_odds", "best_pipeline.joblib"), ("odds", "best_pipeline_with_odds.joblib")):
    predictor = MatchPredictor(selected[variant]["key"]).fit(feats, oos, load_elo_params())
    joblib.dump(predictor, cfg.MODELS_DIR / fname, compress=3)
    print(f"\nPipeline sauvegardé : models/{fname} ({selected[variant]['label']}, {predictor.n_train} matchs "
          f"d'entraînement jusqu'au {predictor.train_end.date()})")
