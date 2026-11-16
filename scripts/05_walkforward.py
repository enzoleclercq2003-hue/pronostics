"""Étape 5 : validation walk-forward (2020/21 -> 2024/25), calibration, ensembles, sélection.

Le test final (2025/26) n'est PAS consulté ici : la sélection du meilleur modèle se fait
uniquement sur le walk-forward.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as cfg
from src.evaluation import breakdown, comparison_table, oos_for_season, post_process

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 45)

# 1. Prédictions hors échantillon : les saisons 2017-2019 servent d'amorce (calibration / ensembles)
seasons = cfg.TUNING_SEASONS + cfg.WALK_FORWARD_SEASONS
t0 = time.time()
oos = pd.concat([oos_for_season(s) for s in seasons], ignore_index=True)
print(f"Prédictions hors échantillon : {oos.shape}, {oos.Date.min().date()} -> {oos.Date.max().date()} ({time.time() - t0:.0f} s)")

# 2. Calibration + ensembles appris sur les saisons antérieures à chaque saison de test
pred = post_process(oos, cfg.WALK_FORWARD_SEASONS)
pred.to_pickle(cfg.PROCESSED_DIR / "walkforward_predictions.pkl")
print(f"Walk-forward : {pred.shape}, saisons {sorted(pred.season.unique())}")

# 3. Tableau comparatif (mêmes matchs pour tous)
table = comparison_table(pred)
table.to_csv(cfg.RESULTS_DIR / "model_comparison_walkforward.csv", index=False)
show = table[["modèle", "famille", "accuracy", "logloss", "brier", "rps", "f1_D", "logloss_season_std",
              "dlogloss_vs_book", "p_bootstrap", "p_dm", "n"]].sort_values("logloss")
print("\nComparaison walk-forward (5 saisons regroupées, triée par log loss) :")
print(show.to_string(index=False, float_format="%.4f"))

# 4. Effet de la calibration (log loss brut / température / isotonique)
from src.evaluation import VARIANTS, get
from src.metrics import evaluate
from src.models import MODEL_NAMES
y = pred["y"].to_numpy(int)
cal = pd.DataFrame([{"variante": v, "modèle": m,
                     **{k: evaluate(y, get(pred, f"{v}|{m}|{k}"))["logloss"] for k in ("raw", "temp", "iso")}}
                    for v in VARIANTS for m in MODEL_NAMES])
cal.to_csv(cfg.RESULTS_DIR / "calibration_effect.csv", index=False)
print("\nEffet de la calibration sur la log loss :")
print(cal.to_string(index=False, float_format="%.5f"))

# 5. Sélection du meilleur modèle de chaque variante (log loss walk-forward)
selected = {}
for variant, fam in VARIANTS.items():
    best = table[table.famille == fam].sort_values("logloss").iloc[0]
    selected[variant] = {"key": best["clé"], "label": best["modèle"], "logloss": best["logloss"], "accuracy": best["accuracy"]}
(cfg.RESULTS_DIR / "selected_model.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nModèles retenus :", json.dumps(selected, ensure_ascii=False))

# 6. Détail par saison et par championnat
keys = [selected["no_odds"]["key"], selected["odds"]["key"], "book|shin", "book|closing_shin", "base|elo"]
for by in ("season", "league"):
    b = breakdown(pred, keys, by)
    b.to_csv(cfg.RESULTS_DIR / f"walkforward_by_{by}.csv", index=False)
    print(f"\nLog loss par {by} :")
    print(b.pivot(index=by, columns="clé", values="logloss")[keys].to_string(float_format="%.4f"))
    print(f"Précision par {by} :")
    print(b.pivot(index=by, columns="clé", values="accuracy")[keys].to_string(float_format="%.4f"))

weights = pred.groupby("season")[[c for c in pred.columns if c.startswith("w|no_odds")]].first()
print("\nPoids de la moyenne pondérée (sans cotes) par saison :")
print(weights.rename(columns=lambda c: c.split("|")[-1]).to_string(float_format="%.2f"))
