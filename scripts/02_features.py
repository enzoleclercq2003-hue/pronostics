"""Étape 2 : optimisation d'Elo (K, avantage du terrain) par validation, puis variables."""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import config as cfg
from src.data import load_processed
from src.elo import compute_elo
from src.features import build_features, feature_groups

pd.set_option("display.width", 220, "display.max_columns", 40)
df = load_processed()

# --- 1. Optimisation d'Elo : grille évaluée par log loss sur les saisons de validation ----
# Entraînement de la régression (diff. Elo -> issue) sur 2007..2016, validation sur 2017..2019.
# Les saisons de test walk-forward (>= 2020) ne sont jamais consultées ici.
y = df["FTR"].map({c: i for i, c in enumerate(cfg.CLASSES)}).to_numpy()
tr = ((df.season >= cfg.FIRST_SEASON + 2) & (df.season < min(cfg.TUNING_SEASONS))).to_numpy()
va = df.season.isin(cfg.TUNING_SEASONS).to_numpy()
rows = []
t0 = time.time()
for k, hfa, carry in itertools.product([4, 6, 8, 10, 12, 14, 18, 24, 30], [0, 30, 60, 90], [0.8, 0.9, 1.0]):
    elo = compute_elo(df, k=k, hfa=hfa, carry=carry, promoted_offset=cfg.ELO_PROMOTED_OFFSET)
    x = elo[["elo_diff"]].to_numpy() / 100.0
    lr = LogisticRegression(C=1e3, max_iter=500).fit(x[tr], y[tr])
    rows.append((k, hfa, carry, log_loss(y[va], lr.predict_proba(x[va]), labels=[0, 1, 2])))
grid = pd.DataFrame(rows, columns=["k", "hfa", "carry", "val_logloss"]).sort_values("val_logloss")
grid.to_csv(cfg.RESULTS_DIR / "elo_grid.csv", index=False)
best = grid.iloc[0]
params = {"k": float(best.k), "hfa": float(best.hfa), "carry": float(best.carry),
          "promoted_offset": cfg.ELO_PROMOTED_OFFSET}
cfg.ELO_PARAMS_FILE.write_text(json.dumps(params, indent=2))
print(f"Grille Elo ({len(grid)} combinaisons, {time.time() - t0:.0f} s) - 5 meilleures :")
print(grid.head().to_string(index=False))
print(f"Pire combinaison : {grid.val_logloss.max():.4f}  |  paramètres retenus : {params}")

# --- 2. Construction des variables ------------------------------------------------------
t0 = time.time()
feats = build_features(df, params)
feats.to_pickle(cfg.PROCESSED_DIR / "features.pkl")
groups = feature_groups(feats.columns)
print(f"\nVariables construites en {time.time() - t0:.0f} s")
print(f"Dimensions : {feats.shape}  |  plage : {feats.Date.min().date()} -> {feats.Date.max().date()}")
print("Nombre de variables par groupe :", {k: len(v) for k, v in groups.items()})
model_cols = sum(groups.values(), [])
na = feats[model_cols].isna().mean().sort_values(ascending=False)
print("\nVariables avec le plus de valeurs manquantes :")
print(na.head(8).round(4).to_string())
print("\n5 premières lignes (extrait) :")
show = ["Date", "league", "HomeTeam", "AwayTeam", "FTR", "elo_diff", "h_ewm_pts", "a_ewm_pts", "dc_p_h", "dc_p_d", "dc_p_a", "rank_diff"]
print(feats[show].head().to_string(float_format="%.3f"))
print("\n5 dernières lignes (extrait) :")
print(feats[show].tail().to_string(float_format="%.3f"))

# --- 3. Contrôle de bon sens : corrélation des variables avec la victoire à domicile ------
home_win = (feats.y == 0).astype(float)
corr = feats[model_cols].corrwith(home_win).abs().sort_values(ascending=False)
print("\nCorrélations |r| les plus fortes avec la victoire à domicile (une valeur > 0.5 signalerait une fuite) :")
print(corr.head(8).round(3).to_string())
