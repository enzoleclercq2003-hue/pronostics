"""OPTIONNEL - évaluation uniquement : backtest d'une stratégie « value » à mise fixe.

AVERTISSEMENT : ceci n'est PAS un conseil de pari. La marge des bookmakers rend ce type de
stratégie très difficile ; un résultat positif sur un historique limité relève le plus souvent
du hasard. Le backtest suppose en outre qu'on obtient la cote de clôture, sans limite de mise.

Règle : on mise 1 unité sur une issue quand probabilité_modèle x cote_de_clôture - 1 > seuil.
Modèle utilisé : le modèle « sans cotes » retenu, sur ses prédictions hors échantillon.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as cfg
from src.evaluation import get

selected = json.loads((cfg.RESULTS_DIR / "selected_model.json").read_text(encoding="utf-8"))
key = selected["no_odds"]["key"]
wf = pd.read_pickle(cfg.PROCESSED_DIR / "walkforward_predictions.pkl")
final = pd.read_pickle(cfg.PROCESSED_DIR / "final_test_predictions.pkl")
rng = np.random.default_rng(cfg.SEED)


def backtest(data: pd.DataFrame, threshold: float) -> dict:
    data = data[data["close_h"].notna()]
    p = get(data, key)
    odds = data[["close_h", "close_d", "close_a"]].to_numpy(float)
    y = data["y"].to_numpy(int)
    bet = p * odds - 1 > threshold                                  # (matchs, 3)
    profit = np.where(bet, np.where(np.eye(3, dtype=bool)[y], odds - 1, -1.0), 0.0)
    pnl = profit[bet]
    if len(pnl) == 0:
        return {"paris": 0}
    boot = rng.choice(pnl, size=(cfg.N_BOOTSTRAP, len(pnl)), replace=True).mean(axis=1)
    return {"paris": int(len(pnl)), "part_matchs": bet.any(axis=1).mean(), "taux_réussite": (pnl > 0).mean(),
            "cote_moyenne": odds[bet].mean(), "profit": pnl.sum(), "roi": pnl.mean(),
            "roi_ci_lo": np.quantile(boot, 0.025), "roi_ci_hi": np.quantile(boot, 0.975)}


rows = []
for label, data in (("walk-forward 2020/21-2024/25", wf), (f"test final {cfg.season_label(cfg.FINAL_TEST_SEASON)}", final)):
    odds = data[["close_h", "close_d", "close_a"]].dropna().to_numpy(float)
    margin = (1 / odds).sum(axis=1).mean() - 1
    print(f"{label} : marge moyenne des cotes de clôture = {margin:.2%}")
    for thr in (0.0, 0.05, 0.10, 0.20):
        rows.append({"période": label, "seuil": thr, **backtest(data, thr)})
res = pd.DataFrame(rows)
res.to_csv(cfg.RESULTS_DIR / "value_backtest.csv", index=False)
pd.set_option("display.width", 220)
print(f"\nBacktest « value » (mise fixe de 1 unité, cotes de clôture, modèle {selected['no_odds']['label']}) :")
print(res.to_string(index=False, float_format="%.4f"))
print("\nRappel : évaluation uniquement, ce n'est pas un conseil de pari.")
