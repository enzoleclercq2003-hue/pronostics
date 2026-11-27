"""Décision affichée pour un match : pronostic (au-dessus du seuil) et risque de contre-performance.

- `pick(p)` : indice de l'issue pronostiquée (0 domicile, 1 nul, 2 extérieur) si sa probabilité atteint
  cfg.PICK_THRESHOLD, sinon None (match ouvert).
- `upset_risk(p, book)` : probabilité que le favori du modèle ne gagne pas, d'après le modèle de
  contre-performance (scripts/18_threshold_upsets.py : probabilité du favori selon le modèle et selon
  les bookmakers, écart entre les deux, favori à domicile). Sans cotes publiées, pas d'estimation.
  L'alerte est levée pour les 20 % de pronostics jugés les plus fragiles sur l'historique.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import config as cfg

UPSET_PATH = cfg.MODELS_DIR / "upset_model.joblib"


def pick(p) -> int | None:
    k = int(np.argmax(p))
    return k if p[k] >= cfg.PICK_THRESHOLD else None


@lru_cache(maxsize=1)
def _upset_model():
    if not UPSET_PATH.exists():
        return None
    import joblib
    return joblib.load(UPSET_PATH)


def _logit(x: float) -> float:
    x = min(max(x, 1e-4), 1 - 1e-4)
    return float(np.log(x / (1 - x)))


def upset_risk(p, book) -> tuple[float | None, bool]:
    """(risque, alerte) pour le favori du modèle (équipe à domicile ou à l'extérieur)."""
    bundle = _upset_model()
    if bundle is None or book is None or any(b is None or np.isnan(b) for b in book):
        return None, False
    home = p[0] >= p[2]
    pf, bf = (p[0], book[0]) if home else (p[2], book[2])
    row = {"logit_fav": _logit(pf), "logit_book": _logit(bf), "gap_market": pf - bf, "fav_home": float(home)}
    import pandas as pd
    risk = float(bundle["model"].predict_proba(pd.DataFrame([[row[c] for c in bundle["features"]]], columns=bundle["features"]))[0, 1])
    return risk, risk >= bundle["alert"]


def upset_info() -> dict | None:
    """Repères affichés avec l'alerte : seuil d'alerte, part de contre-performances (tous pronostics / alertes)."""
    b = _upset_model()
    return None if b is None else {"alert": b["alert"], "mean": b.get("mean"), "alert_rate": b.get("alert_rate")}
