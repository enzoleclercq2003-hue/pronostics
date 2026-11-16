"""Elo dynamique par championnat (notes d'avant-match uniquement)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as cfg


def load_elo_params() -> dict:
    """Paramètres optimisés par validation s'ils existent, sinon valeurs par défaut."""
    params = {"k": cfg.ELO_K, "hfa": cfg.ELO_HFA, "carry": cfg.ELO_SEASON_CARRY,
              "promoted_offset": cfg.ELO_PROMOTED_OFFSET}
    if cfg.ELO_PARAMS_FILE.exists():
        params.update(json.loads(cfg.ELO_PARAMS_FILE.read_text()))
    return params


def _margin_multiplier(goal_diff: float) -> float:
    """Pondération par la marge de buts (barème World Football Elo)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def compute_elo(df: pd.DataFrame, k: float, hfa: float, carry: float, promoted_offset: float) -> pd.DataFrame:
    """Parcourt les matchs dans l'ordre chronologique et renvoie les notes AVANT chaque match.

    `df` doit être trié par date. Les lignes sans résultat (matchs à venir) reçoivent leurs
    notes d'avant-match mais ne déclenchent aucune mise à jour.
    """
    rating: dict[tuple[str, str], float] = {}
    last_season: dict[tuple[str, str], int] = {}
    n = len(df)
    elo_h, elo_a = np.empty(n), np.empty(n)
    leagues, seasons = df["league"].to_numpy(), df["season"].to_numpy()
    homes, aways = df["HomeTeam"].to_numpy(), df["AwayTeam"].to_numpy()
    hg, ag = df["FTHG"].to_numpy(dtype=float), df["FTAG"].to_numpy(dtype=float)
    first_season = int(seasons.min()) if n else cfg.FIRST_SEASON
    entry = cfg.ELO_INIT + promoted_offset

    def pre_match(key, season):
        if key not in rating:
            # Nouvelle équipe : 1500 la toute première saison, sinon note d'entrée (promu)
            r = cfg.ELO_INIT if season == first_season else entry
        else:
            r = rating[key]
            gap = season - last_season[key]
            if gap >= 1:  # nouvelle saison : retour partiel vers la moyenne
                r = cfg.ELO_INIT + carry * (r - cfg.ELO_INIT)
            if gap >= 2:  # retour après relégation : rapprochement de la note d'entrée
                r = 0.5 * r + 0.5 * entry
        rating[key] = r
        last_season[key] = season
        return r

    for i in range(n):
        kh, ka = (leagues[i], homes[i]), (leagues[i], aways[i])
        rh, ra = pre_match(kh, seasons[i]), pre_match(ka, seasons[i])
        elo_h[i], elo_a[i] = rh, ra
        if np.isnan(hg[i]) or np.isnan(ag[i]):
            continue
        expected = 1.0 / (1.0 + 10 ** (-(rh + hfa - ra) / 400.0))
        score = 1.0 if hg[i] > ag[i] else (0.5 if hg[i] == ag[i] else 0.0)
        delta = k * _margin_multiplier(hg[i] - ag[i]) * (score - expected)
        rating[kh] = rh + delta
        rating[ka] = ra - delta

    out = pd.DataFrame({"elo_h": elo_h, "elo_a": elo_a}, index=df.index)
    out["elo_diff"] = out["elo_h"] - out["elo_a"]
    out["elo_exp_h"] = 1.0 / (1.0 + 10 ** (-(out["elo_diff"] + hfa) / 400.0))
    return out
