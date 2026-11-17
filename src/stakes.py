"""Enjeu sportif d'un match pour chaque équipe : titre, places européennes, maintien, ou plus rien à jouer.

Calculé avant la rencontre, à partir du classement établi avec les matchs strictement antérieurs
(les matchs du jour sont exclus, comme pour les autres variables).

Pour chaque objectif (titre, top 4, maintien), on mesure si l'équipe peut encore l'atteindre ou le perdre
compte tenu des points restant en jeu (3 par match restant), puis à quel point c'est serré :
    enjeu = 1 - écart / (points restant en jeu + 3)   si l'objectif est encore ouvert, 0 sinon.
L'enjeu croît donc naturellement en fin de saison, quand chaque point compte. « Plus rien à jouer »
vaut 1 quand aucun des trois objectifs n'est encore ouvert, ni dans un sens ni dans l'autre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RELEGATED = 3          # nombre de places menacées (descente directe ou barrage)
TOP = 4                # ligne « Europe » : quatrième place

BASE = ["st_title", "st_europe", "st_releg", "st_any", "st_dead"]
STAKE_COLS = [f"{side}_{c}" for c in BASE for side in ("h", "a")] + ["diff_st_any", "diff_st_dead", "season_progress"]


def _closeness(gap: float, remaining_pts: float) -> float:
    """1 quand l'écart est nul, 0 quand il n'est plus rattrapable."""
    return max(0.0, 1.0 - abs(gap) / (remaining_pts + 3.0)) if abs(gap) <= remaining_pts else 0.0


def build_stakes(df: pd.DataFrame) -> pd.DataFrame:
    """Table (match_id + variables d'enjeu). `df` : matchs nettoyés (Date, league, season, équipes, buts)."""
    df = df.sort_values(["Date", "league", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    out = np.full((n, len(BASE) * 2 + 1), np.nan)
    homes, aways = df["HomeTeam"].to_numpy(), df["AwayTeam"].to_numpy()
    hg, ag = df["FTHG"].to_numpy(dtype=float), df["FTAG"].to_numpy(dtype=float)
    dates = df["Date"].to_numpy()

    for _, idx in df.groupby(["league", "season"], sort=False).indices.items():
        teams = sorted(set(homes[idx]) | set(aways[idx]))
        n_teams = len(teams)
        total = 2 * (n_teams - 1)                                # matchs par équipe sur la saison
        pts = {t: 0.0 for t in teams}
        gd = {t: 0.0 for t in teams}
        played = {t: 0 for t in teams}
        start = 0
        while start < len(idx):
            end = start
            while end < len(idx) and dates[idx[end]] == dates[idx[start]]:
                end += 1
            day = idx[start:end]
            order = sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True)
            line = [pts[t] for t in order]
            for i in day:
                row = []
                for t in (homes[i], aways[i]):
                    rem = 3.0 * max(0, total - played[t])
                    rank = order.index(t)
                    if rank == 0:                                # leader : titre en jeu tant que le 2e peut revenir
                        title = _closeness(pts[t] - line[1], rem) if n_teams > 1 else 0.0
                    else:
                        title = _closeness(line[0] - pts[t], rem)
                    europe_line = line[TOP - 1] if rank >= TOP else line[TOP]   # place à prendre, ou à défendre
                    europe = _closeness(europe_line - pts[t], rem)
                    safe_line = line[n_teams - RELEGATED - 1] if rank >= n_teams - RELEGATED else line[n_teams - RELEGATED]
                    releg = _closeness(pts[t] - safe_line, rem)
                    anyst = max(title, europe, releg)
                    row += [title, europe, releg, anyst, float(anyst == 0.0)]
                # ordre des colonnes : h_x, a_x pour chaque variable
                h, a = row[:5], row[5:]
                out[i, :10] = [v for pair in zip(h, a) for v in pair]
                out[i, 10] = np.mean([played[homes[i]], played[aways[i]]]) / total
            for i in day:
                if np.isnan(hg[i]) or np.isnan(ag[i]):
                    continue
                for t, gf, ga in ((homes[i], hg[i], ag[i]), (aways[i], ag[i], hg[i])):
                    pts[t] += 3 if gf > ga else (1 if gf == ga else 0)
                    gd[t] += gf - ga
                    played[t] += 1
            start = end
    cols = [f"{side}_{c}" for c in BASE for side in ("h", "a")] + ["season_progress"]
    res = pd.DataFrame(out, columns=cols)
    res["diff_st_any"] = res["h_st_any"] - res["a_st_any"]
    res["diff_st_dead"] = res["h_st_dead"] - res["a_st_dead"]
    res.insert(0, "match_id", df["match_id"].to_numpy())
    return res[["match_id"] + STAKE_COLS]
