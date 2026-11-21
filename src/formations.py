"""Compatibilité des systèmes de jeu (expérience, cf. scripts/17).

Avant un match, on ne connaît pas encore les formations : on prend celle que chaque équipe a utilisée
lors de son dernier match de championnat. Variables :
- famille de formation de chaque équipe (défense à trois ou à quatre, nombre d'attaquants) ;
- « compatibilité » d'une affiche de familles : résultat moyen au-delà de ce que prévoyait l'Elo
  (score réel - score attendu) dans tous les matchs antérieurs opposant ces deux familles, lissé vers 0 ;
- même mesure par équipe : comment l'équipe réussit, au-delà de son niveau, contre une défense à
  trois ou à quatre (deux dernières saisons).
Toutes les moyennes n'utilisent que des matchs strictement antérieurs (les matchs du jour exclus).
Source des formations : jeu de données Transfermarkt (2013/14 -> 2025/26), usage non commercial.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.player_features import TM_DIR, map_games

SMOOTH_PAIR, SMOOTH_TEAM = 150.0, 25.0          # nombre de matchs « fictifs » à résultat neutre (lissage)

FAMILIES = ["4-2-3-1", "4-3-3", "4-4-2", "4-1-4-1", "4-diamant", "3-arrière-2-att", "3-arrière-3-att", "5-arrière"]
FORM_COLS = (["h_back3", "a_back3", "h_fwd", "a_fwd", "back_mismatch", "pair_resid", "h_vs_style", "a_vs_style", "style_diff"]
             + [f"h_fam_{i}" for i in range(len(FAMILIES))] + [f"a_fam_{i}" for i in range(len(FAMILIES))])


def family(f: str | float) -> int | None:
    """Formation (« 4-2-3-1 », « 3-5-2 flat », « 4-4-2 double 6 »...) -> indice de famille."""
    if not isinstance(f, str):
        return None
    d = [int(x) for x in re.findall(r"\d", f)]
    if len(d) < 3:
        return None
    if d[0] >= 5:
        return 7
    if d[0] == 3:
        return 6 if d[-1] >= 3 or (len(d) == 4 and d[-2] + d[-1] >= 3) else 5
    if "diamond" in f.lower() or d[:4] == [4, 3, 1, 2]:
        return 4
    if d[:3] == [4, 2, 3] or d[:4] == [4, 4, 1, 1]:
        return 0
    if d[:2] == [4, 3]:
        return 1
    if d[:2] == [4, 1]:
        return 3
    if d[:2] == [4, 4]:
        return 2
    return 0


def _forwards(f: str | float) -> float:
    d = [int(x) for x in re.findall(r"\d", f)] if isinstance(f, str) else []
    return float(d[-1]) if len(d) >= 3 else np.nan


def build_formation_features(matches: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """`matches` : matchs nettoyés ; `feats` : table de variables (pour l'espérance Elo `elo_exp_h`)."""
    games = pd.read_csv(TM_DIR / "games.csv.gz", usecols=["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id",
                                                           "home_club_goals", "away_club_goals", "home_club_formation", "away_club_formation"])
    games["day"] = (pd.to_datetime(games["date"]) - pd.Timestamp("1970-01-01")).dt.days
    link = map_games(matches, games)
    g = link.merge(games[["game_id", "home_club_formation", "away_club_formation"]], on="game_id")
    m = matches[["match_id", "Date", "league", "season", "HomeTeam", "AwayTeam", "FTR"]].merge(
        g[["match_id", "home_club_formation", "away_club_formation"]], on="match_id", how="left")
    m = m.merge(feats[["match_id", "elo_exp_h"]], on="match_id", how="left")
    m = m.sort_values(["Date", "league", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    score = m["FTR"].map({"H": 1.0, "D": .5, "A": 0.0}).to_numpy()
    resid = score - m["elo_exp_h"].to_numpy()                    # résultat au-delà de ce que prévoyait l'Elo
    fh, fa = m["home_club_formation"].map(family).to_numpy(), m["away_club_formation"].map(family).to_numpy()
    nh, na = m["home_club_formation"].map(_forwards).to_numpy(), m["away_club_formation"].map(_forwards).to_numpy()

    n = len(m)
    out = np.full((n, len(FORM_COLS)), np.nan)
    last = {}                                                    # équipe -> (famille, attaquants) au dernier match
    pair_sum, pair_n = np.zeros((8, 8)), np.zeros((8, 8))
    team_hist = {}                                               # (équipe, défense adverse à 3 ?) -> liste (jour, résidu)
    days = (m["Date"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    dates = m["Date"].to_numpy()
    homes, aways = m["HomeTeam"].to_numpy(), m["AwayTeam"].to_numpy()
    start = 0
    while start < n:
        end = start
        while end < n and dates[end] == dates[start]:
            end += 1
        for i in range(start, end):                              # variables : uniquement l'état d'avant ce jour
            lh, la = last.get(homes[i]), last.get(aways[i])
            if lh is None or la is None:
                continue
            (h_f, h_n), (a_f, a_n) = lh, la
            h3, a3 = float(h_f >= 5), float(a_f >= 5)
            c = pair_n[h_f, a_f]
            pr = pair_sum[h_f, a_f] / (c + SMOOTH_PAIR)
            def vs(team, opp3, sign):
                hist = [r for d, r in team_hist.get((team, opp3), []) if days[i] - d <= 730]
                return sign * sum(hist) / (len(hist) + SMOOTH_TEAM) if hist else 0.0
            hv, av = vs(homes[i], bool(a3), 1.0), vs(aways[i], bool(h3), 1.0)
            row = [h3, a3, h_n, a_n, float(h3 != a3), pr, hv, av, hv - av]
            row += [float(h_f == k) for k in range(8)] + [float(a_f == k) for k in range(8)]
            out[i] = row
        for i in range(start, end):                              # mise à jour avec les résultats du jour
            if np.isnan(fh[i]) if isinstance(fh[i], float) else fh[i] is None:
                continue
            if fh[i] is None or fa[i] is None or np.isnan(resid[i]):
                if fh[i] is not None and fa[i] is not None:
                    last[homes[i]], last[aways[i]] = (int(fh[i]), nh[i]), (int(fa[i]), na[i])
                continue
            h_f, a_f = int(fh[i]), int(fa[i])
            pair_sum[h_f, a_f] += resid[i]; pair_n[h_f, a_f] += 1
            team_hist.setdefault((homes[i], a_f >= 5), []).append((days[i], resid[i]))
            team_hist.setdefault((aways[i], h_f >= 5), []).append((days[i], -resid[i]))
            last[homes[i]], last[aways[i]] = (h_f, nh[i]), (a_f, na[i])
        start = end
    res = pd.DataFrame(out, columns=FORM_COLS)
    res.insert(0, "match_id", m["match_id"].to_numpy())
    return res
