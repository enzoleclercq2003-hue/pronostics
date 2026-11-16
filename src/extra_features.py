"""Variables supplémentaires inspirées de l'état de l'art (expérience, cf. scripts/10).

- pi-ratings (Constantinou & Fenton, 2013) : notes domicile / extérieur séparées, mises à jour
  selon l'écart entre différence de buts observée et attendue. Variables du vainqueur du
  Soccer Prediction Challenge 2017 (Hubáček et al.).
- forces d'attaque / défense estimées sur les TIRS CADRÉS (esprit des ratings GAP de
  Wheatcroft, 2020) : approximation gratuite des expected goals, moins bruitée que les buts.

Comme toutes les autres variables : uniquement des matchs strictement antérieurs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.poisson import MIN_MATCHES, fit_poisson_strengths

PI_LAMBDA, PI_GAMMA, PI_B, PI_C = 0.035, 0.7, 10.0, 3.0     # valeurs de l'article d'origine


def _expected_gd(rating: float) -> float:
    return np.sign(rating) * (PI_B ** (abs(rating) / PI_C) - 1.0)


def pi_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Notes pi d'avant-match. `df` trié par date ; les lignes sans résultat ne mettent rien à jour."""
    home_r: dict[tuple, float] = {}
    away_r: dict[tuple, float] = {}
    n = len(df)
    out = np.zeros((n, 4))
    leagues, homes, aways = df["league"].to_numpy(), df["HomeTeam"].to_numpy(), df["AwayTeam"].to_numpy()
    hg, ag = df["FTHG"].to_numpy(dtype=float), df["FTAG"].to_numpy(dtype=float)
    for i in range(n):
        kh, ka = (leagues[i], homes[i]), (leagues[i], aways[i])
        rh_home, rh_away = home_r.get(kh, 0.0), away_r.get(kh, 0.0)
        ra_home, ra_away = home_r.get(ka, 0.0), away_r.get(ka, 0.0)
        expected = _expected_gd(rh_home) - _expected_gd(ra_away)
        out[i] = (rh_home, ra_away, expected, (rh_home + rh_away) / 2 - (ra_home + ra_away) / 2)
        if np.isnan(hg[i]) or np.isnan(ag[i]):
            continue
        error = (hg[i] - ag[i]) - expected
        psi = PI_C * np.log10(1.0 + abs(error)) * np.sign(error)
        d_home, d_away = psi * PI_LAMBDA, -psi * PI_LAMBDA
        home_r[kh] = rh_home + d_home
        away_r[kh] = rh_away + d_home * PI_GAMMA
        away_r[ka] = ra_away + d_away
        home_r[ka] = ra_home + d_away * PI_GAMMA
    return pd.DataFrame(out, index=df.index, columns=["pi_home_h", "pi_away_a", "pi_exp_gd", "pi_overall_diff"])


def rolling_shot_strengths(df: pd.DataFrame, home_col: str = "HST", away_col: str = "AST", prefix: str = "sot") -> pd.DataFrame:
    """Forces d'attaque / défense sur les tirs cadrés, réestimées chaque semaine sur le passé
    (même schéma que le Dixon-Coles glissant : matchs strictement antérieurs au lundi de la semaine)."""
    cols = [f"{prefix}_att_h", f"{prefix}_def_h", f"{prefix}_att_a", f"{prefix}_def_a",
            f"{prefix}_exp_h", f"{prefix}_exp_a", f"{prefix}_exp_diff"]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    days_all = (df["Date"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    decay = np.log(2) / cfg.DC_HALFLIFE_DAYS
    for _, idx in df.groupby("league", sort=False).indices.items():
        g = df.iloc[idx]
        days = days_all[idx]
        monday = days - g["Date"].dt.day_of_week.to_numpy()
        teams = pd.Index(sorted(set(g["HomeTeam"]) | set(g["AwayTeam"])))
        h_idx, a_idx = teams.get_indexer(g["HomeTeam"]), teams.get_indexer(g["AwayTeam"])
        hs, as_ = g[home_col].to_numpy(dtype=float), g[away_col].to_numpy(dtype=float)
        played = ~np.isnan(hs) & ~np.isnan(as_)
        res = np.full((len(g), len(cols)), np.nan)
        for wk in np.unique(monday):
            lo, hi = np.searchsorted(days, wk - cfg.DC_WINDOW_DAYS, "left"), np.searchsorted(days, wk, "left")
            sel = np.arange(lo, hi)[played[lo:hi]]
            if len(sel) < MIN_MATCHES:
                continue
            w = np.exp(-decay * (wk - days[sel]))
            att, dfn, c, gamma = fit_poisson_strengths(h_idx[sel], a_idx[sel], hs[sel], as_[sel], w, len(teams))
            seen = (np.bincount(h_idx[sel], minlength=len(teams)) + np.bincount(a_idx[sel], minlength=len(teams))) > 0
            att, dfn = np.where(seen, att, np.nan), np.where(seen, dfn, np.nan)
            cur = np.where(monday == wk)[0]
            eh = c * gamma * att[h_idx[cur]] * dfn[a_idx[cur]]
            ea = c * att[a_idx[cur]] * dfn[h_idx[cur]]
            res[cur] = np.column_stack([att[h_idx[cur]], dfn[h_idx[cur]], att[a_idx[cur]], dfn[a_idx[cur]], eh, ea, eh - ea])
        out.iloc[idx] = res
    return out


def build_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Table (match_id + variables supplémentaires), à joindre à la table de variables principale."""
    df = df.sort_values(["Date", "league", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    return pd.concat([df[["match_id"]], pi_ratings(df), rolling_shot_strengths(df)], axis=1)


PI_COLS = ["pi_home_h", "pi_away_a", "pi_exp_gd", "pi_overall_diff"]
SOT_COLS = ["sot_att_h", "sot_def_h", "sot_att_a", "sot_def_a", "sot_exp_h", "sot_exp_a", "sot_exp_diff"]
