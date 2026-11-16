"""Modèle de buts Poisson / Dixon-Coles estimé sur fenêtre glissante (passé uniquement)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

import config as cfg

# Valeurs a priori pour une équipe sans aucun match dans la fenêtre (promu typique)
NEW_TEAM_ATT = 0.85
NEW_TEAM_DEF = 1.15
MIN_MATCHES = 60


def fit_poisson_strengths(h_idx, a_idx, hg, ag, w, n_teams, prior=cfg.DC_PRIOR_MATCHES, n_iter=40):
    """Maximum de vraisemblance pondéré du modèle lambda = c*gamma*att_h*def_a, mu = c*att_a*def_h.

    Mises à jour itératives en forme close, avec rétrécissement vers l'équipe moyenne
    (`prior` pseudo-matchs) pour stabiliser les équipes peu observées.
    """
    att, dfn = np.ones(n_teams), np.ones(n_teams)
    w_sum = w.sum()
    s_hg, s_ag = (w * hg).sum(), (w * ag).sum()
    m = (s_hg + s_ag) / (2 * w_sum)                 # buts moyens par équipe et par match
    c, gamma = s_ag / w_sum, s_hg / max(s_ag, 1e-9)
    scored = np.bincount(h_idx, w * hg, n_teams) + np.bincount(a_idx, w * ag, n_teams)
    conceded = np.bincount(h_idx, w * ag, n_teams) + np.bincount(a_idx, w * hg, n_teams)
    for _ in range(n_iter):
        exp_att = (np.bincount(h_idx, w * c * gamma * dfn[a_idx], n_teams)
                   + np.bincount(a_idx, w * c * dfn[h_idx], n_teams))
        att = (scored + prior * m) / (exp_att + prior * m)
        exp_def = (np.bincount(h_idx, w * c * att[a_idx], n_teams)
                   + np.bincount(a_idx, w * c * gamma * att[h_idx], n_teams))
        dfn = (conceded + prior * m) / (exp_def + prior * m)
        base_h, base_a = att[h_idx] * dfn[a_idx], att[a_idx] * dfn[h_idx]
        c = s_ag / (w * base_a).sum()
        gamma = s_hg / (c * (w * base_h).sum())
    return att, dfn, c, gamma


def _tau(hg, ag, lam, mu, rho):
    """Correction de Dixon-Coles pour les scores 0-0, 1-0, 0-1 et 1-1."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, t)
    t = np.where((hg == 0) & (ag == 1), 1 + lam * rho, t)
    t = np.where((hg == 1) & (ag == 0), 1 + mu * rho, t)
    t = np.where((hg == 1) & (ag == 1), 1 - rho, t)
    return t


def fit_rho(hg, ag, lam, mu, w) -> float:
    """Estime le paramètre de dépendance rho par maximum de vraisemblance (1 dimension)."""
    def nll(rho):
        return -(w * np.log(np.clip(_tau(hg, ag, lam, mu, rho), 1e-9, None))).sum()
    return float(minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x)


def outcome_probs(lam, mu, rho, max_goals=cfg.DC_MAX_GOALS) -> np.ndarray:
    """Probabilités (domicile, nul, extérieur) à partir des intensités de buts."""
    lam, mu = np.atleast_1d(lam).astype(float), np.atleast_1d(mu).astype(float)
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals[None, :], lam[:, None])
    pa = poisson.pmf(goals[None, :], mu[:, None])
    joint = ph[:, :, None] * pa[:, None, :]
    joint[:, 0, 0] *= 1 - lam * mu * rho
    joint[:, 0, 1] *= 1 + lam * rho
    joint[:, 1, 0] *= 1 + mu * rho
    joint[:, 1, 1] *= 1 - rho
    joint = np.clip(joint, 0, None)
    p_home = np.tril(joint, -1).sum(axis=(1, 2))
    p_draw = np.trace(joint, axis1=1, axis2=2)
    p_away = np.triu(joint, 1).sum(axis=(1, 2))
    p = np.column_stack([p_home, p_draw, p_away])
    return p / p.sum(axis=1, keepdims=True)


DC_COLS = ["dc_att_h", "dc_def_h", "dc_att_a", "dc_def_a", "dc_lambda", "dc_mu", "dc_p_h", "dc_p_d", "dc_p_a"]


def rolling_dixon_coles(df: pd.DataFrame) -> pd.DataFrame:
    """Variables Dixon-Coles d'avant-match.

    Pour chaque championnat, le modèle est réestimé chaque semaine sur les matchs
    STRICTEMENT antérieurs au lundi de la semaine du match (fenêtre glissante pondérée).
    Le calendrier de réestimation ne dépend que de la date du match : aucune donnée future.
    """
    out = pd.DataFrame(np.nan, index=df.index, columns=DC_COLS)
    days_all = (df["Date"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    decay = np.log(2) / cfg.DC_HALFLIFE_DAYS
    for league, idx in df.groupby("league", sort=False).indices.items():
        g = df.iloc[idx]
        days = days_all[idx]                                  # déjà trié par date
        monday = days - g["Date"].dt.day_of_week.to_numpy()
        teams = pd.Index(sorted(set(g["HomeTeam"]) | set(g["AwayTeam"])))
        h_idx, a_idx = teams.get_indexer(g["HomeTeam"]), teams.get_indexer(g["AwayTeam"])
        hg, ag = g["FTHG"].to_numpy(dtype=float), g["FTAG"].to_numpy(dtype=float)
        played = ~np.isnan(hg) & ~np.isnan(ag)
        res = np.full((len(g), len(DC_COLS)), np.nan)
        for wk in np.unique(monday):
            lo, hi = np.searchsorted(days, wk - cfg.DC_WINDOW_DAYS, "left"), np.searchsorted(days, wk, "left")
            sel = np.arange(lo, hi)[played[lo:hi]]
            if len(sel) < MIN_MATCHES:
                continue
            w = np.exp(-decay * (wk - days[sel]))
            att, dfn, c, gamma = fit_poisson_strengths(h_idx[sel], a_idx[sel], hg[sel], ag[sel], w, len(teams))
            seen = (np.bincount(h_idx[sel], minlength=len(teams)) + np.bincount(a_idx[sel], minlength=len(teams))) > 0
            att, dfn = np.where(seen, att, NEW_TEAM_ATT), np.where(seen, dfn, NEW_TEAM_DEF)
            lam_fit = c * gamma * att[h_idx[sel]] * dfn[a_idx[sel]]
            mu_fit = c * att[a_idx[sel]] * dfn[h_idx[sel]]
            rho = fit_rho(hg[sel], ag[sel], lam_fit, mu_fit, w)
            cur = np.where(monday == wk)[0]
            lam = c * gamma * att[h_idx[cur]] * dfn[a_idx[cur]]
            mu = c * att[a_idx[cur]] * dfn[h_idx[cur]]
            res[cur, 0], res[cur, 1] = att[h_idx[cur]], dfn[h_idx[cur]]
            res[cur, 2], res[cur, 3] = att[a_idx[cur]], dfn[a_idx[cur]]
            res[cur, 4], res[cur, 5] = lam, mu
            res[cur, 6:9] = outcome_probs(lam, mu, rho)
        out.iloc[idx] = res
    return out
