"""Modèle de production avec variables joueurs (valeur marchande des onze, composition officielle).

Trois niveaux, choisis match par match selon l'information disponible :
- « composition » : composition officielle publiée (environ 1 h avant le coup d'envoi) ;
- « effectifs »   : valeur du onze aligné au match précédent et de l'effectif récent ;
- « base »        : modèle d'origine, quand les joueurs d'une des deux équipes sont mal identifiés.

Les deux modèles joueurs sont des régressions logistiques (mêmes hyperparamètres que le modèle de
base) réentraînées à chaque exécution sur tout l'historique depuis 2013/14 : l'entraînement prend
moins d'une seconde, et le modèle intègre ainsi toujours les derniers résultats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.live_players import live_tables
from src.models import Model, load_params
from src.player_features import LIVE_LINEUP_COLS, LIVE_PRE_COLS, build_player_features, load_tables

PLAYER_START_SEASON = 2013
LEVELS = {"composition": "composition officielle prise en compte", "effectifs": "valeur des effectifs prise en compte",
          "base": "modèle de base (effectif mal identifié)"}


def player_fold(base, fixtures: pd.DataFrame | None, history: pd.DataFrame, verbose: bool = False):
    """Table de variables complète (base + joueurs) pour l'historique et les matchs à venir."""
    fold, fx, combined = base.fixture_fold(fixtures, history)
    tables = load_tables()
    extra_link, extra_lineups, report = live_tables(combined, tables, verbose=verbose)
    players = build_player_features(combined, tables, freeze_valuations=True, extra_link=extra_link, extra_lineups=extra_lineups)
    return fold.merge(players, on="match_id", how="left"), fx, report


def fit_player_models(base, fold: pd.DataFrame, train_mask: np.ndarray) -> dict[str, Model]:
    train = fold[train_mask & (fold.season >= PLAYER_START_SEASON)]
    params = load_params("logreg")
    return {"effectifs": Model("logreg", params, base.features + LIVE_PRE_COLS).fit(train),
            "composition": Model("logreg", params, base.features + LIVE_PRE_COLS + LIVE_LINEUP_COLS).fit(train)}


def predict_levels(base, models: dict[str, Model], rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Probabilités et niveau utilisé pour chaque ligne."""
    p = base.predict_from_features(rows)
    level = np.full(len(rows), "base", dtype=object)
    has_pre = rows[["diff_pxi_logmv", "diff_top11_logmv"]].notna().all(axis=1).to_numpy()
    has_xi = has_pre & rows[["diff_xi_logmv", "diff_xi_changes", "diff_xi_top11_share"]].notna().all(axis=1).to_numpy()
    if has_pre.any():
        p[has_pre] = models["effectifs"].predict_proba(rows[has_pre])
        level[has_pre] = "effectifs"
    if has_xi.any():
        p[has_xi] = models["composition"].predict_proba(rows[has_xi])
        level[has_xi] = "composition"
    return p, level


def predict_week(base, fixtures: pd.DataFrame, history: pd.DataFrame, verbose: bool = False, use_players: bool = True) -> tuple[pd.DataFrame, dict]:
    """Prédictions pour des matchs à venir (colonnes date, home_team, away_team).

    `use_players=False` : modèle de base seul, sans aucune donnée joueur (version publique du site :
    les compositions ESPN et les valeurs Transfermarkt ne sont pas utilisables commercialement)."""
    if not use_players:
        fold, fx, _ = base.fixture_fold(fixtures, history)
        rows = fold.set_index("match_id").loc[fx["match_id"]].reset_index()
        p = base.predict_from_features(rows)
        out = pd.DataFrame({"p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "level": "base",
                            "elo_home": rows["elo_h"].to_numpy(), "elo_away": rows["elo_a"].to_numpy()})
        out["detail"] = match_details(fold, rows, {})
        return out, {}
    fold, fx, report = player_fold(base, fixtures, history, verbose)
    models = fit_player_models(base, fold, fold["y"].notna().to_numpy())
    rows = fold.set_index("match_id").loc[fx["match_id"]].reset_index()
    p, level = predict_levels(base, models, rows)
    p_base = base.predict_from_features(rows)
    out = pd.DataFrame({"p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "level": level,
                        "base_home": p_base[:, 0], "base_draw": p_base[:, 1], "base_away": p_base[:, 2],
                        "elo_home": rows["elo_h"].to_numpy(), "elo_away": rows["elo_a"].to_numpy()})
    out["detail"] = match_details(fold, rows, report.get("absences", {}))
    return out, report


def match_details(fold: pd.DataFrame, rows: pd.DataFrame, absences: dict, n_form: int = 5, n_elo: int = 38) -> list[dict]:
    """Contexte affiché dans la fiche d'un match : buts attendus, forme récente, historique Elo, absents."""
    played = fold[fold["y"].notna()]
    home = played[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "elo_h"]].set_axis(["date", "team", "opp", "gf", "ga", "elo"], axis=1).assign(venue="H")
    away = played[["Date", "AwayTeam", "HomeTeam", "FTAG", "FTHG", "elo_a"]].set_axis(["date", "team", "opp", "gf", "ga", "elo"], axis=1).assign(venue="A")
    long = pd.concat([home, away]).sort_values("date", kind="mergesort")
    by_team = {t: g for t, g in long.groupby("team")}

    def team(name: str, elo_now: float) -> dict:
        g = by_team.get(name)
        if g is None:
            return {"form": [], "elo": []}
        last = g.tail(n_form)
        form = [{"date": f"{d:%Y-%m-%d}", "opp": o, "venue": v, "gf": int(gf), "ga": int(ga),
                 "res": "W" if gf > ga else "D" if gf == ga else "L"}
                for d, o, v, gf, ga in zip(last.date, last.opp, last.venue, last.gf, last.ga)]
        elo = [round(float(x)) for x in g["elo"].tail(n_elo - 1)] + ([round(float(elo_now))] if pd.notna(elo_now) else [])
        return {"form": form, "elo": elo}

    details = []
    for r in rows.itertuples():
        lam, mu = getattr(r, "dc_lambda", np.nan), getattr(r, "dc_mu", np.nan)
        absent = absences.get(int(r.match_id), {})
        details.append({"lam": None if pd.isna(lam) else float(lam), "mu": None if pd.isna(mu) else float(mu),
                        "home": {**team(r.HomeTeam, r.elo_h), "absent": absent.get("h", [])},
                        "away": {**team(r.AwayTeam, r.elo_a), "absent": absent.get("a", [])}})
    return details
