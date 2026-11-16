"""Prolonge les variables joueurs sur la saison en cours : compositions ESPN + valeurs Transfermarkt.

Les joueurs ESPN sont rattachés aux joueurs Transfermarkt par leur nom (au sein du club d'abord,
puis globalement). Un joueur non retrouvé garde une valeur inconnue ; son onze reçoit alors la
médiane des coéquipiers (cf. player_features._aggregate).
"""
from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches

import numpy as np
import pandas as pd

import config as cfg
from src.espn import load_season_lineups
from src.player_features import TM_DIR, _days, map_games

GAME_OFFSET = 10_000_000_000        # identifiants de match synthétiques (hors plage Transfermarkt)


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def _team_votes(events: pd.DataFrame, history: pd.DataFrame) -> dict:
    """Nom d'équipe ESPN -> nom de l'historique (vote sur championnat + date + score)."""
    h = history[history.season == cfg.CURRENT_SEASON][["league", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    j = events[events.completed].merge(h, left_on=["league", "Date", "hg", "ag"], right_on=["league", "Date", "FTHG", "FTAG"])
    votes = pd.concat([j[["league", "espn_home", "HomeTeam"]].set_axis(["league", "espn", "team"], axis=1),
                       j[["league", "espn_away", "AwayTeam"]].set_axis(["league", "espn", "team"], axis=1)])
    return votes.groupby(["league", "espn"])["team"].agg(lambda s: s.value_counts().idxmax()).to_dict()


class PlayerMatcher:
    """Associe (nom ESPN, club) à un player_id Transfermarkt."""

    def __init__(self, vals: pd.DataFrame, club_pools: dict[int, set]):
        players = pd.read_csv(TM_DIR / "players.csv.gz", usecols=["player_id", "name", "last_season", "current_club_id"])
        last_val = vals.sort_values("day").groupby("player_id")["market_value_in_eur"].last()
        players["value"] = players.player_id.map(last_val).fillna(0)
        players["norm"] = players["name"].map(_norm)
        self.players = players
        active = players[players.last_season >= 2024].sort_values("value", ascending=False)
        self.global_exact = active.drop_duplicates("norm").set_index("norm")["player_id"].to_dict()
        self.global_names = list(self.global_exact)
        self.pools = {}
        for club, ids in club_pools.items():
            sub = players[players.player_id.isin(ids) | (players.current_club_id == club)].sort_values("value", ascending=False)
            self.pools[club] = sub.drop_duplicates("norm").set_index("norm")["player_id"].to_dict()
        self.cache: dict[tuple, tuple[int | None, str]] = {}

    def match(self, name: str, club: int) -> tuple[int | None, str]:
        key = (name, club)
        if key not in self.cache:
            self.cache[key] = self._match(_norm(name), self.pools.get(club, {}))
        return self.cache[key]

    def _match(self, n: str, pool: dict) -> tuple[int | None, str]:
        if not n:
            return None, "aucun"
        if n in pool:
            return pool[n], "club exact"
        if n in self.global_exact:
            return self.global_exact[n], "global exact"
        tokens = n.split()
        # même nom de famille dans le club (« Vinicius Junior » / « Vinicius Jr », prénoms composés, etc.)
        cands = [k for k in pool if k.split()[-1] == tokens[-1] or (len(tokens) > 1 and set(tokens) <= set(k.split())) or set(k.split()) <= set(tokens)]
        if len(cands) == 1:
            return pool[cands[0]], "club nom"
        close = get_close_matches(n, list(pool), n=1, cutoff=0.8)
        if close:
            return pool[close[0]], "club approché"
        close = get_close_matches(n, self.global_names, n=1, cutoff=0.92)
        if close:
            return self.global_exact[close[0]], "global approché"
        return None, "aucun"


def probable_absences(vals: pd.DataFrame, link: pd.DataFrame, lineups: pd.DataFrame, names: dict) -> dict:
    """Pour chaque match de la saison en cours : joueurs importants absents de la dernière feuille de match.

    « Important » = parmi les onze joueurs les plus chers alignés ou sur le banc lors des cinq derniers
    matchs de championnat du club. Renvoie {match_id: {"h": [noms], "a": [noms]}}. C'est un indice
    (blessure, suspension, choix de l'entraîneur), pas une liste de blessés confirmée.
    """
    value = vals.sort_values("day").groupby("player_id")["market_value_in_eur"].last()
    cg = pd.concat([link[["match_id", "game_id", "day", "home_club_id"]].rename(columns={"home_club_id": "club_id"}).assign(side="h"),
                    link[["match_id", "game_id", "day", "away_club_id"]].rename(columns={"away_club_id": "club_id"}).assign(side="a")])
    cg = cg.sort_values(["club_id", "day"], kind="mergesort").reset_index(drop=True)
    cg["k"] = cg.groupby("club_id").cumcount()
    lu = lineups.merge(cg[["game_id", "club_id", "k"]], on=["game_id", "club_id"])
    squads = {(c, k): set(g) for (c, k), g in lu.groupby(["club_id", "k"])["player_id"]}
    out: dict = {}
    for r in cg.itertuples():
        prev = squads.get((r.club_id, r.k - 1))
        if not prev:
            continue
        recent = set().union(*(squads.get((r.club_id, r.k - lag), set()) for lag in range(1, 6)))
        known = sorted((p for p in recent if p in value.index and value[p] > 0), key=lambda p: -value[p])
        if len(known) < 11:
            continue
        missing = [names.get(p) for p in known[:11] if p not in prev]
        out.setdefault(int(r.match_id), {"h": [], "a": []})[r.side] = [n for n in missing if n]
    return out


def live_tables(combined: pd.DataFrame, tables: dict[str, pd.DataFrame], verbose: bool = False):
    """Renvoie (extra_link, extra_lineups, rapport) pour les matchs de la saison en cours de `combined`
    (historique + matchs à venir), à passer à build_player_features."""
    cur = combined[combined.season == cfg.CURRENT_SEASON]
    played_hist = combined[combined["FTR"].notna()]
    # Club Transfermarkt de chaque équipe (dernier identifiant connu) ; identifiant synthétique pour un club jamais vu
    link = map_games(played_hist, tables["games"])
    names = link.merge(played_hist[["match_id", "HomeTeam", "AwayTeam"]], on="match_id")
    club_of = {**names.sort_values("day").groupby("AwayTeam")["away_club_id"].last().to_dict(),
               **names.sort_values("day").groupby("HomeTeam")["home_club_id"].last().to_dict()}
    for i, t in enumerate(sorted(set(cur.HomeTeam) | set(cur.AwayTeam))):
        club_of.setdefault(t, -(i + 1))
    # Effectif récent connu de chaque club (compositions de la dernière saison Transfermarkt)
    lu = tables["lineups"]
    recent = lu[lu.day >= lu.day.max() - 400]
    pools = recent.groupby("club_id")["player_id"].agg(set).to_dict()

    dates = {lg: [pd.Timestamp(d) for d in g.Date.unique()] for lg, g in cur.groupby("league")}
    events, lineups = load_season_lineups(dates, verbose=verbose)
    team_map = _team_votes(events, played_hist)
    events["HomeTeam"] = [team_map.get((l, t)) for l, t in zip(events.league, events.espn_home)]
    events["AwayTeam"] = [team_map.get((l, t)) for l, t in zip(events.league, events.espn_away)]
    ev = events.dropna(subset=["HomeTeam", "AwayTeam"]).merge(
        cur[["match_id", "league", "HomeTeam", "AwayTeam", "Date"]], on=["league", "HomeTeam", "AwayTeam"], suffixes=("_espn", ""))
    ev["game_id"] = GAME_OFFSET + ev["event"].astype("int64")

    # Lien : tous les matchs de la saison en cours (ceux sans événement ESPN gardent un identifiant synthétique)
    xl = cur[["match_id", "league", "season", "Date", "HomeTeam", "AwayTeam"]].merge(ev[["match_id", "game_id"]], on="match_id", how="left")
    xl["game_id"] = xl["game_id"].fillna(2 * GAME_OFFSET + xl["match_id"]).astype("int64")
    xl["day"] = _days(xl["Date"]).astype("int64")
    xl["home_club_id"], xl["away_club_id"] = xl.HomeTeam.map(club_of), xl.AwayTeam.map(club_of)
    extra_link = xl[["match_id", "game_id", "league", "season", "day", "home_club_id", "away_club_id"]]

    matcher = PlayerMatcher(tables["vals"], pools)
    lu_ev = lineups.merge(ev[["event", "game_id", "league", "HomeTeam", "AwayTeam", "espn_home"]], on="event")
    lu_ev["team"] = np.where(lu_ev.espn_team == lu_ev.espn_home, lu_ev.HomeTeam, lu_ev.AwayTeam)
    lu_ev["club_id"] = lu_ev.team.map(club_of)
    res = [matcher.match(n, c) for n, c in zip(lu_ev.name, lu_ev.club_id)]
    lu_ev["how"] = [h for _, h in res]
    # joueur non retrouvé : identifiant synthétique stable (valeur inconnue)
    lu_ev["player_id"] = [pid if pid is not None else -int(a) for (pid, _), a in zip(res, lu_ev.athlete)]
    lu_ev["type"] = np.where(lu_ev.starter, "starting_lineup", "substitutes")
    extra_lineups = lu_ev[["game_id", "club_id", "player_id", "type"]].drop_duplicates(["game_id", "club_id", "player_id"])
    starters = lu_ev[lu_ev.starter]
    names = dict(zip(lu_ev.player_id, lu_ev.name))
    try:
        absences = probable_absences(tables["vals"], extra_link, extra_lineups, names)
    except Exception:                                   # information d'affichage : jamais bloquant
        absences = {}
    report = {"absences": absences, "events": len(ev), "matches_current": len(cur), "starters": len(starters),
              "matched_share": float((starters.how != "aucun").mean()) if len(starters) else float("nan"),
              "how": starters.how.value_counts().to_dict(),
              "unmatched": starters[starters.how == "aucun"].groupby("team")["name"].agg(lambda s: sorted(set(s))).to_dict(),
              "with_lineup": set(ev.loc[ev.event.isin(lineups.event), "match_id"])}
    return extra_link, extra_lineups, report
