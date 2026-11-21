"""Calendrier des matchs à venir (flux publics de fixturedownload.com) et cotes disponibles.

Les noms d'équipes du flux sont reliés à ceux de l'historique automatiquement : on apparie les
matchs déjà joués (championnat + date + score) et on retient, pour chaque nom, le correspondant
majoritaire. Aucun dictionnaire à maintenir quand un promu arrive.
"""
from __future__ import annotations

import io
import json
import urllib.request
from datetime import timedelta

import numpy as np
import pandas as pd

import config as cfg
from src.pipeline import resolve_team

FEEDS = {"E0": "epl", "SP1": "la-liga", "I1": "serie-a", "D1": "bundesliga", "F1": "ligue-1"}
FEED_URL = "https://fixturedownload.com/feed/json/{slug}-{season}"
ODDS_URL = "https://www.football-data.co.uk/fixtures.csv"
HEADERS = {"User-Agent": "match-predictor/1.0 (usage personnel)"}
CACHE = cfg.RAW_DIR / "fixtures"
LOCAL_TZ = "Europe/Paris"


def _download(url: str, name: str) -> bytes:
    """Télécharge et met en cache ; en cas d'échec réseau, réutilise la dernière copie."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30).read()
        path.write_bytes(data)
        return data
    except Exception:
        if path.exists():
            return path.read_bytes()
        raise


def load_calendar(history: pd.DataFrame, season: int = cfg.CURRENT_SEASON) -> pd.DataFrame:
    """Tous les matchs de la saison (joués ou non), avec les noms d'équipes de l'historique."""
    parts = []
    for league, slug in FEEDS.items():
        if league not in cfg.LEAGUES:
            continue
        raw = json.loads(_download(FEED_URL.format(slug=slug, season=season), f"{slug}-{season}.json"))
        f = pd.DataFrame(raw)
        f["league"] = league
        f["kickoff"] = pd.to_datetime(f["DateUtc"], utc=True)
        f["Date"] = f["kickoff"].dt.tz_localize(None).dt.normalize()
        hist = history[(history.league == league) & (history.season == season)]
        j = f.dropna(subset=["HomeTeamScore"]).merge(
            hist, left_on=["Date", "HomeTeamScore", "AwayTeamScore"], right_on=["Date", "FTHG", "FTAG"], suffixes=("_feed", ""))
        votes = pd.concat([j[["HomeTeam_feed", "HomeTeam"]].set_axis(["feed", "team"], axis=1),
                           j[["AwayTeam_feed", "AwayTeam"]].set_axis(["feed", "team"], axis=1)])
        mapping = votes.groupby("feed")["team"].agg(lambda s: s.value_counts().idxmax()).to_dict()
        known = sorted(set(hist.HomeTeam) | set(hist.AwayTeam)) or sorted(set(history[history.league == league].HomeTeam))
        for col in ("HomeTeam", "AwayTeam"):
            f[col] = [mapping.get(t) or resolve_team(t, known) for t in f[col]]
        f["played"] = f["HomeTeamScore"].notna()
        f["FTHG"], f["FTAG"] = pd.to_numeric(f["HomeTeamScore"], errors="coerce"), pd.to_numeric(f["AwayTeamScore"], errors="coerce")
        parts.append(f[["league", "kickoff", "Date", "HomeTeam", "AwayTeam", "played", "RoundNumber", "FTHG", "FTAG"]])
    return pd.concat(parts, ignore_index=True).sort_values(["kickoff", "league", "HomeTeam"]).reset_index(drop=True)


def upcoming_week(calendar: pd.DataFrame, history: pd.DataFrame, today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, str]:
    """Matchs non joués de la semaine en cours (lundi -> dimanche). S'il n'y en a plus, les sept
    jours qui suivent le prochain match (trêve internationale, par exemple). Renvoie aussi un libellé."""
    today = (today or pd.Timestamp.now(tz=LOCAL_TZ)).tz_convert("UTC")
    done = set(zip(history.league, history.HomeTeam, history.AwayTeam, history.season))
    season = cfg.CURRENT_SEASON
    todo = calendar[~calendar.played & (calendar.kickoff >= today - timedelta(hours=3))]
    todo = todo[[(l, h, a, season) not in done for l, h, a in zip(todo.league, todo.HomeTeam, todo.AwayTeam)]]
    monday = (today.tz_convert(LOCAL_TZ).normalize() - timedelta(days=today.tz_convert(LOCAL_TZ).day_of_week)).tz_convert("UTC")
    week = todo[todo.kickoff < monday + timedelta(days=7)]
    if len(week):
        return week.reset_index(drop=True), "Semaine en cours"
    if todo.empty:
        return todo, "Aucun match à venir dans le calendrier"
    start = todo.kickoff.min()
    nxt = todo[todo.kickoff < start.normalize() + timedelta(days=7)]
    return nxt.reset_index(drop=True), "Aucun match cette semaine : prochaine journée"


def load_odds() -> pd.DataFrame:
    """Cotes d'avant-match publiées par football-data.co.uk pour les prochains jours (peut être vide)."""
    try:
        f = pd.read_csv(io.BytesIO(_download(ODDS_URL, "fixtures_odds.csv")), encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=["league", "HomeTeam", "AwayTeam", "odds_h", "odds_d", "odds_a"])
    from src.data import TEAM_ALIASES
    f = f[f["Div"].isin(cfg.LEAGUES)].rename(columns={"Div": "league"}).copy()
    for col in ("HomeTeam", "AwayTeam"):
        f[col] = f[col].astype(str).str.strip().map(lambda t: TEAM_ALIASES.get(t, t))
    for out, cands in (("odds_h", ["B365H", "AvgH"]), ("odds_d", ["B365D", "AvgD"]), ("odds_a", ["B365A", "AvgA"])):
        f[out] = np.nan
        for c in cands:
            if c in f.columns:
                f[out] = f[out].fillna(pd.to_numeric(f[c], errors="coerce"))
    return f[["league", "HomeTeam", "AwayTeam", "odds_h", "odds_d", "odds_a"]].dropna()
