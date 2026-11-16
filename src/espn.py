"""Compositions de la saison en cours via l'API publique d'ESPN (sans clé).

Sert à prolonger les variables « joueurs » au-delà de juillet 2026, date d'arrêt du jeu de données
Transfermarkt : onze de départ et remplaçants de chaque match de championnat, et composition
officielle des matchs à venir dès qu'elle est publiée (environ une heure avant le coup d'envoi).
Tout est mis en cache dans data/raw/espn ; un match terminé n'est téléchargé qu'une fois.
"""
from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

import config as cfg

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/{what}"
SLUGS = {"E0": "eng.1", "SP1": "esp.1", "I1": "ita.1", "D1": "ger.1", "F1": "fra.1"}
HEADERS = {"User-Agent": "match-predictor/1.0 (usage personnel)"}
CACHE = cfg.RAW_DIR / "espn"
PAUSE = 0.25          # secondes entre deux requêtes, par politesse


def _get(url: str) -> dict:
    time.sleep(PAUSE)
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30).read())


def _cached(path, url: str, final) -> dict | None:
    """Lit le cache s'il est définitif, sinon télécharge. `final(data)` dit si la réponse ne changera plus."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if final(data):
            return data
    try:
        data = _get(url)
    except Exception:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _events(league: str, day: pd.Timestamp) -> list[dict]:
    slug = SLUGS[league]
    path = CACHE / slug / f"scoreboard_{day:%Y%m%d}.json"
    url = BASE.format(slug=slug, what="scoreboard") + f"?dates={day:%Y%m%d}"
    data = _cached(path, url, lambda d: bool(d.get("events")) and all(e["status"]["type"].get("completed") for e in d["events"]))
    out = []
    for e in (data or {}).get("events", []):
        comp = {c["homeAway"]: c for c in e["competitions"][0]["competitors"]}
        done = bool(e["status"]["type"].get("completed"))
        out.append({"league": league, "event": e["id"], "Date": pd.Timestamp(e["date"]).tz_localize(None).normalize(),
                    "espn_home": comp["home"]["team"]["displayName"], "espn_away": comp["away"]["team"]["displayName"],
                    "hg": float(comp["home"].get("score") or 0) if done else None,
                    "ag": float(comp["away"].get("score") or 0) if done else None, "completed": done})
    return out


def _lineup(league: str, event: str, completed: bool) -> list[dict]:
    slug = SLUGS[league]
    path = CACHE / slug / f"summary_{event}.json"
    url = BASE.format(slug=slug, what="summary") + f"?event={event}"
    has = lambda d: any(r.get("roster") for r in d.get("rosters", []))
    data = _cached(path, url, lambda d: completed and has(d) and d.get("_final", False))
    if data is None:
        return []
    if completed and has(data) and not data.get("_final"):
        data = {"rosters": data.get("rosters", []), "_final": True}      # on ne garde que l'utile, marqué définitif
        path.write_text(json.dumps(data), encoding="utf-8")
    rows = []
    for r in data.get("rosters", []):
        for p in r.get("roster", []):
            ath = p.get("athlete", {})
            rows.append({"event": event, "espn_team": r["team"]["displayName"], "athlete": str(ath.get("id")),
                         "name": ath.get("displayName") or ath.get("fullName") or "", "starter": bool(p.get("starter"))})
    return rows


def load_season_lineups(dates_by_league: dict[str, list[pd.Timestamp]], verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Événements et compositions pour les dates données (dates de match connues par championnat)."""
    events, lineups = [], []
    for league, days in dates_by_league.items():
        if league not in SLUGS:
            continue
        for day in sorted(set(days)):
            for e in _events(league, day):
                events.append(e)
                rows = _lineup(league, e["event"], e["completed"])
                lineups.extend(rows)
        if verbose:
            print(f"  ESPN {league} : {sum(1 for e in events if e['league'] == league)} matchs", flush=True)
    ev = pd.DataFrame(events).drop_duplicates("event") if events else pd.DataFrame(
        columns=["league", "event", "Date", "espn_home", "espn_away", "hg", "ag", "completed"])
    lu = pd.DataFrame(lineups) if lineups else pd.DataFrame(columns=["event", "espn_team", "athlete", "name", "starter"])
    return ev, lu
