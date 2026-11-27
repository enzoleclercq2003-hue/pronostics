"""Blessés et suspendus avant chaque match, depuis les listes structurées de Rotowire.

Rotowire publie, pour chaque match à venir des cinq grands championnats, la liste des joueurs blessés
(OUT), incertains (QUES) et suspendus (SUS) de chaque équipe, tenue à jour par sa rédaction. C'est une
source bien plus sûre que l'extraction automatique dans le texte des articles (src/news.py) : le statut
de chaque joueur y est donné explicitement. Une page par championnat, gardée une heure en cache ;
usage personnel, quelques requêtes par heure au plus.

`absences(fixtures, roster, team_names)` renvoie le même format que news.extract_absences, plus
l'ensemble des équipes couvertes (celles dont Rotowire publie le match).
"""
from __future__ import annotations

import html
import re
import time
import unicodedata

import pandas as pd
import requests

import config as cfg
from src.fixtures import _simil

PAGES = {"E0": "EPL", "SP1": "LIGA", "I1": "SERI", "D1": "BUND", "F1": "FRAN"}
URL = "https://www.rotowire.com/soccer/lineups.php?league={code}"
CACHE_DIR = cfg.RAW_DIR / "rotowire"
MAX_AGE = 3600                       # secondes
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"}
STATUS = {"OUT": ("out", "confirmée", "Blessé"), "SUS": ("out", "confirmée", "Suspendu"), "QUES": ("doubt", "incertain", "Incertain")}
COLS = ["team", "player", "player_id", "status", "confidence", "until", "long", "sources"]


def _page(code: str) -> str | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{code}.html"
    if path.exists() and time.time() - path.stat().st_mtime < MAX_AGE:
        return path.read_text(encoding="utf-8")
    try:
        r = requests.get(URL.format(code=code), headers=HEADERS, timeout=25)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
        return r.text
    except Exception:
        return path.read_text(encoding="utf-8") if path.exists() else None     # dernière version connue


def parse(page: str) -> list[dict]:
    """Un dict par match : home, away, et pour chaque côté la liste (nom complet, statut)."""
    out = []
    for box in page.split('<div class="lineup__box">')[1:]:
        teams = [html.unescape(t).strip() for t in re.findall(r'lineup__mteam is-(?:home|visit)">\s*([^<]+?)\s*<', box)]
        lists = re.findall(r'<ul class="lineup__list is-(home|visit)">(.*?)</ul>', box, flags=re.S)
        if len(teams) != 2 or len(lists) != 2:
            continue
        sides = {}
        for side, body in lists:
            inj = body.split("lineup__title is-middle", 1)
            players = re.findall(r'<a title="([^"]+)"[^>]*>.*?</a>\s*<span class="lineup__inj">\s*([A-Z]+)\s*</span>', inj[1], flags=re.S) if len(inj) > 1 else []
            sides[side] = [(html.unescape(n).strip(), s) for n, s in players]
        out.append({"home": teams[0], "away": teams[1], "h": sides.get("home", []), "a": sides.get("visit", [])})
    return out


def _n(s: str) -> str:
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", s.lower()) if c.isalnum() or c == " ").split())


def _player(name: str, cands: pd.DataFrame):
    """Joueur de l'effectif (noms ESPN) correspondant au nom Rotowire, sinon None."""
    if cands.empty:
        return None
    n = _n(name)
    norm = cands["name"].map(_n)
    hit = cands[norm == n]
    if len(hit) == 1:
        return hit.iloc[0]
    last = n.split()[-1]
    hit = cands[norm.str.split().str[-1] == last]
    if len(hit) == 1 and len(last) >= 4:
        return hit.iloc[0]
    sims = norm.map(lambda x: _simil(x, n) if x and n else 0)
    return cands.loc[sims.idxmax()] if sims.max() >= 0.85 else None


def absences(fixtures: pd.DataFrame, roster: pd.DataFrame, team_names: dict, now: pd.Timestamp | None = None):
    """(absences au format de news.extract_absences, équipes couvertes par Rotowire)."""
    now = now or pd.Timestamp.now(tz="UTC")
    rows, covered = [], set()
    for league, code in PAGES.items():
        fx = fixtures[fixtures.league == league]
        if fx.empty:
            continue
        page = _page(code)
        if not page:
            continue
        names = lambda t: [t, *team_names.get(t, [])]
        sim = lambda rw, t: max(_simil(rw, x) for x in names(t))
        for m in parse(page):
            # l'affiche à venir dont les deux noms ressemblent le plus ; rapprochement incertain : on s'abstient
            best, r = max(((sim(m["home"], f.HomeTeam) + sim(m["away"], f.AwayTeam), f) for f in fx.itertuples()), key=lambda x: x[0])
            if best < 1.4:
                continue
            for team, lst in ((r.HomeTeam, m["h"]), (r.AwayTeam, m["a"])):
                covered.add(team)
                cands = roster[roster.team == team]
                for name, st in lst:
                    if st not in STATUS:
                        continue
                    status, conf, label = STATUS[st]
                    p = _player(name, cands)
                    rows.append({"team": team, "player": p["name"] if p is not None else name,
                                 "player_id": int(p["player_id"]) if p is not None else None,
                                 "status": status, "confidence": conf, "until": now + pd.Timedelta(days=5), "long": False,
                                 "sources": [{"source": "Rotowire", "title": f"{label} · liste d'avant-match", "link": URL.format(code=code),
                                              "date": f"{now:%Y-%m-%d}"}]})
    ab = pd.DataFrame(rows, columns=COLS).drop_duplicates(["team", "player"]) if rows else pd.DataFrame(columns=COLS)
    return ab, covered


def merge(inj: pd.DataFrame, covered: set, from_news: pd.DataFrame) -> pd.DataFrame:
    """Rotowire fait foi. Pour une équipe couverte, les articles ne font qu'ajouter leurs liens et la durée
    aux joueurs que Rotowire déclare absents : une absence repérée seulement dans le texte d'un article est
    écartée (l'extraction automatique peut se tromper de joueur). Les équipes non couvertes gardent les
    absences tirées des articles."""
    if from_news is None or not len(from_news):
        return inj
    news = from_news.copy()
    keyed = {(t, _n(p)): i for i, (t, p) in enumerate(zip(inj.team, inj.player))}
    by_id = {(t, pid): i for i, (t, pid) in enumerate(zip(inj.team, inj.player_id)) if pid is not None and pd.notna(pid)}
    inj = inj.copy().reset_index(drop=True)
    extra = []
    for r in news.itertuples():
        i = by_id.get((r.team, r.player_id), keyed.get((r.team, _n(r.player))))
        if i is not None:                                   # même joueur : on ajoute les articles et la durée
            inj.at[i, "sources"] = inj.at[i, "sources"] + list(r.sources)
            inj.at[i, "long"] = bool(inj.at[i, "long"] or r.long)
            inj.at[i, "until"] = max(inj.at[i, "until"], r.until)
        elif r.team not in covered:
            extra.append(r._asdict())
    if extra:
        inj = pd.concat([inj, pd.DataFrame(extra).drop(columns="Index", errors="ignore")], ignore_index=True)
    return inj[COLS]
