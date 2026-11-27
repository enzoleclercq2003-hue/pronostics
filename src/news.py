"""Absences annoncées dans l'actualité (blessures, suspensions), à partir de flux RSS publics et gratuits.

Aucune IA payante : on repère dans les titres et chapôs les noms des joueurs de l'effectif de chaque
club (compositions de la saison), puis des expressions d'absence, de doute ou de retour en anglais,
français, espagnol, italien et allemand. Une absence est « confirmée » si au moins deux médias
différents l'annoncent, ou un seul avec une formulation sans ambiguïté (opération, fracture,
plusieurs semaines...). Un article de retour plus récent annule l'absence.

Seuls les faits extraits, le titre et le lien de l'article sont conservés (jamais le texte complet).
Les articles sont accumulés dans data/processed/news_items.pkl : cet historique servira à mesurer,
dans quelques semaines, si ces informations améliorent réellement les prédictions.
"""
from __future__ import annotations

import html
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd

import config as cfg
from src.live_players import _norm

UA = {"User-Agent": "match-predictor/1.0 (usage personnel)"}
FEEDS = {          # média : adresse du flux public
    "BBC Sport": "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "The Guardian": "https://www.theguardian.com/football/rss",
    "Sky Sports": "https://www.skysports.com/rss/12040",
    "L'Équipe": "https://dwh.lequipe.fr/api/edito/rss?path=/Football/",
    "AS": "https://feeds.as.com/mrss-s/pages/as/site/as.com/section/futbol/subsection/primera/",
    "Mundo Deportivo": "https://www.mundodeportivo.com/rss/futbol",
    "La Gazzetta dello Sport": "https://www.gazzetta.it/rss/calcio.xml",
    "Corriere dello Sport": "https://www.corrieredellosport.it/rss/calcio/serie-a",
    "Sportschau": "https://www.sportschau.de/fussball/bundesliga/index~rss2.xml",
    "Foot Mercato": "https://www.footmercato.net/flux-rss",
}
STORE = cfg.PROCESSED_DIR / "news_items.pkl"
KEEP_DAYS = 45

# Expressions (testées sur le texte normalisé : minuscules, sans accents)
RETURN = r"\b(back in training|returns? to training|back from injury|returns?|is back|fit again|available again|reprend l.?entrainement|de retour|fait son retour|retour (?:a|de|en|dans)|rejou\w*|reprend\w*|vuelve|volvera|regresa|recupera(?:do)?|alta medica|rientr\w*|tornat\w* (?:in gruppo|disponibil\w*)|torna (?:in gruppo|disponibile)|recuperat\w*|wieder fit|zuruck im training|comeback|kehrt zuruck)\b"
DOUBT = r"\b(doubt(?:ful)?|fitness (?:test|concern)|race against time|incertain|incertitude|pas certain|en duda|duda|in dubbio|in forse|fraglich|angeschlagen|wackelt)\b"
OUT = (r"\b(injur(?:y|ed|ies)|ruled out|sidelined|out for|will miss|to miss|misses|set to miss|suspended|suspension|ban(?:ned)?|"
       r"hamstring|knee|ankle|acl|surgery|operation|carried off|blesse|blessure|forfait|absent|indisponible|manquer\w*|opere|suspendu|rupture|"
       r"lesion(?:ado)?|baja|operado|sancionado|fractura|rotura|se pierde|se perdera|infortun(?:io|ato)|squalific(?:ato|a)|stop di|"
       r"verletz(?:t|ung)|fallt (?:aus|lange)|ausfall|gesperrt|sperre|operiert|kreuzband|muskelfaser)\b")
LONG = r"\b(surgery|operation|acl|cruciate|months?|season|opere|opere|operation|plusieurs (?:semaines|mois)|fin de saison|fractur\w*|rotura|meses|temporada|mesi|stagione|kreuzband|monate|saison-aus|saisonende)\b"
STRONG = r"\b(ruled out|will miss|surgery|operation|opere|forfait|rupture|fractur\w*|rotura|se pierde|baja para|operado|out for|sidelined|kreuzband|fallt (?:aus|lange)|operiert|squalificato|suspended for|ban of)\b"

# Noms courants des clubs dans la presse (en plus du nom de la base et du nom ESPN)
ALIASES = {
    "Paris SG": ["psg", "paris saint germain", "paris sg"], "Marseille": ["om", "marseille"], "Lyon": ["ol", "lyon"], "Monaco": ["asm", "monaco"],
    "Lille": ["losc", "lille"], "Rennes": ["rennes", "srfc"], "Lens": ["lens", "rcl"], "Nice": ["nice", "ogcn"], "Strasbourg": ["strasbourg", "racing"],
    "Barcelona": ["barca", "barcelona", "barcelone"], "Real Madrid": ["real madrid", "madrid"], "Ath Madrid": ["atletico", "atleti"],
    "Ath Bilbao": ["athletic", "bilbao"], "Sevilla": ["sevilla", "seville"], "Betis": ["betis"], "Sociedad": ["real sociedad", "la real"],
    "Man City": ["man city", "manchester city", "city"], "Man United": ["man utd", "man united", "manchester united", "united"],
    "Tottenham": ["spurs", "tottenham"], "Nott'm Forest": ["forest", "nottingham forest"], "Newcastle": ["newcastle", "magpies"],
    "Inter": ["inter", "nerazzurri"], "Milan": ["milan", "rossoneri"], "Juventus": ["juve", "juventus"], "Napoli": ["napoli", "naples"],
    "Roma": ["roma", "giallorossi"], "Lazio": ["lazio", "biancocelesti"], "Atalanta": ["atalanta", "dea"],
    "Bayern Munich": ["bayern", "fc bayern"], "Dortmund": ["dortmund", "bvb", "borussia dortmund"], "Leverkusen": ["leverkusen", "bayer"],
    "RB Leipzig": ["leipzig", "rb leipzig"], "Ein Frankfurt": ["frankfurt", "eintracht"], "M'gladbach": ["gladbach", "monchengladbach"],
}


def _date(s: str | None) -> pd.Timestamp | None:
    if not s:
        return None
    try:
        return pd.Timestamp(parsedate_to_datetime(s)).tz_convert("UTC")
    except Exception:
        try:
            return pd.Timestamp(s).tz_convert("UTC")
        except Exception:
            return None


def fetch(verbose: bool = False) -> pd.DataFrame:
    """Télécharge les flux, ajoute les nouveaux articles au stock local et renvoie les articles récents."""
    rows = []
    for source, url in FEEDS.items():
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20).read()
            root = ET.fromstring(raw)
        except Exception as exc:                        # un flux indisponible n'empêche pas les autres
            if verbose:
                print(f"  flux {source} indisponible : {type(exc).__name__}")
            continue
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            title = html.unescape(re.sub(r"<[^>]+>", " ", title)).strip()
            desc = html.unescape(re.sub(r"<[^>]+>", " ", it.findtext("description") or "")).strip()
            lede = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0][:280]
            link = (it.findtext("link") or "").strip()
            date = _date(it.findtext("pubDate"))
            if title and link and date is not None:
                rows.append({"source": source, "title": title, "link": link, "date": date,
                             "text": _norm(title + ". " + lede)})   # texte normalisé, gardé en mémoire seulement
        time.sleep(.3)
    new = pd.DataFrame(rows, columns=["source", "title", "link", "date", "text"])
    old = pd.read_pickle(STORE) if STORE.exists() else new.iloc[:0]
    allr = pd.concat([old, new], ignore_index=True).drop_duplicates("link", keep="last")
    cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=KEEP_DAYS)
    allr = allr[allr["date"] >= cut].sort_values("date").reset_index(drop=True)
    allr.to_pickle(STORE)
    if verbose:
        print(f"  actualités : {len(new)} articles lus, {len(allr)} en stock")
    return allr


NEGATION = r"\b(no (?:serious )?injury|not injured|escapes? (?:serious )?injury|scan (?:is )?clear|pas de blessure|plus de peur que de mal|descarta\w* (?:una )?lesion|sin lesion|nessun infortunio|keine (?:schwere )?verletzung|entwarnung|nothing serious|rien de grave|nada grave)\b"
WINDOW = 7          # distance maximale (en mots) entre le nom du joueur et l'expression


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text)


def _positions(words: list[str], pattern: str) -> list[int]:
    """Indices des mots où commence une expression du motif (on teste chaque fenêtre de quatre mots)."""
    rx, out = re.compile(pattern), []
    for i in range(len(words)):
        if rx.match(" ".join(words[i:i + 4])):
            out.append(i)
    return out


def _find(words: list[str], name: list[str]) -> list[tuple[int, int]]:
    n = len(name)
    return [(i, i + n) for i in range(len(words) - n + 1) if words[i:i + n] == name]


# mots qui ouvrent une autre proposition : une expression située au-delà ne se rapporte plus au même joueur
# (« Doué carried off for PSG while Kane vows to return »)
CLAUSE_BREAK = {"while", "but", "whereas", "although", "alors", "tandis", "mais", "mientras", "pero", "aunque", "mentre", "ma", "pero",
                "wahrend", "aber", "doch"}
LIST_WORDS = {"and", "et", "y", "e", "und", "en"}     # « Mendes et Doué absents » : la liste partage l'expression


def _closest(spans: list[tuple[int, int]], i: int, words: list[str]) -> set[tuple[int, int]]:
    """Noms auxquels se rapporte une expression placée au mot i : le nom le plus proche, plus les noms
    coordonnés avec lui (« Hakimi, Mendes et Doué forfaits »). Évite d'attribuer à un joueur la blessure
    d'un coéquipier cité dans la même phrase (« le message de soutien de Kvaratskhelia à Doué après sa blessure »)."""
    if not spans:
        return set()
    dist = lambda sp: i - sp[1] if i >= sp[1] else max(sp[0] - i, 0)
    best = min(dist(sp) for sp in spans)
    chosen = {sp for sp in spans if dist(sp) == best}
    srt = sorted(spans)
    grew = True
    while grew:                                            # propagation le long d'une énumération
        grew = False
        for x, y in zip(srt, srt[1:]):
            if x[1] <= y[0] and (y[0] - x[1] == 0 or (y[0] - x[1] == 1 and words[x[1]] in LIST_WORDS)):
                if (x in chosen) != (y in chosen):
                    chosen |= {x, y}; grew = True
    return chosen


def _linked(words: list[str], a: int, b: int, i: int) -> bool:
    """Aucun mot de rupture de proposition entre le nom (mots a..b) et l'expression au mot i."""
    gap = words[b:i] if i >= b else words[i + 1:a]
    return not CLAUSE_BREAK.intersection(gap)


def extract_absences(items: pd.DataFrame, roster: pd.DataFrame, team_names: dict, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Absences par joueur, à partir du titre et de la première phrase de chaque article.

    `roster` : colonnes team, name, player_id ; `team_names` : {équipe: [noms alternatifs]}.
    Règles (la précision passe avant l'exhaustivité) :
    - le joueur est cité par son nom complet, ou par son nom de famille seul s'il est distinctif ET que
      son club est cité dans l'article ; un nom inclus dans un nom plus long cité (Bernardo / Bernardo Silva) est ignoré ;
    - l'expression d'absence, de doute ou de retour doit se trouver à moins de WINDOW mots du nom ;
    - une négation (« pas de blessure », « descartan una lesión ») annule l'article pour ce joueur.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    roster = roster.dropna(subset=["name"]).copy()
    roster["norm"] = roster["name"].map(_norm)
    roster = roster[roster["norm"].str.len() >= 4].drop_duplicates(["team", "norm"])
    roster["toks"] = roster["norm"].map(_words)
    roster["last"] = roster["toks"].str[-1]
    by_last = roster.groupby("last")["player_id"].nunique()
    aliases = {t: [_words(_norm(a)) for a in {t, *team_names.get(t, []), *ALIASES.get(t, [])} if a] for t in roster["team"].unique()}
    players = list(roster.itertuples())

    hits = []
    for it in items.itertuples():
        words = _words(it.text)
        if not words:
            continue
        teams_in = {t for t, al in aliases.items() if any(_find(words, a) for a in al if a)}
        found = []                                              # (joueur, début, fin)
        echo = set()      # rappels par le seul nom d'un joueur déjà cité en entier : ils « captent » les expressions
                          # voisines (pour ne pas les attribuer à un coéquipier) sans produire d'absence à eux seuls
        for p in players:
            spans = _find(words, p.toks)
            if not spans and len(p.toks) > 1 and len(p.last) >= 5 and p.team in teams_in and by_last.get(p.last, 0) == 1:
                spans = _find(words, [p.last])
            elif spans and len(p.toks) > 1 and len(p.last) >= 3:
                bare = _find(words, [p.last])                  # « le soutien de Kvaratskhelia à Doué après sa blessure »
                echo |= set(bare)
                spans += bare
            found += [(p, a, b) for a, b in spans]
        # un nom contenu dans un nom plus long cité au même endroit désigne l'autre joueur
        found = [(p, a, b) for p, a, b in found if not any(a2 <= a and b <= b2 and (b2 - a2) > (b - a) for _, a2, b2 in found)]
        # même nom pour deux joueurs : on garde celui dont le club est cité, sinon personne
        names = {}
        for p, a, b in found:
            names.setdefault((a, b), []).append(p)
        if not found:
            continue
        neg, back, doubt, out, strong = (_positions(words, rx) for rx in (NEGATION, RETURN, DOUBT, OUT, STRONG))
        spans = list(names)
        is_long = bool(re.search(LONG, it.text))
        for (a, b), cands in names.items():
            if (a, b) in echo:
                continue
            if len(cands) > 1:
                cands = [p for p in cands if p.team in teams_in]
                if len(cands) != 1:
                    continue
            p = cands[0]
            def dist(pos):          # distance de l'expression la plus proche qui se rapporte à ce joueur, sinon None
                d = [i - b if i >= b else a - i for i in pos if a - WINDOW <= i <= b + WINDOW
                     and (a, b) in _closest(spans, i, words) and _linked(words, a, b, i)]
                return min(d) if d else None
            near = lambda pos: dist(pos) is not None
            if near(neg):
                continue
            # plusieurs expressions autour du nom : la plus proche l'emporte (à égalité : retour, doute, absence)
            cand = [(dist(pos), k, st) for k, (st, pos) in enumerate((("back", back), ("doubt", doubt), ("out", out))) if near(pos)]
            status = min(cand)[2] if cand else None
            if status:
                hits.append({"team": p.team, "player": p.name, "player_id": p.player_id, "status": status, "strong": near(strong),
                             "long": is_long, "source": it.source, "title": it.title, "link": it.link, "date": it.date})
    cols = ["team", "player", "player_id", "status", "confidence", "until", "long", "sources"]
    if not hits:
        return pd.DataFrame(columns=cols)
    h = pd.DataFrame(hits).drop_duplicates(["player_id", "link"]).sort_values("date")
    out = []
    for (team, pid), g in h.groupby(["team", "player_id"]):
        last_back = g.loc[g.status == "back", "date"].max()
        live = g[g.status != "back"]
        if pd.notna(last_back):
            live = live[live.date > last_back]                 # un retour annoncé efface les absences antérieures
        if live.empty:
            continue
        outs = live[live.status == "out"]
        is_long = bool(outs["long"].any()) if len(outs) else False
        until = live["date"].max() + pd.Timedelta(days=120 if is_long else 10)
        if until < now:
            continue
        status = "out" if len(outs) else "doubt"
        n_src = outs["source"].nunique()
        confidence = ("confirmée" if n_src >= 2 or outs["strong"].any() else "probable") if status == "out" else "incertain"
        src = live.drop_duplicates("source", keep="last").sort_values("date", ascending=False)
        out.append({"team": team, "player": g["player"].iloc[0], "player_id": pid, "status": status, "confidence": confidence,
                    "until": until, "long": is_long,
                    "sources": [{"source": r.source, "title": r.title, "link": r.link, "date": f"{r.date:%Y-%m-%d}"} for r in src.itertuples()][:4]})
    return pd.DataFrame(out, columns=cols)
