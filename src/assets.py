"""Ressources visuelles de l'application : icône (logo) et écussons des équipes.

- `make_icon()` dessine le logo (anneau en trois segments, comme les trois issues d'un match)
  et l'enregistre en PNG et ICO dans app/.
- `ensure_team_logos()` télécharge une fois les écussons des équipes de la saison en cours
  (CDN d'ESPN, déjà utilisé pour les compositions) dans app/logos/, en version claire et sombre.
"""
from __future__ import annotations

import io
import json
import urllib.request

import pandas as pd
from PIL import Image, ImageDraw

import config as cfg
from src.espn import CACHE, HEADERS, SLUGS

APP_DIR = cfg.ROOT_DIR / "app" if hasattr(cfg, "ROOT_DIR") else cfg.RAW_DIR.parent.parent / "app"
LOGO_DIR = APP_DIR / "logos"
# Segments de l'anneau : (angle de départ, étendue) en degrés, sens horaire depuis le haut
SEGMENTS = [(-90, 170), (112, 56), (200, 38)]
GRADIENT = [(0x29, 0x97, 0xFF), (0x8F, 0x7B, 0xFF), (0xD8, 0x6B, 0xFF)]     # du bas vers le haut


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _vertical_gradient(size: int, colors, top: float, bottom: float) -> Image.Image:
    """Dégradé vertical : colors[0] en bas (y = bottom), colors[-1] en haut (y = top)."""
    img = Image.new("RGB", (1, size))
    for y in range(size):
        t = min(max((bottom - y) / (bottom - top), 0), 1) * (len(colors) - 1)
        i = min(int(t), len(colors) - 2)
        img.putpixel((0, y), _lerp(colors[i], colors[i + 1], t - i))
    return img.resize((size, size))


def _arc_mask(size: int, radius: float, width: float, start: float, extent: float) -> Image.Image:
    """Masque d'un arc à bouts arrondis (angles en degrés, sens horaire depuis 3 h, comme en SVG)."""
    import math
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    c = size / 2
    d.arc([c - radius - width / 2, c - radius - width / 2, c + radius + width / 2, c + radius + width / 2],
          start, start + extent, fill=255, width=round(width))
    for ang in (start, start + extent):
        x, y = c + radius * math.cos(math.radians(ang)), c + radius * math.sin(math.radians(ang))
        d.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=255)
    return mask


def make_icon(size: int = 1024) -> Image.Image:
    """Logo de l'application : carré arrondi sombre, anneau en trois segments."""
    s = size
    bg = _vertical_gradient(s, [(5, 5, 8), (26, 26, 32)], 0, s).convert("RGBA")
    shape = Image.new("L", (s, s), 0)
    ImageDraw.Draw(shape).rounded_rectangle([0, 0, s - 1, s - 1], radius=round(0.225 * s), fill=255)
    icon = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    icon.paste(bg, (0, 0), shape)
    radius, width = 0.30 * s, 0.115 * s
    grad = _vertical_gradient(s, GRADIENT, s / 2 - radius - width / 2, s / 2 + radius + width / 2).convert("RGBA")
    icon.paste(grad, (0, 0), _arc_mask(s, radius, width, *SEGMENTS[0]))
    for (start, extent), alpha in zip(SEGMENTS[1:], (0.55, 0.26)):
        white = Image.new("RGBA", (s, s), (255, 255, 255, 255))
        mask = _arc_mask(s, radius, width, start, extent).point(lambda v, a=alpha: round(v * a))
        icon.paste(white, (0, 0), mask)
    APP_DIR.mkdir(exist_ok=True)
    icon.resize((256, 256), Image.LANCZOS).save(APP_DIR / "icon.png")
    icon.resize((256, 256), Image.LANCZOS).save(APP_DIR / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return icon


def _espn_teams() -> pd.DataFrame:
    """(championnat, nom ESPN, identifiant, logo) et événements joués, lus dans le cache des calendriers ESPN."""
    events = []
    for league, slug in SLUGS.items():
        for path in (CACHE / slug).glob("scoreboard_*.json"):
            for e in json.loads(path.read_text(encoding="utf-8")).get("events", []):
                comp = {c["homeAway"]: c for c in e["competitions"][0]["competitors"]}
                done = bool(e["status"]["type"].get("completed"))
                events.append({"league": league, "Date": pd.Timestamp(e["date"]).tz_localize(None).normalize(), "completed": done,
                               "espn_home": comp["home"]["team"]["displayName"], "espn_away": comp["away"]["team"]["displayName"],
                               "home_id": comp["home"]["team"]["id"], "away_id": comp["away"]["team"]["id"],
                               "hg": float(comp["home"].get("score") or 0) if done else None,
                               "ag": float(comp["away"].get("score") or 0) if done else None})
    return pd.DataFrame(events)


# Couleurs historiques des clubs les plus connus (principale, secondaire, motif), saisies à la main :
# celles de la source automatique sont parfois approximatives. Motifs : band (écharpe), stripes (rayures), half (mi-parti).
COLOR_OVERRIDES = {
    "Arsenal": ("#ef0107", "#ffffff", "band"), "Aston Villa": ("#670e36", "#95bfe5", "band"), "Brighton": ("#0057b8", "#ffffff", "stripes"),
    "Chelsea": ("#034694", "#ffffff", "band"), "Crystal Palace": ("#1b458f", "#c4122e", "stripes"), "Everton": ("#003399", "#ffffff", "band"),
    "Liverpool": ("#c8102e", "#f6eb61", "band"), "Man City": ("#6cabdd", "#ffffff", "band"), "Man United": ("#da291c", "#000000", "band"),
    "Newcastle": ("#ffffff", "#241f20", "stripes"), "Tottenham": ("#ffffff", "#132257", "band"), "Leeds": ("#ffffff", "#1d428a", "band"),
    "Barcelona": ("#a50044", "#004d98", "stripes"), "Real Madrid": ("#ffffff", "#febe10", "band"), "Ath Madrid": ("#ffffff", "#cb3524", "stripes"),
    "Ath Bilbao": ("#ffffff", "#ee2523", "stripes"), "Sevilla": ("#ffffff", "#d4021d", "band"), "Betis": ("#ffffff", "#00954c", "stripes"),
    "Sociedad": ("#ffffff", "#0067b1", "stripes"), "Valencia": ("#ffffff", "#ee3524", "band"), "Villarreal": ("#ffe667", "#005187", "band"),
    "Celta": ("#8ac3ee", "#ffffff", "band"), "Espanol": ("#ffffff", "#007fc8", "stripes"),
    "Juventus": ("#ffffff", "#000000", "stripes"), "Inter": ("#0068a8", "#000000", "stripes"), "Milan": ("#fb090b", "#000000", "stripes"),
    "Napoli": ("#12a0d7", "#ffffff", "band"), "Roma": ("#8e1f2f", "#f0bc42", "band"), "Lazio": ("#87d8f7", "#ffffff", "band"),
    "Fiorentina": ("#482e92", "#ffffff", "band"), "Atalanta": ("#1e71b8", "#000000", "stripes"), "Torino": ("#8a1e03", "#ffffff", "band"),
    "Bologna": ("#a21c26", "#1a2f48", "stripes"), "Genoa": ("#ad1919", "#002d5a", "half"), "Udinese": ("#ffffff", "#000000", "stripes"),
    "Bayern Munich": ("#dc052d", "#ffffff", "band"), "Dortmund": ("#fde100", "#000000", "band"), "Leverkusen": ("#e32221", "#000000", "band"),
    "RB Leipzig": ("#ffffff", "#dd0741", "band"), "Stuttgart": ("#ffffff", "#e32219", "band"), "Ein Frankfurt": ("#000000", "#e1000f", "band"),
    "M'gladbach": ("#ffffff", "#1d9053", "band"), "Werder Bremen": ("#1d9053", "#ffffff", "band"), "Freiburg": ("#e2001a", "#000000", "band"),
    "Hoffenheim": ("#1961b5", "#ffffff", "band"), "Hamburg": ("#ffffff", "#0a3f86", "band"), "Schalke 04": ("#004d9d", "#ffffff", "band"),
    "FC Koln": ("#ffffff", "#ed1c24", "band"), "Union Berlin": ("#eb1923", "#ffd700", "band"), "Mainz": ("#c3141e", "#ffffff", "band"),
    "Paris SG": ("#004170", "#da291c", "band"), "Marseille": ("#ffffff", "#2faee0", "band"), "Lyon": ("#ffffff", "#da001a", "band"),
    "Monaco": ("#e30613", "#ffffff", "half"), "Lille": ("#e01e13", "#20325f", "band"), "Nice": ("#d6001c", "#000000", "stripes"),
    "Rennes": ("#e13327", "#000000", "band"), "Lens": ("#fff200", "#ec1c24", "band"), "Strasbourg": ("#009fe3", "#ffffff", "band"),
    "Toulouse": ("#5a2d82", "#ffffff", "band"), "Brest": ("#e30613", "#ffffff", "band"),
}


def export_team_colors(history: pd.DataFrame) -> dict:
    """Écrit app/team_colors.json : {équipe: [couleur principale, couleur secondaire]} (codes hexadécimaux).
    Sert aux pastilles du site public, qui n'affiche aucun écusson. Les couleurs d'un club ne sont pas
    protégées comme l'est son logo. À relancer en début de saison pour les promus."""
    from src.live_players import _team_votes
    events = _espn_teams()
    names = _team_votes(events, history[history["FTR"].notna()])
    colors = {}
    for league, slug in SLUGS.items():
        for path in (CACHE / slug).glob("scoreboard_*.json"):
            for e in json.loads(path.read_text(encoding="utf-8")).get("events", []):
                for c in e["competitions"][0]["competitors"]:
                    team = names.get((league, c["team"]["displayName"]))
                    if team and c["team"].get("color"):
                        colors[team] = ["#" + c["team"]["color"].lower(), "#" + (c["team"].get("alternateColor") or "ffffff").lower()]
    path = APP_DIR / "team_colors.json"
    known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    known.update({t: [p, s, "band"] for t, (p, s) in colors.items()})
    known.update({t: list(v) for t, v in COLOR_OVERRIDES.items()})     # couleurs et motifs vérifiés à la main
    path.write_text(json.dumps(dict(sorted(known.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    return known


def _download(url: str) -> bytes | None:
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=20).read()
    except Exception:
        return None


def ensure_team_logos(history: pd.DataFrame, size: int = 112) -> dict:
    """Télécharge les écussons manquants et renvoie {équipe: {"light": fichier, "dark": fichier}}."""
    from src.live_players import _team_votes
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    index_path = LOGO_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    events = _espn_teams()
    if events.empty:
        return index
    names = _team_votes(events, history[history["FTR"].notna()])            # (championnat, nom ESPN) -> notre nom
    ids = {**dict(zip(zip(events.league, events.espn_home), events.home_id)),
           **dict(zip(zip(events.league, events.espn_away), events.away_id))}
    for key, team in names.items():
        if team in index:
            continue
        entry = {}
        for theme, folder in (("light", "500"), ("dark", "500-dark")):
            raw = _download(f"https://a.espncdn.com/i/teamlogos/soccer/{folder}/{ids[key]}.png")
            if raw is None:
                continue
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGBA")
            except Exception:
                continue
            img.thumbnail((size, size), Image.LANCZOS)
            name = f"{ids[key]}{'-dark' if theme == 'dark' else ''}.png"
            img.save(LOGO_DIR / name)
            entry[theme] = name
        if entry:
            entry.setdefault("dark", entry.get("light"))
            entry.setdefault("light", entry.get("dark"))
            index[team] = entry
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return index
