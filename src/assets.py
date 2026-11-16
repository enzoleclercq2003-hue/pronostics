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
