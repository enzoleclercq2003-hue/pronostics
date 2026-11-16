"""Téléchargement (avec cache), lecture et nettoyage des données de matchs."""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import requests

import config as cfg

RESULT_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
STAT_COLS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]

# Familles de cotes d'avant-match, par ordre de préférence (H, D, A)
PRE_ODDS = [
    ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"), ("BWH", "BWD", "BWA"), ("IWH", "IWD", "IWA"),
    ("WHH", "WHD", "WHA"), ("VCH", "VCD", "VCA"),
]
# Familles de cotes de clôture, par ordre de préférence
CLOSE_ODDS = [
    ("PSCH", "PSCD", "PSCA"), ("B365CH", "B365CD", "B365CA"), ("AvgCH", "AvgCD", "AvgCA"),
]
ODDS_COLS = sorted({c for fam in PRE_ODDS + CLOSE_ODDS for c in fam})

# Harmonisation des noms d'équipes (variantes rencontrées selon les saisons / sources)
TEAM_ALIASES = {
    "Man United": "Man United", "Manchester United": "Man United", "Manchester Utd": "Man United",
    "Manchester City": "Man City", "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton": "Wolves",
    "Newcastle United": "Newcastle", "West Ham United": "West Ham",
    "Nottingham Forest": "Nott'm Forest", "Nottm Forest": "Nott'm Forest",
    "Sporting Gijon": "Sp Gijon", "Athletic Bilbao": "Ath Bilbao", "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid", "Real Sociedad": "Sociedad", "Real Betis": "Betis",
    "Espanyol": "Espanol", "Celta Vigo": "Celta", "Rayo Vallecano": "Vallecano",
    "Deportivo La Coruna": "La Coruna", "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Leverkusen", "Borussia M'gladbach": "M'gladbach",
    "Monchengladbach": "M'gladbach", "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln", "Cologne": "FC Koln", "Koln": "FC Koln",
    "Mainz 05": "Mainz", "Hertha Berlin": "Hertha", "Paris Saint-Germain": "Paris SG",
    "PSG": "Paris SG", "Saint-Etienne": "St Etienne", "AC Milan": "Milan",
    "Internazionale": "Inter", "AS Roma": "Roma", "Hellas Verona": "Verona",
}


def _download(url: str) -> bytes | None:
    """Télécharge une URL ; renvoie None si le fichier est absent ou inaccessible."""
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "football-ml-research/1.0"})
        if r.status_code == 200 and len(r.content) > 200 and b"HomeTeam" in r.content[:2000]:
            return r.content
    except requests.RequestException:
        pass
    return None


def fetch_raw(league: str, season: int, refresh: bool = False) -> tuple[bytes | None, str]:
    """Renvoie le contenu brut d'un fichier (cache -> source principale -> secours).

    La saison en cours est retéléchargée si `refresh` vaut True (elle évolue chaque semaine).
    """
    path = cfg.RAW_DIR / f"{league}_{cfg.season_code(season)}.csv"
    src_path = path.with_suffix(".source")
    if path.exists() and not refresh:
        source = src_path.read_text().strip() if src_path.exists() else "football-data.co.uk"
        return path.read_bytes(), f"cache ({source})"
    content = _download(cfg.PRIMARY_URL.format(season_code=cfg.season_code(season), league=league))
    source = "football-data.co.uk"
    if content is None:
        content = _download(cfg.FALLBACK_URL.format(
            slug=cfg.FALLBACK_SLUGS[league], season_code=cfg.season_code(season)))
        source = "github datasets/football-datasets (secours, sans cotes)"
    if content is None:
        return (path.read_bytes(), "cache") if path.exists() else (None, "indisponible")
    path.write_bytes(content)
    src_path.write_text(source)
    time.sleep(0.2)  # politesse envers le serveur
    return content, source


def _read_csv(content: bytes) -> pd.DataFrame:
    """Lit un CSV en gérant l'encodage (utf-8 puis latin-1) et les lignes mal formées."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", dtype=str, engine="python")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    """Gère les formats jj/mm/aa, jj/mm/aaaa et aaaa-mm-jj."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    for fmt in ("%d/%m/%y", "%Y-%m-%d"):
        miss = out.isna()
        if miss.any():
            out.loc[miss] = pd.to_datetime(s[miss], format=fmt, errors="coerce")
    return out


def clean_one(df: pd.DataFrame, league: str, season: int) -> tuple[pd.DataFrame, list[str]]:
    """Nettoie un fichier saison : colonnes, types, dates, noms d'équipes."""
    df = df.rename(columns={"HT": "HomeTeam", "AT": "AwayTeam"})
    missing = [c for c in RESULT_COLS + STAT_COLS if c not in df.columns]
    keep = [c for c in RESULT_COLS + STAT_COLS + ODDS_COLS if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(how="all")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df["Date"] = _parse_dates(df["Date"])
    df = df.dropna(subset=["Date"])
    for c in ("HomeTeam", "AwayTeam"):
        df[c] = df[c].astype(str).str.strip().replace(TEAM_ALIASES)
    for c in df.columns:
        if c not in ("Date", "HomeTeam", "AwayTeam", "FTR"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in STAT_COLS + ODDS_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()
    df = df[df["FTR"].isin(cfg.CLASSES) & df["FTHG"].notna() & df["FTAG"].notna()]
    # Cotes aberrantes (<= 1) -> manquantes
    for c in ODDS_COLS:
        df.loc[df[c] <= 1.0, c] = np.nan
    df["league"] = league
    df["season"] = season
    return df, missing


def _coalesce_odds(df: pd.DataFrame, families, prefix: str) -> pd.DataFrame:
    """Première famille de cotes complète disponible, par ordre de préférence."""
    out = pd.DataFrame(np.nan, index=df.index, columns=[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"])
    src = pd.Series("", index=df.index, dtype=object)
    for fam in families:
        ok = df[list(fam)].notna().all(axis=1) & out.iloc[:, 0].isna()
        out.loc[ok] = df.loc[ok, list(fam)].to_numpy()
        src[ok] = fam[0][:-1]
    out[f"{prefix}_src"] = src
    return out


def load_matches(refresh_current: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Charge tous les championnats / saisons et renvoie la table nettoyée, triée par date."""
    frames, report = [], []
    for league in cfg.LEAGUES:
        for season in cfg.ALL_SEASONS:
            refresh = refresh_current and season == cfg.CURRENT_SEASON
            content, source = fetch_raw(league, season, refresh=refresh)
            if content is None:
                report.append((league, season, 0, source, "fichier absent"))
                continue
            df, missing = clean_one(_read_csv(content), league, season)
            frames.append(df)
            report.append((league, season, len(df), source, ",".join(missing)))
    rep = pd.DataFrame(report, columns=["league", "season", "n_matches", "source", "missing_cols"])
    rep.to_csv(cfg.RESULTS_DIR / "data_report.csv", index=False)

    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates(subset=["league", "Date", "HomeTeam", "AwayTeam"], keep="first")
    df = pd.concat([df, _coalesce_odds(df, PRE_ODDS, "odds"), _coalesce_odds(df, CLOSE_ODDS, "close")], axis=1)
    df = df.sort_values(["Date", "league", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    if verbose:
        print(f"Doublons supprimés : {n0 - len(df)}")
    return df


def load_processed() -> pd.DataFrame:
    return pd.read_pickle(cfg.PROCESSED_DIR / "matches.pkl")
