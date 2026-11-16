"""Variables au niveau JOUEUR (compositions, valeurs marchandes, forme individuelle, fatigue).

Source : jeu de données ouvert « transfermarkt-datasets » (dcaribou), tables games, game_lineups,
appearances, player_valuations. Couverture : 2012/13 -> 2025/26 (mises à jour arrêtées en juillet 2026).

Deux familles, étiquetées séparément :
- PRE_COLS  : connues plusieurs jours avant le match. Le « onze attendu » est le onze aligné au
              match de championnat précédent ; valeurs, forme et fatigue sont évaluées à la date du match
              avec uniquement des informations strictement antérieures.
- LINEUP_COLS : nécessitent la composition officielle (publiée environ 1 h avant le coup d'envoi).
              Ce n'est pas une fuite, mais ces variables ne sont utilisables qu'au dernier moment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg

TM_DIR = cfg.RAW_DIR.parent / "external" / "transfermarkt"
COMP_TO_LEAGUE = {"GB1": "E0", "ES1": "SP1", "IT1": "I1", "L1": "D1", "FR1": "F1"}
FORM_WINDOW_DAYS, FATIGUE_DAYS, SQUAD_GAMES = 365, 14, 5
MIN_KNOWN = 8          # nombre minimal de titulaires dont la valeur marchande est connue

PRE_BASE = ["pxi_logmv", "top11_logmv", "pxi_ga90", "pxi_min14", "pxi_min365", "rest_all", "games30_all"]
LINEUP_BASE = ["xi_logmv", "xi_ga90", "xi_min14", "xi_min365", "xi_changes", "xi_top11_share", "xi_vs_prev"]


def _cols(base):
    return [f"{side}_{c}" for c in base for side in ("h", "a")] + [f"diff_{c}" for c in base]


PRE_COLS, LINEUP_COLS = _cols(PRE_BASE), _cols(LINEUP_BASE)
# Sous-ensemble maintenable en direct : compositions de championnat + valeurs marchandes seulement
LIVE_PRE_COLS = _cols(["pxi_logmv", "top11_logmv"])
LIVE_LINEUP_COLS = _cols(["xi_logmv", "xi_changes", "xi_top11_share", "xi_vs_prev"])


def _days(s: pd.Series) -> pd.Series:
    return (pd.to_datetime(s) - pd.Timestamp("1970-01-01")).dt.days


def load_tables(max_date: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """Charge les tables utiles ; `max_date` tronque tout (sert au test anti-fuite)."""
    games = pd.read_csv(TM_DIR / "games.csv.gz", usecols=["game_id", "competition_id", "season", "date", "home_club_id",
                                                           "away_club_id", "home_club_goals", "away_club_goals"])
    lineups = pd.read_csv(TM_DIR / "game_lineups.csv.gz", usecols=["date", "game_id", "player_id", "club_id", "type"])
    apps = pd.read_csv(TM_DIR / "appearances.csv.gz", usecols=["game_id", "player_id", "date", "goals", "assists", "minutes_played"])
    vals = pd.read_csv(TM_DIR / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"])
    out = {"games": games, "lineups": lineups, "apps": apps, "vals": vals}
    limit = None if max_date is None else _days(pd.Series([max_date])).iloc[0]
    for name in list(out):
        t = out[name].assign(day=_days(out[name]["date"])).dropna(subset=["day"])
        t["day"] = t["day"].astype("int64")
        out[name] = t if limit is None else t[t["day"] <= limit].copy()
    return out


def map_games(matches: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Relie les matchs Transfermarkt à nos match_id. La correspondance des noms de clubs est
    apprise automatiquement (vote majoritaire sur championnat + date + score)."""
    g = games[games.competition_id.isin(COMP_TO_LEAGUE)].copy()
    g["league"] = g.competition_id.map(COMP_TO_LEAGUE)
    m = matches[["match_id", "league", "season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    m["day"] = _days(m["Date"])
    j = g.merge(m, left_on=["league", "day", "home_club_goals", "away_club_goals"], right_on=["league", "day", "FTHG", "FTAG"])
    votes = pd.concat([j[["home_club_id", "HomeTeam"]].set_axis(["club", "team"], axis=1),
                       j[["away_club_id", "AwayTeam"]].set_axis(["club", "team"], axis=1)])
    club_map = votes.groupby("club")["team"].agg(lambda s: s.value_counts().idxmax())
    g["HomeTeam"], g["AwayTeam"] = g.home_club_id.map(club_map), g.away_club_id.map(club_map)
    out = g.merge(m[["match_id", "league", "season", "HomeTeam", "AwayTeam"]], on=["league", "season", "HomeTeam", "AwayTeam"])
    return out[["match_id", "game_id", "league", "season", "day", "home_club_id", "away_club_id"]].drop_duplicates("match_id")


def _player_stats(rows: pd.DataFrame, apps: pd.DataFrame, vals: pd.DataFrame, freeze_valuations: bool = False) -> pd.DataFrame:
    """Ajoute à des lignes (player_id, day) : valeur marchande, buts+passes et minutes sur 365 j,
    minutes sur 14 j. Tout est calculé sur des dates STRICTEMENT antérieures à `day`."""
    rows = rows.sort_values("day", kind="mergesort").reset_index(drop=True)
    v = vals.sort_values("day", kind="mergesort")[["player_id", "day", "market_value_in_eur"]].rename(columns={"day": "vday"})
    rows["vcut"] = rows["day"]
    if freeze_valuations:
        # Simule la production : dernières valeurs connues = mi-juin précédant la saison
        date = pd.Timestamp("1970-01-01") + pd.to_timedelta(rows["day"], unit="D")
        start_year = np.where(date.dt.month >= 7, date.dt.year, date.dt.year - 1)
        cut = (pd.to_datetime(pd.Series(start_year, index=rows.index).astype(str) + "-06-13") - pd.Timestamp("1970-01-01")).dt.days
        rows["vcut"] = np.minimum(rows["day"], cut)
        rows = rows.sort_values("vcut", kind="mergesort").reset_index(drop=True)
    rows = pd.merge_asof(rows, v, left_on="vcut", right_on="vday", by="player_id", allow_exact_matches=False).drop(columns=["vday", "vcut"])
    rows = rows.sort_values("day", kind="mergesort").reset_index(drop=True)
    a = apps.groupby(["player_id", "day"], as_index=False)[["goals", "assists", "minutes_played"]].sum()
    a = a.sort_values(["player_id", "day"], kind="mergesort")
    a["cum_ga"] = (a.goals + a.assists).groupby(a.player_id).cumsum()
    a["cum_min"] = a.minutes_played.groupby(a.player_id).cumsum()
    a = a.sort_values("day", kind="mergesort")[["player_id", "day", "cum_ga", "cum_min"]].rename(columns={"day": "aday"})
    for lag, tag in ((0, "now"), (FATIGUE_DAYS, "f"), (FORM_WINDOW_DAYS, "y")):
        rows["key"] = rows["day"] - lag
        # lag = 0 : dernière apparition strictement avant le jour du match ; sinon : état au début de la fenêtre
        rows = rows.sort_values("key", kind="mergesort")
        got = pd.merge_asof(rows[["player_id", "key"]], a, left_on="key", right_on="aday", by="player_id",
                            allow_exact_matches=False)
        rows[f"ga_{tag}"], rows[f"min_{tag}"] = got["cum_ga"].fillna(0).to_numpy(), got["cum_min"].fillna(0).to_numpy()
    rows["ga365"], rows["min365"] = rows.ga_now - rows.ga_y, rows.min_now - rows.min_y
    rows["min14"] = rows.min_now - rows.min_f
    return rows.drop(columns=["key", "ga_now", "ga_f", "ga_y", "min_now", "min_f", "min_y"])


def _aggregate(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    grp = rows.groupby(["game_id", "club_id"])
    # Joueur sans valeur connue (jeune, recrue non identifiée) : médiane de ses coéquipiers du onze
    val = rows["market_value_in_eur"]
    # (au moins 8 valeurs connues sur 11, sinon la valeur du onze est déclarée manquante)
    med = grp["market_value_in_eur"].transform("median")
    mv = val.fillna(med).groupby([rows["game_id"], rows["club_id"]]).sum(min_count=1)
    mv = mv.where(grp["market_value_in_eur"].count() >= MIN_KNOWN)
    out = pd.DataFrame({f"{prefix}_logmv": np.log(mv.where(mv > 0)),
                        f"{prefix}_ga90": grp["ga365"].sum() / grp["min365"].sum().clip(lower=900) * 90,
                        f"{prefix}_min14": grp["min14"].mean(),
                        f"{prefix}_min365": grp["min365"].mean()})
    out.loc[grp.size() < 9, :] = np.nan        # composition incomplète : on préfère une valeur manquante
    return out


def build_player_features(matches: pd.DataFrame, tables: dict[str, pd.DataFrame] | None = None,
                          freeze_valuations: bool = False, extra_link: pd.DataFrame | None = None,
                          extra_lineups: pd.DataFrame | None = None) -> pd.DataFrame:
    """Table (match_id + variables joueurs). Matchs hors couverture : valeurs manquantes.

    `extra_link` / `extra_lineups` prolongent les données au-delà du jeu Transfermarkt (saison en
    cours, via ESPN) : mêmes colonnes que `link` (match_id, game_id, day, home_club_id, away_club_id)
    et que la table des compositions (game_id, club_id, player_id, type)."""
    t = tables or load_tables()
    games, lineups, apps, vals = t["games"], t["lineups"], t["apps"], t["vals"]
    link = map_games(matches, games)
    if extra_link is not None and len(extra_link):
        link = pd.concat([link, extra_link[~extra_link.match_id.isin(link.match_id)]], ignore_index=True)
        lineups = pd.concat([lineups[["game_id", "club_id", "player_id", "type"]], extra_lineups], ignore_index=True)

    # Vue « un club, un match de championnat », ordonnée dans le temps
    cg = pd.concat([link[["game_id", "day", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
                    link[["game_id", "day", "away_club_id"]].rename(columns={"away_club_id": "club_id"})])
    cg = cg.sort_values(["club_id", "day"], kind="mergesort").reset_index(drop=True)
    cg["k"] = cg.groupby("club_id").cumcount()

    lu = lineups[lineups.game_id.isin(link.game_id)][["game_id", "club_id", "player_id", "type"]]
    lu = lu.merge(cg[["game_id", "club_id", "k"]], on=["game_id", "club_id"])
    starters = lu[lu["type"] == "starting_lineup"]

    def shifted(src: pd.DataFrame, lag: int) -> pd.DataFrame:
        """Joueurs du match k-lag du club, rattachés au match k (évalués ensuite à la date du match k)."""
        s = src[["club_id", "k", "player_id"]].assign(k=src["k"] + lag)
        return s.merge(cg, on=["club_id", "k"])[["game_id", "club_id", "day", "player_id"]]

    xi = starters.merge(cg[["game_id", "club_id", "day"]], on=["game_id", "club_id"])[["game_id", "club_id", "day", "player_id"]]
    pxi = shifted(starters, 1)
    squad = pd.concat([shifted(lu, lag) for lag in range(1, SQUAD_GAMES + 1)]).drop_duplicates(["game_id", "club_id", "player_id"])

    xi, pxi, squad = (_player_stats(r, apps, vals, freeze_valuations) for r in (xi, pxi, squad))
    feat = cg[["game_id", "club_id", "day"]].set_index(["game_id", "club_id"])
    feat = feat.join(_aggregate(pxi, "pxi")).join(_aggregate(xi, "xi"))

    # Les 11 joueurs les plus chers de l'effectif récent (qualité de l'effectif, composition non requise)
    squad = squad.sort_values("market_value_in_eur", ascending=False)
    top = squad.groupby(["game_id", "club_id"]).head(11)
    top_grp = top.groupby(["game_id", "club_id"])["market_value_in_eur"]
    top_sum = top_grp.sum(min_count=1).where(top_grp.count() >= 11)
    feat["top11_logmv"] = np.log(top_sum.where(top_sum > 0))
    in_xi = top.merge(xi[["game_id", "club_id", "player_id"]], on=["game_id", "club_id", "player_id"])
    feat["xi_top11_share"] = (in_xi.groupby(["game_id", "club_id"])["market_value_in_eur"].sum() / top_sum).reindex(feat.index)
    feat.loc[feat["xi_logmv"].notna() & feat["top11_logmv"].notna() & feat["xi_top11_share"].isna(), "xi_top11_share"] = 0.0
    kept = xi.merge(pxi[["game_id", "club_id", "player_id"]], on=["game_id", "club_id", "player_id"])
    feat["xi_changes"] = 11 - kept.groupby(["game_id", "club_id"]).size().reindex(feat.index).fillna(0)
    feat.loc[feat["xi_logmv"].isna() | feat["pxi_logmv"].isna(), "xi_changes"] = np.nan
    feat["xi_vs_prev"] = feat["xi_logmv"] - feat["pxi_logmv"]

    # Repos réel, toutes compétitions confondues (coupes nationales et européennes comprises)
    clubs = pd.unique(cg.club_id)
    allg = pd.concat([games[["day", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
                      games[["day", "away_club_id"]].rename(columns={"away_club_id": "club_id"})])
    allg = allg[allg.club_id.isin(clubs)].drop_duplicates().sort_values(["club_id", "day"])
    by_club = {c: d.to_numpy() for c, d in allg.groupby("club_id")["day"]}
    rest, n30 = [], []
    for (_, club), day in zip(feat.index, feat["day"].to_numpy()):
        d = by_club.get(club, np.array([], dtype=int))
        i = np.searchsorted(d, day, "left")
        rest.append(min(day - d[i - 1], 21) if i > 0 else np.nan)
        n30.append(i - np.searchsorted(d, day - 30, "left"))
    feat["rest_all"], feat["games30_all"] = rest, n30

    feat = feat.drop(columns="day").reset_index()
    base = PRE_BASE + LINEUP_BASE
    home = link[["match_id", "game_id", "home_club_id"]].merge(feat, left_on=["game_id", "home_club_id"], right_on=["game_id", "club_id"])
    away = link[["match_id", "game_id", "away_club_id"]].merge(feat, left_on=["game_id", "away_club_id"], right_on=["game_id", "club_id"])
    out = home[["match_id"] + base].rename(columns={c: f"h_{c}" for c in base}).merge(
        away[["match_id"] + base].rename(columns={c: f"a_{c}" for c in base}), on="match_id")
    for c in base:
        out[f"diff_{c}"] = out[f"h_{c}"] - out[f"a_{c}"]
    return matches[["match_id"]].merge(out, on="match_id", how="left")[["match_id"] + PRE_COLS + LINEUP_COLS]
