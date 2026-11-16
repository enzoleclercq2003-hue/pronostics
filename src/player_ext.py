"""Variables joueurs étendues (expérience, cf. scripts/14) : indisponibilités, postes, âge, banc.

Il n'existe pas de source gratuite de blessures avec historique. On les approche par ce que les
feuilles de match permettent de savoir AVANT le coup d'envoi :
- absents      : joueurs importants de l'effectif récent absents de la dernière feuille de match
                 (blessure, suspension en cours, mise à l'écart) ;
- suspendus    : carton rouge au match précédent, ou cinquième / dixième... carton jaune de la saison ;
- postes       : valeur marchande du onze attendu par ligne (attaque, milieu, défense) et confrontation
                 attaque contre défense adverse ;
- divers       : âge moyen du onze attendu, valeur du banc, stabilité du onze.

Tout est calculé à partir des matchs de championnat strictement antérieurs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.player_features import MIN_KNOWN, SQUAD_GAMES, TM_DIR, _days, _player_stats, load_tables, map_games

ABS_BASE = ["abs1_share", "abs2_share", "susp_share", "out_share"]
POS_BASE = ["att_logmv", "mid_logmv", "def_logmv"]
MISC_BASE = ["pxi_age", "bench_logmv", "pxi_stab"]


def _cols(base):
    return [f"{side}_{c}" for c in base for side in ("h", "a")] + [f"diff_{c}" for c in base]


ABS_COLS, MISC_COLS = _cols(ABS_BASE), _cols(MISC_BASE)
POS_COLS = _cols(POS_BASE) + ["mu_h", "mu_a", "mu_diff"]
EXT_COLS = ABS_COLS + POS_COLS + MISC_COLS


def position_group(pos: pd.Series) -> pd.Series:
    """Poste détaillé (Transfermarkt) -> ligne : att / mid / def (gardien compris)."""
    p = pos.fillna("").str.lower()
    out = pd.Series("mid", index=pos.index)
    out[p.str.contains("forward|winger|striker|attack")] = "att"
    out[p.str.contains("back|goalkeeper|sweeper|defen")] = "def"
    out[p.str.contains("midfield")] = "mid"
    out[p == ""] = np.nan
    return out


def build_ext_features(matches: pd.DataFrame, tables: dict | None = None, freeze_valuations: bool = True) -> pd.DataFrame:
    t = tables or load_tables()
    games, apps, vals = t["games"], t["apps"], t["vals"]
    lineups = pd.read_csv(TM_DIR / "game_lineups.csv.gz", usecols=["game_id", "player_id", "club_id", "type", "position"])
    cards = pd.read_csv(TM_DIR / "appearances.csv.gz", usecols=["game_id", "player_id", "yellow_cards", "red_cards"])
    dob = pd.read_csv(TM_DIR / "players.csv.gz", usecols=["player_id", "date_of_birth"])
    dob["dob_day"] = _days(dob["date_of_birth"])
    link = map_games(matches, games)

    cg = pd.concat([link[["game_id", "day", "season", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
                    link[["game_id", "day", "season", "away_club_id"]].rename(columns={"away_club_id": "club_id"})])
    cg = cg.sort_values(["club_id", "day"], kind="mergesort").reset_index(drop=True)
    cg["k"] = cg.groupby("club_id").cumcount()
    lu = lineups[lineups.game_id.isin(link.game_id)].merge(cg[["game_id", "club_id", "k", "season"]], on=["game_id", "club_id"])
    lu["starter"] = lu["type"] == "starting_lineup"
    lu["line"] = position_group(lu["position"])

    # Suspensions : rouge au match, ou jaune portant le total de la saison à un multiple de 5
    c = cards.merge(lu[["game_id", "player_id", "club_id", "k", "season"]], on=["game_id", "player_id"])
    c = c.sort_values(["player_id", "season", "k"], kind="mergesort")
    c["cum_y"] = c.groupby(["player_id", "season"])["yellow_cards"].cumsum()
    c["banned_next"] = (c.red_cards > 0) | ((c.yellow_cards > 0) & (c.cum_y % 5 == 0))
    lu = lu.merge(c[["game_id", "player_id", "banned_next"]], on=["game_id", "player_id"], how="left")
    lu["banned_next"] = lu["banned_next"].fillna(False).astype(bool)

    def shifted(src: pd.DataFrame, lag: int, extra=()) -> pd.DataFrame:
        """Lignes du match k-lag d'un club, rattachées à son match k (date du match k)."""
        s = src[["club_id", "k", "player_id", *extra]].assign(k=src["k"] + lag)
        return s.merge(cg[["game_id", "club_id", "k", "day"]], on=["club_id", "k"]).drop(columns="k")

    key = ["game_id", "club_id"]
    value = lambda rows: _player_stats(rows, apps, vals, freeze_valuations)[[*rows.columns, "market_value_in_eur"]]

    # Onze attendu (titulaires du match précédent), banc précédent, effectif récent
    pxi = value(shifted(lu[lu.starter], 1, ("line",)))
    bench = value(shifted(lu[~lu.starter], 1))
    squad = pd.concat([shifted(lu, lag) for lag in range(1, SQUAD_GAMES + 1)]).drop_duplicates(key + ["player_id"])
    squad = value(squad).sort_values("market_value_in_eur", ascending=False)
    top = squad.groupby(key).head(11).copy()
    top_grp = top.groupby(key)["market_value_in_eur"]
    top_sum = top_grp.sum(min_count=1).where(top_grp.count() >= 11)

    prev1 = shifted(lu, 1, ("banned_next",)).drop(columns="day")
    prev2 = shifted(lu, 2).drop(columns="day").assign(in2=True)
    top = top.merge(prev1.assign(in1=True), on=key + ["player_id"], how="left").merge(prev2, on=key + ["player_id"], how="left")
    top["in1"], top["in2"] = top["in1"].fillna(False).astype(bool), top["in2"].fillna(False).astype(bool)
    top["banned_next"] = top["banned_next"].fillna(False).astype(bool)
    share = lambda mask: (top.loc[mask].groupby(key)["market_value_in_eur"].sum() / top_sum).reindex(top_sum.index).fillna(0.0).where(top_sum.notna())

    feat = cg.set_index(key)[["day"]]
    feat["abs1_share"] = share(~top.in1)
    feat["abs2_share"] = share(~top.in1 & ~top.in2)
    feat["susp_share"] = share(top.banned_next)
    feat["out_share"] = feat["abs1_share"] + feat["susp_share"]

    # Valeur par ligne du onze attendu
    known = pxi.groupby(key)["market_value_in_eur"].count()
    for line in ("att", "mid", "def"):
        s = pxi[pxi.line == line].groupby(key)["market_value_in_eur"].sum(min_count=1)
        feat[f"{line}_logmv"] = np.log(s.where(s > 0)).reindex(feat.index).where(known.reindex(feat.index) >= MIN_KNOWN)

    # Âge moyen, banc, stabilité du onze
    age = pxi.merge(dob[["player_id", "dob_day"]], on="player_id", how="left")
    age["age"] = (age["day"] - age["dob_day"]) / 365.25
    feat["pxi_age"] = age.groupby(key)["age"].mean().where(age.groupby(key)["age"].count() >= MIN_KNOWN)
    b = bench.groupby(key)["market_value_in_eur"]
    feat["bench_logmv"] = np.log(b.sum(min_count=1).where(b.count() >= 4))
    s1 = shifted(lu[lu.starter], 1)[key + ["player_id"]]
    s2 = shifted(lu[lu.starter], 2)[key + ["player_id"]]
    feat["pxi_stab"] = s1.merge(s2, on=key + ["player_id"]).groupby(key).size().reindex(feat.index)
    has2 = s2.groupby(key).size().reindex(feat.index).notna()
    feat["pxi_stab"] = feat["pxi_stab"].fillna(0).where(has2)

    feat = feat.drop(columns="day").reset_index()
    base = ABS_BASE + POS_BASE + MISC_BASE
    home = link[["match_id", "game_id", "home_club_id"]].merge(feat, left_on=["game_id", "home_club_id"], right_on=key)
    away = link[["match_id", "game_id", "away_club_id"]].merge(feat, left_on=["game_id", "away_club_id"], right_on=key)
    out = home[["match_id"] + base].rename(columns={c: f"h_{c}" for c in base}).merge(
        away[["match_id"] + base].rename(columns={c: f"a_{c}" for c in base}), on="match_id")
    for c in base:
        out[f"diff_{c}"] = out[f"h_{c}"] - out[f"a_{c}"]
    out["mu_h"] = out["h_att_logmv"] - out["a_def_logmv"]          # attaque domicile contre défense extérieur
    out["mu_a"] = out["a_att_logmv"] - out["h_def_logmv"]
    out["mu_diff"] = out["mu_h"] - out["mu_a"]
    return matches[["match_id"]].merge(out, on="match_id", how="left")[["match_id"] + EXT_COLS]
