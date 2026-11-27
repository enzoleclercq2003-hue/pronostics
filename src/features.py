"""Construction des variables d'avant-match.

RÈGLE : chaque variable d'un match n'utilise que des matchs strictement antérieurs.
Toutes les statistiques sont décalées d'un match (shift) avant agrégation ; les lignes sans
résultat (matchs à venir) reçoivent des variables mais n'alimentent aucun historique.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.elo import compute_elo, load_elo_params
from src.odds import implied_proportional
from src.poisson import DC_COLS, rolling_dixon_coles

EWM_STATS = ["pts", "gf", "ga", "shots", "shots_ag", "sot", "sot_ag", "corners", "fouls", "cards"]
ROLL_STATS = ["pts", "gf", "ga"]
VENUE_STATS = ["pts", "gf", "ga"]
PROFILE_STATS = ["shots", "sot", "gf", "ga", "corners", "fouls", "cards", "win"]
META_COLS = ["match_id", "Date", "league", "season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
             "odds_h", "odds_d", "odds_a", "close_h", "close_d", "close_a"]


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (équipe, match), triée par équipe puis par date."""
    def side(team, opp, gf, ga, s, s_ag, st, st_ag, c, f, y, r, is_home):
        return pd.DataFrame({
            "match_id": df["match_id"].to_numpy(), "date": df["Date"].to_numpy(),
            "league": df["league"].to_numpy(), "season": df["season"].to_numpy(),
            "team": df[team].to_numpy(), "is_home": is_home,
            "gf": df[gf].to_numpy(dtype=float), "ga": df[ga].to_numpy(dtype=float),
            "shots": df[s].to_numpy(dtype=float), "shots_ag": df[s_ag].to_numpy(dtype=float),
            "sot": df[st].to_numpy(dtype=float), "sot_ag": df[st_ag].to_numpy(dtype=float),
            "corners": df[c].to_numpy(dtype=float), "fouls": df[f].to_numpy(dtype=float),
            "cards": df[y].to_numpy(dtype=float) + df[r].to_numpy(dtype=float),
        })
    home = side("HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "HF", "HY", "HR", 1)
    away = side("AwayTeam", "HomeTeam", "FTAG", "FTHG", "AS", "HS", "AST", "HST", "AC", "AF", "AY", "AR", 0)
    long = pd.concat([home, away], ignore_index=True)
    played = long["gf"].notna() & long["ga"].notna()
    long["played"] = played.astype(int)
    long["pts"] = np.where(~played, np.nan, np.where(long.gf > long.ga, 3.0, np.where(long.gf == long.ga, 1.0, 0.0)))
    long["win"] = np.where(~played, np.nan, (long.gf > long.ga).astype(float))
    return long.sort_values(["team", "date"], kind="mergesort").reset_index(drop=True)


def _team_features(long: pd.DataFrame) -> pd.DataFrame:
    """Variables par (équipe, match) calculées sur les matchs précédents de l'équipe."""
    feats = long[["match_id", "is_home"]].copy()
    team = long["team"]
    shifted = long.groupby("team", sort=False)[EWM_STATS + ["win"]].shift(1)   # exclut le match courant

    # Forme : moyennes à pondération exponentielle (demi-vie ~5 matchs)
    ewm = shifted[EWM_STATS].groupby(team, sort=False).ewm(halflife=cfg.EWM_HALFLIFE).mean()
    ewm = ewm.reset_index(level=0, drop=True)
    for c in EWM_STATS:
        feats[f"ewm_{c}"] = ewm[c]

    # Forme : fenêtres glissantes de 3, 5 et 10 matchs
    for w in cfg.ROLL_WINDOWS:
        roll = shifted[ROLL_STATS].groupby(team, sort=False).rolling(w, min_periods=1).mean()
        roll = roll.reset_index(level=0, drop=True)
        for c in ROLL_STATS:
            feats[f"roll{w}_{c}"] = roll[c]

    # Forme spécifique au lieu (matchs à domicile pour l'hôte, à l'extérieur pour le visiteur)
    keys = [long["team"], long["is_home"]]
    v_shift = long.groupby(["team", "is_home"], sort=False)[VENUE_STATS].shift(1)
    v_ewm = v_shift.groupby(keys, sort=False).ewm(halflife=cfg.EWM_HALFLIFE).mean().reset_index(level=[0, 1], drop=True)
    v_roll = v_shift.groupby(keys, sort=False).rolling(5, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    for c in VENUE_STATS:
        feats[f"venue_ewm_{c}"] = v_ewm[c]
    feats["venue_roll5_pts"] = v_roll["pts"]

    # Profil de style (pour le clustering) : moyennes sur les 38 derniers matchs
    prof = shifted[PROFILE_STATS].groupby(team, sort=False).rolling(
        cfg.CLUSTER_PROFILE_WINDOW, min_periods=cfg.CLUSTER_MIN_MATCHES).mean().reset_index(level=0, drop=True)
    for c in PROFILE_STATS:
        feats[f"prof_{c}"] = prof[c]

    # Repos : jours depuis le dernier match (plafonnés) et matchs sur les 14 derniers jours
    days = (long["date"] - pd.Timestamp("1970-01-01")).dt.days
    feats["rest_days"] = (days - days.groupby(team, sort=False).shift(1)).clip(upper=30)

    def _count_14d(d: pd.Series) -> pd.Series:
        arr = d.to_numpy()
        return pd.Series(np.arange(len(arr)) - np.searchsorted(arr, arr - 14, "left"), index=d.index)
    feats["matches_14d"] = days.groupby(team, sort=False).transform(_count_14d).astype(float)

    # Saison en cours : journée, points et différence de buts par match (avant la rencontre)
    grp = long.groupby(["team", "season"], sort=False)
    played_before = grp["played"].cumsum() - long["played"]
    feats["matchday"] = played_before + 1.0
    for src, name in (("pts", "season_ppg"), ("gd", "season_gdpg")):
        val = (long["gf"] - long["ga"]) if src == "gd" else long["pts"]
        val = val.fillna(0.0)
        cum_before = val.groupby([long["team"], long["season"]], sort=False).cumsum() - val
        feats[name] = np.where(played_before > 0, cum_before / played_before.replace(0, np.nan), np.nan)

    # Promu : équipe absente de la division la saison précédente
    present = set(zip(long["league"], long["season"], long["team"]))
    first = long["season"].min()
    feats["promoted"] = [
        0.0 if s == first else float((lg, s - 1, t) not in present)
        for lg, s, t in zip(long["league"], long["season"], long["team"])
    ]
    return feats


def _rank_and_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Classement avant la rencontre et confrontations directes récentes (boucle chronologique)."""
    n = len(df)
    rank_h, rank_a = np.full(n, np.nan), np.full(n, np.nan)
    h2h_n, h2h_pts, h2h_gd = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    homes, aways = df["HomeTeam"].to_numpy(), df["AwayTeam"].to_numpy()
    hg, ag = df["FTHG"].to_numpy(dtype=float), df["FTAG"].to_numpy(dtype=float)
    dates = df["Date"].to_numpy()

    # Classement : état de la table avant la date du match (les matchs du jour sont exclus)
    for _, idx in df.groupby(["league", "season"], sort=False).indices.items():
        # équipe -> [points, diff. de buts, buts pour] ; une équipe entre dans la table le jour de son premier
        # match (sans quoi la liste des équipes de toute la saison, connue seulement après coup, serait utilisée)
        table: dict[str, list[float]] = {}
        n_played = 0
        start = 0
        while start < len(idx):
            end = start
            while end < len(idx) and dates[idx[end]] == dates[idx[start]]:
                end += 1
            day = idx[start:end]
            for i in day:
                table.setdefault(homes[i], [0.0, 0.0, 0.0]); table.setdefault(aways[i], [0.0, 0.0, 0.0])
            if n_played > 0:
                # égalité parfaite départagée par le nom : ordre reproductible d'une exécution à l'autre
                order = sorted(table, key=lambda t: (-table[t][0], -table[t][1], -table[t][2], t))
                pos = {t: r + 1 for r, t in enumerate(order)}
                for i in day:
                    rank_h[i], rank_a[i] = pos[homes[i]], pos[aways[i]]
            for i in day:
                if np.isnan(hg[i]) or np.isnan(ag[i]):
                    continue
                th, ta = table[homes[i]], table[aways[i]]
                th[0] += 3 if hg[i] > ag[i] else (1 if hg[i] == ag[i] else 0)
                ta[0] += 3 if ag[i] > hg[i] else (1 if hg[i] == ag[i] else 0)
                th[1] += hg[i] - ag[i]; ta[1] += ag[i] - hg[i]
                th[2] += hg[i]; ta[2] += ag[i]
                n_played += 1
            start = end

    # Confrontations directes : 6 dernières rencontres, vues du côté de l'équipe à domicile
    history: dict[tuple, list] = {}
    for i in range(n):
        key = (homes[i], aways[i]) if homes[i] < aways[i] else (aways[i], homes[i])
        past = history.get(key, [])[-6:]
        if past:
            sign = 1.0 if key[0] == homes[i] else -1.0           # stocké du point de vue de key[0]
            gds = np.array([sign * g for g in past])
            h2h_n[i] = len(past)
            h2h_gd[i] = gds.mean()
            h2h_pts[i] = np.where(gds > 0, 3.0, np.where(gds == 0, 1.0, 0.0)).mean()
        if not (np.isnan(hg[i]) or np.isnan(ag[i])):
            gd_key0 = (hg[i] - ag[i]) if key[0] == homes[i] else (ag[i] - hg[i])
            history.setdefault(key, []).append(gd_key0)
    return pd.DataFrame({"rank_h": rank_h, "rank_a": rank_a, "rank_diff": rank_h - rank_a,
                         "h2h_n": h2h_n, "h2h_pts_h": h2h_pts, "h2h_gd_h": h2h_gd}, index=df.index)


def build_features(df: pd.DataFrame, elo_params: dict | None = None) -> pd.DataFrame:
    """Construit la table de variables (une ligne par match) à partir de la table de matchs.

    `df` : sortie de `load_matches`, éventuellement complétée de matchs à venir (résultats NaN).
    """
    df = df.sort_values(["Date", "league", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    elo_params = elo_params or load_elo_params()
    out = df[META_COLS].copy()
    out["y"] = df["FTR"].map({c: i for i, c in enumerate(cfg.CLASSES)})

    # Variables par équipe -> colonnes h_*, a_* et diff_*
    tf = _team_features(_to_long(df))
    team_cols = [c for c in tf.columns if c not in ("match_id", "is_home")]
    home = tf[tf.is_home == 1].set_index("match_id")[team_cols].add_prefix("h_")
    away = tf[tf.is_home == 0].set_index("match_id")[team_cols].add_prefix("a_")
    out = out.join(home, on="match_id").join(away, on="match_id")
    diffs = {f"diff_{c}": out[f"h_{c}"] - out[f"a_{c}"] for c in team_cols if not c.startswith("prof_")}
    out = pd.concat([out, pd.DataFrame(diffs, index=out.index)], axis=1)

    out = pd.concat([out, compute_elo(df, **elo_params), rolling_dixon_coles(df), _rank_and_h2h(df)], axis=1)

    # Championnat (one-hot) et saison
    for lg in cfg.LEAGUES:
        out[f"league_{lg}"] = (df["league"] == lg).astype(float)
    out["season_num"] = df["season"].astype(float)

    # Cotes d'avant-match -> probabilités implicites (uniquement pour la variante "avec cotes")
    odds = df[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    ok = ~np.isnan(odds).any(axis=1)
    probs = np.full_like(odds, np.nan)
    probs[ok] = implied_proportional(odds[ok])
    out["odds_p_h"], out["odds_p_d"], out["odds_p_a"] = probs.T
    out["odds_overround"] = (1.0 / odds).sum(axis=1) - 1.0
    return out.copy()


# --- Groupes de variables ---------------------------------------------------
def feature_groups(columns) -> dict[str, list[str]]:
    """Répartit les colonnes en groupes (pour les ablations)."""
    cols = list(columns)
    groups = {
        "elo": [c for c in cols if c.startswith("elo_")],
        "dc": [c for c in cols if c in DC_COLS],
        "odds": [c for c in cols if c.startswith("odds_p_") or c == "odds_overround"],
        "cluster": [c for c in cols if c.startswith("cl_")],
        "league": [c for c in cols if c.startswith("league_")],
    }
    used = set(sum(groups.values(), []))
    excluded = set(META_COLS) | {"y"} | {c for c in cols if "prof_" in c}
    groups["base"] = [c for c in cols if c not in used and c not in excluded]
    return groups


def select_features(columns, with_odds=False, with_cluster=True, with_elo=True) -> list[str]:
    g = feature_groups(columns)
    feats = g["base"] + g["dc"] + g["league"]
    if with_elo:
        feats += g["elo"]
    if with_cluster:
        feats += g["cluster"]
    if with_odds:
        feats += g["odds"]
    return feats
