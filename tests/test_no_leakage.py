"""Tests automatiques anti-fuite.

Test 1 (troncature) : on supprime toutes les données postérieures à un match, on recalcule
les variables et on vérifie qu'elles sont identiques à celles calculées sur l'historique complet.
Test 2 (masquage) : on efface en plus le résultat et les statistiques du match lui-même
(et de tous les matchs du même jour) ; ses variables ne doivent pas changer.
Test 3 : même principe pour les variables de clustering (ajustées sur le pli d'entraînement).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import config as cfg
from src.clustering import add_cluster_features
from src.data import STAT_COLS, load_processed
from src.features import META_COLS, build_features

N_SAMPLES = 12
RESULT_STATS = ["FTHG", "FTAG", "FTR"] + STAT_COLS


@pytest.fixture(scope="module")
def matches():
    return load_processed()


@pytest.fixture(scope="module")
def full_features(matches):
    return build_features(matches).set_index("match_id")


def _sample_ids(matches):
    rng = np.random.default_rng(cfg.SEED)
    eligible = matches[matches.season >= cfg.FIRST_SEASON + 1]
    return rng.choice(eligible["match_id"].to_numpy(), size=N_SAMPLES, replace=False)


def _feature_cols(feats):
    return [c for c in feats.columns if c not in META_COLS and c != "y" and not c.startswith("odds_")]


def _assert_same(full_row, new_row, cols, label):
    a, b = full_row[cols].to_numpy(dtype=float), new_row[cols].to_numpy(dtype=float)
    same = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
    assert same.all(), f"{label} : variables différentes -> {[c for c, s in zip(cols, same) if not s][:10]}"


def test_truncation(matches, full_features):
    """Supprimer le futur ne change aucune variable."""
    cols = _feature_cols(full_features)
    for mid in _sample_ids(matches):
        date = matches.loc[matches.match_id == mid, "Date"].iloc[0]
        truncated = build_features(matches[matches.Date <= date]).set_index("match_id")
        _assert_same(full_features.loc[mid], truncated.loc[mid], cols, f"match {mid} ({date.date()})")


def test_own_match_not_used(matches, full_features):
    """Effacer le résultat et les stats du match (et du jour) ne change aucune variable."""
    cols = _feature_cols(full_features)
    for mid in _sample_ids(matches)[:6]:
        date = matches.loc[matches.match_id == mid, "Date"].iloc[0]
        masked = matches[matches.Date <= date].copy()
        masked.loc[masked.Date == date, RESULT_STATS] = np.nan
        feats = build_features(masked).set_index("match_id")
        _assert_same(full_features.loc[mid], feats.loc[mid], cols, f"match {mid} masqué")


def test_cluster_features(matches, full_features):
    """Les variables de clustering d'un match ne dépendent pas des matchs postérieurs."""
    feats = full_features.reset_index()
    test_season = cfg.WALK_FORWARD_SEASONS[0]
    train_mask = (feats.season < test_season).to_numpy()
    full = add_cluster_features(feats, train_mask, k=4)[0].set_index("match_id")
    cl_cols = [c for c in full.columns if c.startswith("cl_")]
    assert cl_cols, "aucune variable de clustering produite"
    ids = feats.loc[feats.season == test_season, "match_id"].to_numpy()
    for mid in np.random.default_rng(cfg.SEED).choice(ids, size=5, replace=False):
        date = full.loc[mid, "Date"]
        keep = (feats.Date <= date).to_numpy()
        sub = feats[keep].copy()
        sub.loc[sub.Date == date, ["FTHG", "FTAG", "FTR", "y"]] = np.nan
        part = add_cluster_features(sub, train_mask[keep], k=4)[0].set_index("match_id")
        _assert_same(full.loc[mid], part.loc[mid], cl_cols, f"clustering, match {mid}")


def test_extra_features(matches):
    """Pi-ratings et forces sur tirs cadrés : identiques après suppression du futur et masquage du jour."""
    from src.extra_features import build_extra_features
    full = build_extra_features(matches).set_index("match_id")
    cols = list(full.columns)
    for mid in _sample_ids(matches)[:5]:
        date = matches.loc[matches.match_id == mid, "Date"].iloc[0]
        masked = matches[matches.Date <= date].copy()
        masked.loc[masked.Date == date, RESULT_STATS] = np.nan
        part = build_extra_features(masked).set_index("match_id")
        _assert_same(full.loc[mid], part.loc[mid], cols, f"variables supplémentaires, match {mid}")


def test_player_features(matches):
    """Variables joueurs : identiques quand on supprime tout ce qui est postérieur au match et les
    feuilles de match du jour même (seule la composition du match reste, connue avant le coup d'envoi)."""
    from src.player_features import TM_DIR, build_player_features, load_tables
    if not (TM_DIR / "games.csv.gz").exists():
        pytest.skip("données joueurs non téléchargées")
    full = build_player_features(matches).set_index("match_id")
    cols = list(full.columns)
    eligible = matches[(matches.season >= 2015) & (matches.season <= 2025)]
    for mid in np.random.default_rng(cfg.SEED).choice(eligible["match_id"].to_numpy(), size=3, replace=False):
        date = matches.loc[matches.match_id == mid, "Date"].iloc[0]
        tables = load_tables(max_date=date)
        day = tables["games"]["day"].max()
        tables["apps"] = tables["apps"][tables["apps"]["day"] < day]
        tables["vals"] = tables["vals"][tables["vals"]["day"] < day]
        part = build_player_features(matches[matches.Date <= date], tables).set_index("match_id")
        _assert_same(full.loc[mid], part.loc[mid], cols, f"variables joueurs, match {mid}")


def test_no_match_stats_in_features(full_features):
    """Aucune colonne brute du match (buts, tirs, corners, cartons) parmi les variables."""
    cols = set(_feature_cols(full_features))
    assert not (cols & set(RESULT_STATS)), "statistiques du match présentes dans les variables"


def test_stakes(matches):
    """Variables d'enjeu : identiques après suppression du futur et masquage des résultats du jour."""
    from src.stakes import build_stakes
    full = build_stakes(matches).set_index("match_id")
    cols = list(full.columns)
    for mid in _sample_ids(matches)[:6]:
        date = matches.loc[matches.match_id == mid, "Date"].iloc[0]
        masked = matches[matches.Date <= date].copy()
        masked.loc[masked.Date == date, RESULT_STATS] = np.nan
        part = build_stakes(masked).set_index("match_id")
        _assert_same(full.loc[mid], part.loc[mid], cols, f"enjeu, match {mid}")


def test_formations(matches, full_features):
    """Variables de formation : identiques après suppression du futur et masquage des résultats du jour."""
    from src.formations import build_formation_features
    from src.player_features import TM_DIR
    if not (TM_DIR / "games.csv.gz").exists():
        pytest.skip("données Transfermarkt non téléchargées")
    feats = full_features.reset_index()
    full = build_formation_features(matches, feats).set_index("match_id")
    eligible = matches[matches.season.between(2015, 2025)]
    for mid in np.random.default_rng(cfg.SEED).choice(eligible["match_id"].to_numpy(), size=4, replace=False):
        date = matches.loc[matches.match_id == mid, "Date"].iloc[0]
        masked = matches[matches.Date <= date].copy()
        masked.loc[masked.Date == date, RESULT_STATS] = np.nan
        part = build_formation_features(masked, feats).set_index("match_id")
        _assert_same(full.loc[mid], part.loc[mid], list(full.columns), f"formations, match {mid}")
