"""Plis chronologiques : chargement des variables + clustering réajusté sur chaque pli d'entraînement."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.clustering import add_cluster_features, load_k

FOLD_DIR = cfg.PROCESSED_DIR / "folds"
FOLD_DIR.mkdir(exist_ok=True)
BURN_IN_SEASONS = 1      # la première saison sert uniquement à initialiser les historiques


def load_features() -> pd.DataFrame:
    return pd.read_pickle(cfg.PROCESSED_DIR / "features.pkl")


def load_fold(test_season: int, method: str = "kmeans", use_cache: bool = True) -> pd.DataFrame:
    """Table complète avec variables de clustering ajustées sur les saisons < test_season."""
    path = FOLD_DIR / f"{method}_{test_season}.pkl"
    if use_cache and path.exists():
        return pd.read_pickle(path)
    feats = load_features()
    out, _ = add_cluster_features(feats, (feats.season < test_season).to_numpy(), k=load_k(), method=method)
    out.to_pickle(path)
    return out


def split(fold: pd.DataFrame, test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entraînement : saisons strictement antérieures. Test : la saison N (matchs joués, cotes connues)."""
    played = fold["y"].notna()
    train = fold[played & (fold.season >= cfg.FIRST_SEASON + BURN_IN_SEASONS) & (fold.season < test_season)]
    test = fold[played & (fold.season == test_season) & fold["odds_h"].notna()]
    return train, test
