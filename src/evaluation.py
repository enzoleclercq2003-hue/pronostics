"""Évaluation walk-forward : prédictions hors échantillon, calibration, ensembles, tableaux."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config as cfg
from src.features import select_features
from src.folds import load_fold, split
from src.metrics import bootstrap_ci, diebold_mariano, evaluate, paired_bootstrap, per_match_losses
from src.models import (MODEL_LABELS, MODEL_NAMES, IsotonicCalibrator, Model, Stacker, TemperatureScaler,
                        WeightedAverage, load_params)
from src.odds import implied_proportional, implied_shin

import warnings
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)   # ajout de colonnes une à une : sans incidence ici

OOS_DIR = cfg.PROCESSED_DIR / "oos"
OOS_DIR.mkdir(exist_ok=True)
VARIANTS = {"no_odds": "sans cotes", "odds": "avec cotes"}
META = ["match_id", "Date", "league", "season", "HomeTeam", "AwayTeam", "y",
        "odds_h", "odds_d", "odds_a", "close_h", "close_d", "close_a"]
SUFFIX = ["H", "D", "A"]


def _put(df: pd.DataFrame, name: str, p: np.ndarray):
    for j, s in enumerate(SUFFIX):
        df[f"{name}|{s}"] = p[:, j]


def get(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[[f"{name}|{s}" for s in SUFFIX]].to_numpy(dtype=float)


def baselines(train: pd.DataFrame, test: pd.DataFrame, out: pd.DataFrame):
    """Références : fréquences (toujours domicile), Elo seul, Dixon-Coles, bookmakers."""
    freq = np.bincount(train["y"].astype(int), minlength=3) / len(train)
    _put(out, "base|home", np.tile(freq, (len(test), 1)))
    lr = LogisticRegression(C=1e3, max_iter=1000).fit(train[["elo_diff"]].to_numpy() / 100, train["y"].astype(int))
    _put(out, "base|elo", lr.predict_proba(test[["elo_diff"]].to_numpy() / 100))
    dc = test[["dc_p_h", "dc_p_d", "dc_p_a"]].to_numpy(dtype=float).copy()
    dc[np.isnan(dc).any(axis=1)] = freq
    _put(out, "base|dixon_coles", dc)
    odds = test[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    _put(out, "book|proportional", implied_proportional(odds))
    _put(out, "book|shin", implied_shin(odds))
    close = test[["close_h", "close_d", "close_a"]].to_numpy(dtype=float)
    ok = ~np.isnan(close).any(axis=1)
    pc = np.full_like(close, np.nan)
    if ok.any():
        pc[ok] = implied_shin(close[ok])
    _put(out, "book|closing_shin", pc)


def oos_for_season(season: int, n_threads: int = -1, verbose: bool = True) -> pd.DataFrame:
    """Entraîne tous les modèles sur les saisons < `season` et prédit la saison `season` (avec cache)."""
    path = OOS_DIR / f"oos_{season}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    fold = load_fold(season)
    train, test = split(fold, season)
    out = test[META].copy()
    baselines(train, test, out)
    for variant in VARIANTS:
        feats = select_features(fold.columns, with_odds=variant == "odds")
        for name in MODEL_NAMES:
            model = Model(name, load_params(name, variant), feats, n_threads).fit(train)
            _put(out, f"{variant}|{name}|raw", model.predict_proba(test))
        if verbose:
            print(f"  saison {cfg.season_label(season)} [{VARIANTS[variant]}] : {len(train)} matchs d'entraînement, "
                  f"{len(test)} de test, {len(feats)} variables", flush=True)
    out = out.copy()
    out.to_pickle(path)
    return out


def post_process(oos: pd.DataFrame, target_seasons: list[int]) -> pd.DataFrame:
    """Calibration et ensembles pour chaque saison cible, appris sur les prédictions hors
    échantillon des saisons STRICTEMENT antérieures (jeu de calibration distinct et postérieur
    à l'entraînement des modèles qui l'ont produit)."""
    parts = []
    for season in target_seasons:
        prior, cur = oos[oos.season < season], oos[oos.season == season].copy()
        y_prior = prior["y"].to_numpy(dtype=int)
        for variant in VARIANTS:
            for name in MODEL_NAMES:
                key = f"{variant}|{name}"
                p_prior, p_cur = get(prior, f"{key}|raw"), get(cur, f"{key}|raw")
                _put(cur, f"{key}|temp", TemperatureScaler().fit(p_prior, y_prior).transform(p_cur))
                _put(cur, f"{key}|iso", IsotonicCalibrator().fit(p_prior, y_prior).transform(p_cur))
            base_prior = [get(prior, f"{variant}|{m}|raw") for m in MODEL_NAMES]
            base_cur = [get(cur, f"{variant}|{m}|raw") for m in MODEL_NAMES]
            wavg = WeightedAverage().fit(base_prior, y_prior)
            _put(cur, f"{variant}|ens_weighted", wavg.transform(base_cur))
            _put(cur, f"{variant}|ens_stacking", Stacker().fit(base_prior, y_prior).transform(base_cur))
            for m, w in zip(MODEL_NAMES, wavg.weights_):
                cur[f"w|{variant}|{m}"] = w
        parts.append(cur.copy())
    return pd.concat(parts, ignore_index=True)


def prediction_catalog() -> list[tuple[str, str, str]]:
    """(clé, libellé, famille) de toutes les prédictions à comparer."""
    cat = [("base|home", "Toujours victoire domicile (fréquences)", "référence"),
           ("base|elo", "Elo seul", "référence"),
           ("base|dixon_coles", "Dixon-Coles (modèle de buts)", "référence"),
           ("book|proportional", "Bookmakers B365 (proportionnel)", "bookmakers"),
           ("book|shin", "Bookmakers B365 (Shin)", "bookmakers"),
           ("book|closing_shin", "Bookmakers clôture (Shin)", "bookmakers")]
    for variant, vlabel in VARIANTS.items():
        for m in MODEL_NAMES:
            cat.append((f"{variant}|{m}|temp", f"{MODEL_LABELS[m]} (calibré)", vlabel))
        cat.append((f"{variant}|ens_weighted", "Ensemble : moyenne pondérée", vlabel))
        cat.append((f"{variant}|ens_stacking", "Ensemble : stacking", vlabel))
    return cat


def comparison_table(pred: pd.DataFrame, reference: str = "book|shin") -> pd.DataFrame:
    """Tableau comparatif sur exactement les mêmes matchs, avec IC bootstrap et tests appariés."""
    pred = pred[pred[f"book|closing_shin|H"].notna()].reset_index(drop=True)
    y = pred["y"].to_numpy(dtype=int)
    ref_losses = per_match_losses(y, get(pred, reference))
    rows = []
    for key, label, family in prediction_catalog():
        p = get(pred, key)
        m, losses = evaluate(y, p), per_match_losses(y, p)
        row = {"clé": key, "modèle": label, "famille": family, **m}
        for metric in ("accuracy", "logloss", "brier", "rps"):
            lo, hi = bootstrap_ci(losses[metric])
            row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = lo, hi
        # par saison : moyenne et écart-type
        by_season = [evaluate(y[idx], p[idx]) for idx in pred.groupby("season").indices.values()]
        for metric in ("accuracy", "logloss"):
            vals = [s[metric] for s in by_season]
            row[f"{metric}_season_mean"], row[f"{metric}_season_std"] = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if key != reference:
            d, lo, hi, pval = paired_bootstrap(losses["logloss"], ref_losses["logloss"])
            dm_stat, dm_p = diebold_mariano(losses["logloss"], ref_losses["logloss"])
            row.update({"dlogloss_vs_book": d, "dlogloss_ci_lo": lo, "dlogloss_ci_hi": hi,
                        "p_bootstrap": pval, "dm_stat": dm_stat, "p_dm": dm_p})
        rows.append(row)
    return pd.DataFrame(rows)


def breakdown(pred: pd.DataFrame, keys: list[str], by: str) -> pd.DataFrame:
    """Métriques par saison ou par championnat pour quelques prédictions."""
    pred = pred[pred[f"book|closing_shin|H"].notna()]
    rows = []
    for value, grp in pred.groupby(by):
        y = grp["y"].to_numpy(dtype=int)
        for key in keys:
            m = evaluate(y, get(grp, key))
            rows.append({by: value, "clé": key, "accuracy": m["accuracy"], "logloss": m["logloss"],
                         "brier": m["brier"], "rps": m["rps"], "f1_D": m["f1_D"], "n": m["n"]})
    return pd.DataFrame(rows)
