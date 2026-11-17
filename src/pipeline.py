"""Pipeline complet de prédiction : historique -> variables -> clustering -> modèle(s) -> calibration."""
from __future__ import annotations

from difflib import get_close_matches

import numpy as np
import pandas as pd

import config as cfg
from src.clustering import add_cluster_features
from src.data import ODDS_COLS, STAT_COLS, TEAM_ALIASES
from src.features import build_features, select_features
from src.models import MODEL_NAMES, Model, Stacker, TemperatureScaler, WeightedAverage, load_params


class MatchPredictor:
    """Modèle final sauvegardé (joblib). `kind` : 'single', 'ens_weighted' ou 'ens_stacking'."""

    def __init__(self, key: str):
        parts = key.split("|")
        self.key, self.variant = key, parts[0]
        self.kind = parts[1] if parts[1].startswith("ens_") else "single"
        self.model_names = MODEL_NAMES if self.kind != "single" else [parts[1]]

    # -- entraînement ----------------------------------------------------------------------
    def fit(self, feats: pd.DataFrame, oos: pd.DataFrame, elo_params: dict, n_threads: int = -1, stakes: pd.DataFrame | None = None):
        """`feats` : table de variables complète ; `oos` : prédictions hors échantillon passées,
        utilisées pour apprendre la calibration ou la combinaison (jamais les données d'entraînement).
        `stakes` : variables d'enjeu (src/stakes.py) à ajouter ; elles sont alors recalculées à chaque prédiction."""
        from src.evaluation import get
        self.elo_params = elo_params
        self.use_stakes = stakes is not None
        if self.use_stakes:
            feats = feats.merge(stakes, on="match_id", how="left")
        played = feats["y"].notna().to_numpy()
        fold, self.clusterer = add_cluster_features(feats, played)
        self.features = select_features(fold.columns, with_odds=self.variant == "odds")
        train = fold[played & (fold.season > cfg.FIRST_SEASON)]
        self.models = [Model(n, load_params(n, self.variant), self.features, n_threads).fit(train) for n in self.model_names]
        y = oos["y"].to_numpy(dtype=int)
        raw = [get(oos, f"{self.variant}|{n}|raw") for n in self.model_names]
        if self.kind == "single":
            self.combiner = TemperatureScaler().fit(raw[0], y)
        elif self.kind == "ens_weighted":
            self.combiner = WeightedAverage().fit(raw, y)
        else:
            self.combiner = Stacker().fit(raw, y)
        self.train_end = feats.loc[played, "Date"].max()
        self.n_train = len(train)
        return self

    # -- prédiction ------------------------------------------------------------------------
    def predict_from_features(self, fold: pd.DataFrame) -> np.ndarray:
        raw = [m.predict_proba(fold) for m in self.models]
        return self.combiner.transform(raw[0] if self.kind == "single" else raw)

    def fixture_fold(self, fixtures: pd.DataFrame | None, history: pd.DataFrame):
        """Table de variables (historique + matchs à venir), clustering compris.
        Renvoie (table, matchs à venir préparés, historique combiné)."""
        if fixtures is not None and len(fixtures):
            fx = _prepare_fixtures(fixtures, history)
            combined = pd.concat([history, fx], ignore_index=True)
        else:
            fx, combined = history.iloc[:0], history
        combined = combined.sort_values(["Date", "league", "HomeTeam"], kind="mergesort").reset_index(drop=True)
        feats = build_features(combined, self.elo_params)
        if getattr(self, "use_stakes", False):                  # absent des modèles sauvegardés avant l'ajout de l'enjeu
            from src.stakes import build_stakes
            feats = feats.merge(build_stakes(combined), on="match_id", how="left")
        fold, _ = add_cluster_features(feats, np.zeros(len(feats), dtype=bool), clusterer=self.clusterer)
        return fold, fx, combined

    def predict(self, fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        """Probabilités pour des matchs à venir.

        `fixtures` : colonnes date, home_team, away_team (+ odds_h, odds_d, odds_a facultatives).
        `history` : table de matchs nettoyée (data/processed/matches.pkl). Les variables sont
        reconstruites à partir de l'historique strictement antérieur à chaque match.
        """
        fold, fx, _ = self.fixture_fold(fixtures, history)
        rows = fold[fold["match_id"].isin(fx["match_id"])].set_index("match_id").loc[fx["match_id"]].reset_index()
        if self.variant == "odds" and rows["odds_p_h"].isna().any():
            raise ValueError("Le modèle 'avec cotes' exige les colonnes odds_h, odds_d, odds_a pour chaque match.")
        p = self.predict_from_features(rows)
        out = pd.DataFrame({
            "date": rows["Date"].dt.date, "league": rows["league"].map(cfg.LEAGUES),
            "home_team": rows["HomeTeam"], "away_team": rows["AwayTeam"],
            "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2],
            "prediction": np.array(["domicile", "nul", "extérieur"])[p.argmax(axis=1)],
            "elo_home": rows["elo_h"], "elo_away": rows["elo_a"],
        })
        return out


def resolve_team(name: str, known: list[str]) -> str:
    """Associe un nom saisi à un nom d'équipe de l'historique (alias puis correspondance approchée)."""
    name = TEAM_ALIASES.get(str(name).strip(), str(name).strip())
    if name in known:
        return name
    lower = {k.lower(): k for k in known}
    match = get_close_matches(name.lower(), list(lower), n=1, cutoff=0.6)
    if not match:
        raise ValueError(f"Équipe inconnue : {name!r}. Exemples de noms valides : {sorted(known)[:8]} ...")
    return lower[match[0]]


def _prepare_fixtures(fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    fx = fixtures.rename(columns={c: c.strip().lower() for c in fixtures.columns}).copy()
    missing = {"date", "home_team", "away_team"} - set(fx.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier de matchs : {sorted(missing)}")
    known = sorted(set(history["HomeTeam"]) | set(history["AwayTeam"]))
    last_league = history.sort_values("Date").groupby("HomeTeam")["league"].last()
    out = pd.DataFrame(index=range(len(fx)), columns=history.columns)
    out["Date"] = pd.to_datetime(fx["date"].to_numpy(), dayfirst=False, errors="raise")
    out["HomeTeam"] = [resolve_team(t, known) for t in fx["home_team"]]
    out["AwayTeam"] = [resolve_team(t, known) for t in fx["away_team"]]
    out["league"] = out["HomeTeam"].map(last_league).to_numpy()
    bad = out["league"].to_numpy() != out["AwayTeam"].map(last_league).to_numpy()
    if bad.any():
        raise ValueError(f"Équipes de championnats différents : {out.loc[bad, ['HomeTeam', 'AwayTeam']].values.tolist()}")
    out["season"] = np.where(out["Date"].dt.month >= 7, out["Date"].dt.year, out["Date"].dt.year - 1)
    for c in ["FTHG", "FTAG"] + STAT_COLS + ODDS_COLS + ["close_h", "close_d", "close_a"]:
        out[c] = np.nan
    for c in ("odds_h", "odds_d", "odds_a"):
        out[c] = pd.to_numeric(fx[c], errors="coerce").to_numpy() if c in fx.columns else np.nan
    out["FTR"] = np.nan
    out["odds_src"], out["close_src"] = "", ""
    out["match_id"] = history["match_id"].max() + 1 + np.arange(len(out))
    return out.astype({c: history[c].dtype for c in history.columns if c not in ("FTR", "odds_src", "close_src", "HomeTeam", "AwayTeam", "league")}, errors="ignore")
