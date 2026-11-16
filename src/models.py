"""Modèles, imputation, calibration et ensembles."""
from __future__ import annotations

import copy
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import config as cfg

MODEL_NAMES = ["logreg", "rf", "et", "hgb", "lgbm", "xgb", "catboost", "mlp"]
MODEL_LABELS = {
    "logreg": "Régression logistique", "rf": "Random Forest", "et": "ExtraTrees",
    "hgb": "HistGradientBoosting", "lgbm": "LightGBM", "xgb": "XGBoost",
    "catboost": "CatBoost", "mlp": "MLP",
}
EARLY_STOP_MODELS = {"hgb", "lgbm", "xgb", "catboost", "mlp"}
MAX_ITER = {"hgb": 1500, "lgbm": 3000, "xgb": 3000, "catboost": 2000, "mlp": 50}
PATIENCE = {"hgb": 40, "lgbm": 60, "xgb": 60, "catboost": 60, "mlp": 6}
PARAMS_DIR = cfg.PROCESSED_DIR / "best_params"
PARAMS_DIR.mkdir(exist_ok=True)


# --- Imputation par la moyenne du championnat ---------------------------------------------
class LeagueMeanImputer:
    """Remplace les valeurs manquantes par la moyenne du championnat (apprise sur l'entraînement)."""

    def fit(self, X: pd.DataFrame):
        self.columns_ = list(X.columns)
        self.league_cols_ = [c for c in self.columns_ if c.startswith("league_")]
        self.global_ = X.mean().fillna(0.0)
        self.means_ = {}
        for lc in self.league_cols_:
            sub = X[X[lc] == 1.0]
            self.means_[lc] = sub.mean().fillna(self.global_) if len(sub) else self.global_
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.columns_]
        out = X.to_numpy(dtype=float, copy=True)
        for lc in self.league_cols_:
            rows = (X[lc] == 1.0).to_numpy()
            if rows.any():
                block = out[rows]
                nan = np.isnan(block)
                if nan.any():
                    block[nan] = np.broadcast_to(self.means_[lc].to_numpy(), block.shape)[nan]
                    out[rows] = block
        nan = np.isnan(out)
        if nan.any():
            out[nan] = np.broadcast_to(self.global_.to_numpy(), out.shape)[nan]
        return out


# --- Fabrique d'estimateurs ---------------------------------------------------------------
def _build(name: str, params: dict, n_iter: int | None, n_threads: int):
    p = dict(params)
    seed = cfg.SEED
    if name == "logreg":
        return LogisticRegression(C=p["C"], max_iter=3000)
    if name in ("rf", "et"):
        cls = RandomForestClassifier if name == "rf" else ExtraTreesClassifier
        return cls(n_estimators=p.get("n_estimators", 500), max_depth=p["max_depth"],
                   min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                   max_samples=p.get("max_samples"), bootstrap=True, n_jobs=n_threads, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=n_iter or MAX_ITER[name], learning_rate=p["learning_rate"], max_leaf_nodes=p["max_leaf_nodes"],
            min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
            max_features=p["max_features"], early_stopping=n_iter is None, n_iter_no_change=PATIENCE[name],
            random_state=seed)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="multiclass", n_estimators=n_iter or MAX_ITER[name], learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"], max_depth=p["max_depth"], min_child_samples=p["min_child_samples"],
            colsample_bytree=p["colsample_bytree"], subsample=p["subsample"], subsample_freq=1,
            reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"], random_state=seed, n_jobs=n_threads, verbose=-1)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="multi:softprob", tree_method="hist", n_estimators=n_iter or MAX_ITER[name],
            learning_rate=p["learning_rate"], max_depth=p["max_depth"], min_child_weight=p["min_child_weight"],
            subsample=p["subsample"], colsample_bytree=p["colsample_bytree"], reg_alpha=p["reg_alpha"],
            reg_lambda=p["reg_lambda"], gamma=p["gamma"], eval_metric="mlogloss",
            early_stopping_rounds=PATIENCE[name] if n_iter is None else None,
            random_state=seed, n_jobs=n_threads)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            loss_function="MultiClass", iterations=n_iter or MAX_ITER[name], learning_rate=p["learning_rate"],
            depth=p["depth"], l2_leaf_reg=p["l2_leaf_reg"], rsm=p["rsm"], border_count=64,
            random_seed=seed, thread_count=n_threads, verbose=0, allow_writing_files=False)
    if name == "mlp":
        hidden = tuple([p["units"]] * p["n_layers"])
        return MLPClassifier(hidden_layer_sizes=hidden, alpha=p["alpha"], learning_rate_init=p["learning_rate_init"],
                             batch_size=p["batch_size"], random_state=seed)
    raise ValueError(name)


class Model:
    """Imputation + (standardisation) + estimateur, avec arrêt précoce chronologique optionnel."""

    def __init__(self, name: str, params: dict, features: list[str], n_threads: int = -1):
        self.name, self.params, self.features, self.n_threads = name, dict(params), list(features), n_threads
        self.best_iter_ = None

    # -- interface DataFrame --
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        self.imputer_ = LeagueMeanImputer().fit(train[self.features])
        Xv = self.imputer_.transform(val[self.features]) if val is not None else None
        yv = val["y"].to_numpy(dtype=int) if val is not None else None
        return self.fit_arrays(self.imputer_.transform(train[self.features]), train["y"].to_numpy(dtype=int), Xv, yv)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba_arrays(self.imputer_.transform(df[self.features]))

    # -- interface tableaux (données déjà imputées) --
    def fit_arrays(self, X, y, X_val=None, y_val=None):
        """Avec jeu de validation : arrêt précoce. Sans : nombre d'itérations fixé (params['n_iter'])."""
        name = self.name
        early = X_val is not None and name in EARLY_STOP_MODELS
        n_iter = None if early else self.params.get("n_iter")
        self.scaler_ = None
        if name in ("logreg", "mlp"):
            self.scaler_ = StandardScaler().fit(X)
            X = self.scaler_.transform(X)
            X_val = self.scaler_.transform(X_val) if X_val is not None else None
        est = _build(name, self.params, n_iter, self.n_threads)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "mlp":
                est = self._fit_mlp(est, X, y, X_val if early else None, y_val, n_iter)
            elif name == "logreg":
                # OpenBLAS multi-thread devient pathologiquement lent au-delà d'une certaine largeur (0,2 s -> 90 s)
                from threadpoolctl import threadpool_limits
                with threadpool_limits(limits=1):
                    est.fit(X, y)
            elif not early:
                est.fit(X, y)
            elif name == "hgb":
                est.fit(X, y, X_val=X_val, y_val=y_val)
                self.best_iter_ = int(max(est.n_iter_ - PATIENCE[name], 1))
            elif name == "lgbm":
                import lightgbm as lgb
                est.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(PATIENCE[name], verbose=False)])
                self.best_iter_ = int(est.best_iteration_)
            elif name == "xgb":
                est.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
                self.best_iter_ = int(est.best_iteration + 1)
            elif name == "catboost":
                est.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=PATIENCE[name], use_best_model=True)
                self.best_iter_ = int(est.get_best_iteration() + 1)
        self.est_ = est
        return self

    def _fit_mlp(self, est, X, y, X_val, y_val, n_iter):
        classes = np.array([0, 1, 2])
        if X_val is None:
            for _ in range(n_iter or 20):
                est.partial_fit(X, y, classes=classes)
            return est
        best, best_loss, wait = None, np.inf, 0
        for epoch in range(MAX_ITER["mlp"]):
            est.partial_fit(X, y, classes=classes)
            p = np.clip(est.predict_proba(X_val), 1e-12, 1)
            loss = -np.log(p[np.arange(len(y_val)), y_val]).mean()
            if loss < best_loss - 1e-5:
                best, best_loss, wait, self.best_iter_ = copy.deepcopy(est), loss, 0, epoch + 1
            else:
                wait += 1
                if wait >= PATIENCE["mlp"]:
                    break
        return best

    def predict_proba_arrays(self, X) -> np.ndarray:
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self.est_.predict_proba(X), dtype=float)


def load_params(name: str, variant: str = "no_odds") -> dict:
    """Hyperparamètres optimisés par Optuna. La variante "avec cotes" réutilise ceux de la
    variante "sans cotes" quand elle n'a pas été réglée séparément."""
    path = PARAMS_DIR / f"{name}_{variant}.json"
    if not path.exists():
        path = PARAMS_DIR / f"{name}_no_odds.json"
    return json.loads(path.read_text())["params"]


# --- Calibration --------------------------------------------------------------------------
class TemperatureScaler:
    """Calibration par température : p' ∝ p^(1/T). Un seul paramètre, donc robuste."""

    def fit(self, p: np.ndarray, y: np.ndarray):
        logp = np.log(np.clip(p, 1e-12, 1))
        y = np.asarray(y, dtype=int)

        def nll(log_t):
            z = logp / np.exp(log_t)
            z = z - z.max(axis=1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            return -lp[np.arange(len(y)), y].mean()
        self.t_ = float(np.exp(minimize_scalar(nll, bounds=(-1.5, 1.5), method="bounded").x))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        z = np.log(np.clip(p, 1e-12, 1)) / self.t_
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)


class IsotonicCalibrator:
    """Calibration isotonique un-contre-tous, puis renormalisation."""

    def fit(self, p: np.ndarray, y: np.ndarray):
        self.iso_ = [IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(p[:, j], (y == j).astype(float))
                     for j in range(3)]
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        q = np.column_stack([iso.predict(p[:, j]) for j, iso in enumerate(self.iso_)])
        return q / q.sum(axis=1, keepdims=True)


# --- Ensembles ----------------------------------------------------------------------------
class WeightedAverage:
    """Moyenne pondérée des probabilités ; poids (>= 0, somme 1) minimisant la log loss hors échantillon."""

    def fit(self, probs: list[np.ndarray], y: np.ndarray):
        P = np.stack(probs)                                    # (modèles, matchs, 3)
        y = np.asarray(y, dtype=int)
        rows = np.arange(len(y))

        def nll(theta):
            w = np.exp(theta - theta.max()); w /= w.sum()
            mix = np.tensordot(w, P, axes=1)
            return -np.log(np.clip(mix[rows, y], 1e-12, 1)).mean()
        res = minimize(nll, np.zeros(len(probs)), method="L-BFGS-B")
        w = np.exp(res.x - res.x.max())
        self.weights_ = w / w.sum()
        return self

    def transform(self, probs: list[np.ndarray]) -> np.ndarray:
        return np.tensordot(self.weights_, np.stack(probs), axes=1)


class Stacker:
    """Stacking : régression logistique multinomiale sur les log-probabilités hors échantillon."""

    def __init__(self, C: float = 0.02):
        self.C = C

    @staticmethod
    def _design(probs):
        return np.hstack([np.log(np.clip(p, 1e-6, 1)) for p in probs])

    def fit(self, probs: list[np.ndarray], y: np.ndarray):
        Z = self._design(probs)
        self.scaler_ = StandardScaler().fit(Z)
        self.meta_ = LogisticRegression(C=self.C, max_iter=3000).fit(self.scaler_.transform(Z), y)
        return self

    def transform(self, probs: list[np.ndarray]) -> np.ndarray:
        return self.meta_.predict_proba(self.scaler_.transform(self._design(probs)))
