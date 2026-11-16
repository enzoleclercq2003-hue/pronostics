"""Clustering des styles d'équipes (K-means, GMM, hiérarchique), sans fuite.

Le clustering n'est PAS un classifieur : il profile le style des équipes à partir de leurs
statistiques moyennes sur leurs matchs PASSÉS. Le modèle de clustering est ajusté uniquement
sur le pli d'entraînement ; les taux de résultats entre styles sont cumulés dans le passé.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config as cfg
from src.features import PROFILE_STATS

H_PROF = [f"h_prof_{c}" for c in PROFILE_STATS]
A_PROF = [f"a_prof_{c}" for c in PROFILE_STATS]
RATE_PRIOR_WEIGHT = 50.0     # lissage bayésien des taux entre styles (pseudo-matchs)


def load_k(default: int = 4) -> int:
    if cfg.CLUSTER_PARAMS_FILE.exists():
        return int(json.loads(cfg.CLUSTER_PARAMS_FILE.read_text())["k"])
    return default


def team_season_profiles(feats: pd.DataFrame) -> pd.DataFrame:
    """Un profil par (équipe, saison) : le profil d'avant-match du dernier match de la saison."""
    home = feats[["Date", "league", "season", "HomeTeam"] + H_PROF].copy()
    home.columns = ["Date", "league", "season", "team"] + PROFILE_STATS
    away = feats[["Date", "league", "season", "AwayTeam"] + A_PROF].copy()
    away.columns = home.columns
    long = pd.concat([home, away]).dropna(subset=PROFILE_STATS).sort_values("Date", kind="mergesort")
    return long.groupby(["league", "team", "season"], sort=True).tail(1).reset_index(drop=True)


class StyleClusterer:
    """StandardScaler + modèle de clustering, avec affectation au centroïde pour les nouvelles données."""

    def __init__(self, k: int, method: str = "kmeans", seed: int = cfg.SEED):
        self.k, self.method, self.seed = k, method, seed

    def fit(self, profiles: np.ndarray):
        self.scaler_ = StandardScaler().fit(profiles)
        z = self.scaler_.transform(profiles)
        if self.method == "kmeans":
            self.model_ = KMeans(self.k, n_init=10, random_state=self.seed).fit(z)
            labels = self.model_.labels_
        elif self.method == "gmm":
            self.model_ = GaussianMixture(self.k, covariance_type="full", n_init=3, random_state=self.seed).fit(z)
            labels = self.model_.predict(z)
        elif self.method == "ward":
            # Le clustering hiérarchique n'a pas de `predict` : on affecte au centroïde le plus proche
            labels = AgglomerativeClustering(self.k, linkage="ward").fit_predict(z)
            self.centroids_ = np.vstack([z[labels == j].mean(axis=0) for j in range(self.k)])
        else:
            raise ValueError(self.method)
        # Renumérotation stable : cluster 0 = taux de victoires le plus élevé
        win_idx = PROFILE_STATS.index("win")
        win_rate = np.array([profiles[labels == j, win_idx].mean() if (labels == j).any() else -1 for j in range(self.k)])
        order = np.argsort(-win_rate)
        self.relabel_ = np.empty(self.k, dtype=int)
        self.relabel_[order] = np.arange(self.k)
        self.train_labels_ = self.relabel_[labels]
        return self

    def predict(self, profiles: np.ndarray) -> np.ndarray:
        """Cluster de chaque profil ; -1 si le profil est incomplet (trop peu de matchs passés)."""
        out = np.full(len(profiles), -1)
        ok = ~np.isnan(profiles).any(axis=1)
        if ok.any():
            z = self.scaler_.transform(profiles[ok])
            if self.method == "ward":
                raw = ((z[:, None, :] - self.centroids_[None]) ** 2).sum(axis=2).argmin(axis=1)
            else:
                raw = self.model_.predict(z)
            out[ok] = self.relabel_[raw]
        return out


def add_cluster_features(feats: pd.DataFrame, train_mask: np.ndarray, k: int | None = None,
                         method: str = "kmeans", clusterer: StyleClusterer | None = None):
    """Ajoute les variables de clustering (colonnes cl_*) à la table de variables.

    - le clusterer est ajusté sur les profils (équipe, saison) des lignes d'entraînement seulement
      (ou fourni déjà ajusté, pour la prédiction) ;
    - chaque équipe reçoit le cluster de son profil d'avant-match ;
    - cl_rate_* : taux historiques (domicile / nul / extérieur) du duel de styles, cumulés sur
      les matchs strictement antérieurs à la date du match, lissés vers la fréquence globale.
    """
    feats = feats.reset_index(drop=True)
    if clusterer is None:
        prof = team_season_profiles(feats[train_mask])
        clusterer = StyleClusterer(k or load_k(), method).fit(prof[PROFILE_STATS].to_numpy())
    k = clusterer.k
    ch = clusterer.predict(feats[H_PROF].to_numpy(dtype=float))
    ca = clusterer.predict(feats[A_PROF].to_numpy(dtype=float))

    new = {}
    for j in range(k):
        new[f"cl_h_{j}"] = (ch == j).astype(float)
        new[f"cl_a_{j}"] = (ca == j).astype(float)

    # Taux de résultats entre styles : cumul chronologique, mise à jour après chaque journée
    y = feats["y"].to_numpy(dtype=float)
    dates = feats["Date"].to_numpy()
    cell = np.zeros((k, k, 3))
    total = np.array([0.45, 0.26, 0.29]) * 10.0        # a priori faible sur les fréquences globales
    rates = np.full((len(feats), 3), np.nan)
    start, n = 0, len(feats)
    while start < n:
        end = start
        while end < n and dates[end] == dates[start]:
            end += 1
        prior = total / total.sum()
        for i in range(start, end):
            if ch[i] >= 0 and ca[i] >= 0:
                c = cell[ch[i], ca[i]]
                rates[i] = (c + RATE_PRIOR_WEIGHT * prior) / (c.sum() + RATE_PRIOR_WEIGHT)
        for i in range(start, end):
            if not np.isnan(y[i]):
                total[int(y[i])] += 1
                if ch[i] >= 0 and ca[i] >= 0:
                    cell[ch[i], ca[i], int(y[i])] += 1
        start = end
    new["cl_rate_h"], new["cl_rate_d"], new["cl_rate_a"] = rates.T
    out = pd.concat([feats, pd.DataFrame(new, index=feats.index)], axis=1)
    return out, clusterer
