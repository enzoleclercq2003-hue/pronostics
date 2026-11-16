"""Étape 3 : choix de k (silhouette, coude, stabilité), comparaison K-means / GMM / hiérarchique, ACP.

Le choix de k n'utilise que les saisons antérieures à la première saison de test walk-forward.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config as cfg
from src.clustering import StyleClusterer, add_cluster_features, team_season_profiles
from src.features import PROFILE_STATS
from src.plotting import INK2, SERIES, save, set_style

set_style()
pd.set_option("display.width", 200)
feats = pd.read_pickle(cfg.PROCESSED_DIR / "features.pkl")
train = feats[feats.season < cfg.WALK_FORWARD_SEASONS[0]]
prof = team_season_profiles(train)
X = prof[PROFILE_STATS].to_numpy()
Z = StandardScaler().fit_transform(X)
print(f"Profils (équipe, saison) : {prof.shape}  |  saisons {prof.season.min()}..{prof.season.max()}")
print(prof.head().round(2).to_string())

# --- Choix de k : silhouette, inertie (coude), stabilité par sous-échantillonnage ---------
rng = np.random.default_rng(cfg.SEED)
rows = []
for k in cfg.CLUSTER_K_RANGE:
    km = KMeans(k, n_init=10, random_state=cfg.SEED).fit(Z)
    gmm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=cfg.SEED).fit(Z)
    ward = AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
    # Stabilité : K-means réajusté sur 20 sous-échantillons de 80 %, ARI avec la partition de référence
    aris = []
    for b in range(20):
        sub = rng.choice(len(Z), int(0.8 * len(Z)), replace=False)
        km_b = KMeans(k, n_init=5, random_state=cfg.SEED + b).fit(Z[sub])
        aris.append(adjusted_rand_score(km.labels_, km_b.predict(Z)))
    rows.append({
        "k": k, "inertia": km.inertia_, "sil_kmeans": silhouette_score(Z, km.labels_),
        "stability_ari": np.mean(aris), "sil_gmm": silhouette_score(Z, gmm.predict(Z)),
        "bic_gmm": gmm.bic(Z), "sil_ward": silhouette_score(Z, ward),
        "ari_kmeans_gmm": adjusted_rand_score(km.labels_, gmm.predict(Z)),
        "ari_kmeans_ward": adjusted_rand_score(km.labels_, ward),
        "min_cluster_size": np.bincount(km.labels_).min(),
    })
sel = pd.DataFrame(rows)
sel.to_csv(cfg.RESULTS_DIR / "clustering_k_selection.csv", index=False)
print("\nSélection de k :")
print(sel.round(3).to_string(index=False))

# Règle de décision : meilleure silhouette parmi les k dont la stabilité (ARI) est >= 0.8
stable = sel[sel.stability_ari >= 0.8]
best_k = int((stable if len(stable) else sel).sort_values("sil_kmeans", ascending=False).iloc[0].k)
cfg.CLUSTER_PARAMS_FILE.write_text(json.dumps({"k": best_k}))
print(f"\nk retenu : {best_k} (silhouette maximale parmi les partitions stables)")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
axes[0].plot(sel.k, sel.sil_kmeans, marker="o", label="K-means")
axes[0].plot(sel.k, sel.sil_gmm, marker="o", label="GMM")
axes[0].plot(sel.k, sel.sil_ward, marker="o", label="Hiérarchique (Ward)")
axes[0].set(title="Score de silhouette", xlabel="k"); axes[0].legend()
axes[1].plot(sel.k, sel.inertia, marker="o"); axes[1].set(title="Méthode du coude (inertie K-means)", xlabel="k")
axes[2].plot(sel.k, sel.stability_ari, marker="o"); axes[2].set(title="Stabilité K-means (ARI, sous-échantillons)", xlabel="k", ylim=(0, 1.02))
for ax in axes:
    ax.axvline(best_k, color=INK2, lw=0.8, ls=":")
save(fig, cfg.RESULTS_DIR / "clustering_silhouette_elbow.png")

# --- Profil des clusters et ACP 2D --------------------------------------------------------
cl = StyleClusterer(best_k, "kmeans").fit(X)
prof["cluster"] = cl.train_labels_
centers = prof.groupby("cluster")[PROFILE_STATS].mean()
centers["n"] = prof.cluster.value_counts().sort_index()
centers.to_csv(cfg.RESULTS_DIR / "clustering_centroids.csv")
print("\nProfil moyen par cluster (cluster 0 = taux de victoires le plus élevé) :")
print(centers.round(2).to_string())
print("\nExemples d'équipes par cluster (saison 2018/19) :")
for j in range(best_k):
    ex = prof[(prof.cluster == j) & (prof.season == 2018)].team.tolist()[:10]
    print(f"  cluster {j} : {', '.join(ex)}")

pca = PCA(2, random_state=cfg.SEED).fit(Z)
P = pca.transform(Z)
fig, ax = plt.subplots(figsize=(6.5, 5))
for j in range(best_k):
    m = prof.cluster == j
    ax.scatter(P[m, 0], P[m, 1], s=22, color=SERIES[j], edgecolor="white", linewidth=0.5,
               label=f"Cluster {j} (n={m.sum()}, {centers.loc[j, 'win']:.0%} de victoires)")
ax.set(title=f"Styles d'équipes : K-means (k={best_k}) projeté par ACP",
       xlabel=f"Composante 1 ({pca.explained_variance_ratio_[0]:.0%} de la variance)",
       ylabel=f"Composante 2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.legend(loc="best")
save(fig, cfg.RESULTS_DIR / "clustering_pca.png")
load = pd.DataFrame(pca.components_.T, index=PROFILE_STATS, columns=["PC1", "PC2"])
print("\nContributions des variables à l'ACP :")
print(load.round(2).to_string())

# --- Aperçu des variables de clustering sur le premier pli --------------------------------
out, _ = add_cluster_features(feats, (feats.season < cfg.WALK_FORWARD_SEASONS[0]).to_numpy(), k=best_k)
cl_cols = [c for c in out.columns if c.startswith("cl_")]
print(f"\nVariables de clustering : {cl_cols}")
print(out.loc[out.season == cfg.WALK_FORWARD_SEASONS[0], ["Date", "HomeTeam", "AwayTeam", "cl_rate_h", "cl_rate_d", "cl_rate_a"]].head().to_string(float_format="%.3f"))
