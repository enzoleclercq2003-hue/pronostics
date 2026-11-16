"""Étape 8 : graphiques (comparaison, calibration, confusion) et importance des variables (SHAP, permutation)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config as cfg
from src.evaluation import get
from src.features import select_features
from src.folds import load_fold, split
from src.metrics import confusion, per_match_losses
from src.models import Model, load_params
from src.plotting import AXIS, INK, INK2, MUTED, SEQ_BLUES, SERIES, save, set_style

set_style()
R = cfg.RESULTS_DIR
selected = json.loads((R / "selected_model.json").read_text(encoding="utf-8"))
wf = pd.read_pickle(cfg.PROCESSED_DIR / "walkforward_predictions.pkl")
final = pd.read_pickle(cfg.PROCESSED_DIR / "final_test_predictions.pkl")
best_key, best_label = selected["no_odds"]["key"], selected["no_odds"]["label"]
CLASS_LABELS = ["Victoire domicile", "Nul", "Victoire extérieur"]

# --- 1. Comparaison des modèles (walk-forward) : log loss et IC bootstrap -------------------
table = pd.read_csv(R / "model_comparison_walkforward.csv")
home_ll = table.loc[table["clé"] == "base|home", "logloss"].iloc[0]
t = table[table["clé"] != "base|home"].sort_values("logloss", ascending=False).reset_index(drop=True)
fam_color = {"référence": MUTED, "bookmakers": SERIES[1], "sans cotes": SERIES[0], "avec cotes": SERIES[2]}
fig, ax = plt.subplots(figsize=(8.5, 7.5))
for fam, col in fam_color.items():
    m = t.famille == fam
    ax.errorbar(t.logloss[m], t.index[m], xerr=[t.logloss[m] - t.logloss_ci_lo[m], t.logloss_ci_hi[m] - t.logloss[m]],
                fmt="o", color=col, ecolor=col, elinewidth=1.2, capsize=0, markersize=6, label=fam.capitalize())
ax.set_yticks(t.index, [f"{r['modèle']}" + (" [cotes]" if r.famille == "avec cotes" else "") for _, r in t.iterrows()])
ax.set(xlabel="Log loss (plus bas = meilleur), IC bootstrap à 95 %",
       title="Walk-forward 2020/21 à 2024/25 : log loss par modèle")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper right")
fig.text(0.01, -0.01, f"Hors graphique : « toujours victoire domicile » (fréquences) = {home_ll:.3f}", color=INK2, fontsize=8)
save(fig, R / "model_comparison.png")


# --- 2. Courbes de fiabilité (calibration) --------------------------------------------------
def reliability(ax, y, p, cls, color, label, n_bins=10):
    edges = np.quantile(p[:, cls], np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p[:, cls], side="right") - 1, 0, n_bins - 1)
    xs = [p[idx == b, cls].mean() for b in range(n_bins) if (idx == b).any()]
    ys = [(y[idx == b] == cls).mean() for b in range(n_bins) if (idx == b).any()]
    ax.plot(xs, ys, marker="o", color=color, label=label, markeredgecolor="white", markeredgewidth=0.8)


for data, tag, title in ((wf, "walkforward", "walk-forward 2020/21 à 2024/25"), (final, "final_test", f"test final {cfg.season_label(cfg.FINAL_TEST_SEASON)}")):
    y = data["y"].to_numpy(int)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    for cls, ax in enumerate(axes):
        lim = (0, 0.9) if cls != 1 else (0.1, 0.42)
        ax.plot(lim, lim, color=AXIS, lw=1, ls="--", label="Calibration parfaite")
        reliability(ax, y, get(data, best_key), cls, SERIES[0], f"{best_label} (sans cotes)")
        reliability(ax, y, get(data, "book|shin"), cls, SERIES[1], "Bookmakers (Shin)")
        ax.set(title=CLASS_LABELS[cls], xlabel="Probabilité prédite (moyenne par décile)", xlim=lim, ylim=lim)
    axes[0].set_ylabel("Fréquence observée")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Courbes de fiabilité, {title} ({len(y)} matchs)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, R / f"calibration_{tag}.png")

# --- 3. Matrices de confusion (test final) --------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blues", ["#fcfcfb"] + SEQ_BLUES)
y = final["y"].to_numpy(int)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, (key, name) in zip(axes, ((best_key, f"{best_label} (sans cotes)"), ("book|shin", "Bookmakers (Shin)"))):
    cm = confusion(y, get(final, key))
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), ["Domicile", "Nul", "Extérieur"]); ax.set_yticks(range(3), ["Domicile", "Nul", "Extérieur"])
    ax.set(xlabel="Prédit", ylabel="Réel", title=f"{name}\nprécision {np.trace(cm) / cm.sum():.1%}")
    ax.grid(False)
fig.suptitle(f"Matrices de confusion, test final {cfg.season_label(cfg.FINAL_TEST_SEASON)} (effectif et part de la ligne)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, R / "confusion_matrix.png")

# --- 4. Importance des variables : LightGBM sans cotes, entraîné < 2024/25, évalué sur 2024/25 ---
season = cfg.WALK_FORWARD_SEASONS[-1]
fold = load_fold(season)
train, test = split(fold, season)
feats = select_features(fold.columns)
model = Model("lgbm", load_params("lgbm"), feats).fit(train)
X_test = pd.DataFrame(model.imputer_.transform(test[feats]), columns=feats)
y_test = test["y"].to_numpy(int)

# 4a. SHAP (TreeExplainer) : moyenne des |valeurs SHAP| par classe
sv = shap.TreeExplainer(model.est_).shap_values(X_test)
sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)       # (matchs, variables, classes)
mean_abs = np.abs(sv).mean(axis=0)
shap_df = pd.DataFrame(mean_abs, index=feats, columns=cfg.CLASSES)
shap_df["total"] = shap_df.sum(axis=1)
shap_df = shap_df.sort_values("total", ascending=False)
shap_df.to_csv(R / "feature_importance_shap.csv")
top = shap_df.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6.5))
left = np.zeros(len(top))
for j, c in enumerate(cfg.CLASSES):
    ax.barh(top.index, top[c], left=left, color=SERIES[j], edgecolor="#fcfcfb", linewidth=1.5, height=0.7, label=CLASS_LABELS[j])
    left += top[c].to_numpy()
ax.set(xlabel="Moyenne des |valeurs SHAP| (log-odds), empilée par classe",
       title=f"SHAP : 20 variables les plus influentes (LightGBM sans cotes, saison {cfg.season_label(season)})")
ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
save(fig, R / "feature_importance_shap.png")
print("SHAP - 12 premières variables :")
print(shap_df.head(12).to_string(float_format="%.4f"))


# 4b. Importance par permutation (hausse de log loss), par variable puis par groupe
def logloss_of(df):
    return per_match_losses(y_test, model.predict_proba(df))["logloss"].mean()


base = logloss_of(test)
rng = np.random.default_rng(cfg.SEED)


def permute(cols, n_repeats=10):
    deltas = []
    for _ in range(n_repeats):
        shuffled = test.copy()
        shuffled[cols] = test[cols].to_numpy()[rng.permutation(len(test))]   # même permutation pour tout le groupe
        deltas.append(logloss_of(shuffled) - base)
    return float(np.mean(deltas)), float(np.std(deltas))


perm = pd.DataFrame([(f, *permute([f], 5)) for f in feats], columns=["variable", "delta_logloss", "std"])
perm = perm.sort_values("delta_logloss", ascending=False)
perm.to_csv(R / "feature_importance_permutation.csv", index=False)

groups = {
    "Elo": [f for f in feats if f.startswith("elo_")],
    "Dixon-Coles (forces, probabilités)": [f for f in feats if f.startswith("dc_")],
    "Forme pondérée (EWM)": [f for f in feats if "_ewm_" in f and "venue" not in f],
    "Forme, fenêtres 3/5/10": [f for f in feats if "_roll" in f and "venue" not in f],
    "Forme domicile / extérieur": [f for f in feats if "venue" in f],
    "Saison : points, classement, promu": [f for f in feats if ("season_" in f and f != "season_num") or f.startswith("rank_") or "promoted" in f],
    "Repos et calendrier": [f for f in feats if "rest_days" in f or "matches_14d" in f or "matchday" in f],
    "Confrontations directes": [f for f in feats if f.startswith("h2h_")],
    "Clustering (styles)": [f for f in feats if f.startswith("cl_")],
    "Championnat et saison": [f for f in feats if f.startswith("league_") or f == "season_num"],
}
assert sum(len(v) for v in groups.values()) == len(feats), "groupes de variables incomplets"
gperm = pd.DataFrame([(g, len(c), *permute(c)) for g, c in groups.items()], columns=["groupe", "n_variables", "delta_logloss", "std"])
gperm = gperm.sort_values("delta_logloss", ascending=False)
gperm.to_csv(R / "feature_importance_permutation_groups.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
tp = perm.head(15).iloc[::-1]
axes[0].barh(tp.variable, tp.delta_logloss, xerr=tp["std"], color=SERIES[0], ecolor=INK2, height=0.65)
axes[0].set(title="Par variable (15 premières)", xlabel="Hausse de log loss après permutation")
gp = gperm.iloc[::-1]
axes[1].barh(gp.groupe, gp.delta_logloss, xerr=gp["std"], color=SERIES[0], ecolor=INK2, height=0.65)
axes[1].set(title="Par groupe de variables (permutation conjointe)", xlabel="Hausse de log loss après permutation")
for ax in axes:
    ax.grid(axis="y", visible=False); ax.axvline(0, color=AXIS, lw=0.8)
fig.suptitle(f"Importance par permutation (LightGBM sans cotes, saison {cfg.season_label(season)}, log loss de base {base:.4f})",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, R / "feature_importance_permutation.png")
print("\nPermutation par groupe :")
print(gperm.to_string(index=False, float_format="%.5f"))
print("\nPermutation - 10 premières variables :")
print(perm.head(10).to_string(index=False, float_format="%.5f"))
