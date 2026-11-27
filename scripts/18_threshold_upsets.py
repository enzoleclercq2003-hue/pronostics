"""Seuil de pronostic et modèle de contre-performance du favori.

1) Seuil : le modèle ne « statue » sur un match que si l'issue la plus probable atteint un seuil de
   probabilité. Le seuil est choisi sur les saisons de décision (walk-forward 2020-2024, prédictions
   calibrées comme en production), puis vérifié sur le test final 2025/26, jamais utilisé pour le choix.
   Règle de choix : le plus petit seuil (pas de 5 points) pour lequel les pronostics émis sont justes
   au moins deux fois sur trois en walk-forward — un pronostic doit être nettement plus souvent juste
   que faux, pas seulement « un peu plus probable que les autres issues ».

2) Contre-performance : parmi les matchs où le modèle a un favori, peut-on prévoir mieux que
   « 1 − probabilité du favori » ceux où il ne gagne pas (nul ou défaite) ? Variables orientées
   favori/outsider : écart avec le marché, fatigue (repos, matchs rapprochés), enjeu (titre, Europe,
   maintien, fin de saison sans enjeu), forme récente contre niveau de la saison, match attendu fermé,
   promu, confrontations directes. Fenêtre glissante : chaque saison 2020-2025 est prédite par un
   modèle appris sur les saisons antérieures seulement (à partir de 2017). Référence emboîtée : la même
   régression avec la seule probabilité du favori (donc recalibrée exactement de la même façon).

    python scripts/18_threshold_upsets.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg
from src.evaluation import oos_for_season
from src.odds import implied_shin

RES = cfg.RESULTS_DIR
WF, FT = list(range(2020, 2025)), [2025]
THS = [0, .40, .45, .50, .55, .60, .65, .70]
rng = np.random.default_rng(0)


def threshold_table(P, Bk, Y, S):
    rows = []
    for lab, sel in (("validation 2017-2019 (non calibré)", [2017, 2018, 2019]), ("walk-forward 2020-2024", WF), ("test final 2025/26", FT)):
        m = np.isin(S, sel)
        for t in THS:
            for who, Q in (("modèle", P), ("bookmakers", Bk)):
                ok = m & ~np.isnan(Q).any(1)
                f, hit = Q.max(1), Q.argmax(1) == Y
                pick = ok & (f >= t)
                rows.append(dict(jeu=lab, seuil=t, source=who, matchs=int(ok.sum()), pronostics=int(pick.sum()),
                                 part=pick.sum() / ok.sum(), precision=hit[pick].mean(),
                                 nuls_pronostiques=int((pick & (Q.argmax(1) == 1)).sum())))
    return pd.DataFrame(rows)


def main():
    # --- prédictions hors échantillon : brutes 2017-2019, calibrées (température) 2020-2025 comme en production
    raw = pd.concat([oos_for_season(s, verbose=False) for s in range(2017, 2026)], ignore_index=True)
    cal = pd.concat([pd.read_pickle(cfg.PROCESSED_DIR / f) for f in ("walkforward_predictions.pkl", "final_test_predictions.pkl")])
    kr = [f"no_odds|logreg|raw|{s}" for s in "HDA"]
    kt = [f"no_odds|logreg|temp|{s}" for s in "HDA"]
    df = raw[["match_id", "season", "y", "odds_h", "odds_d", "odds_a"] + kr].merge(cal[["match_id"] + kt], on="match_id", how="left")
    P = df[kr].to_numpy().copy()
    has_t = df[kt[0]].notna().to_numpy()
    P[has_t] = df.loc[has_t, kt].to_numpy()
    odds = df[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    Bk = np.full_like(P, np.nan)
    okb = ~np.isnan(odds).any(1)
    Bk[okb] = implied_shin(odds[okb])
    Y, S = df.y.to_numpy(int), df.season.to_numpy()

    tab = threshold_table(P, Bk, Y, S)
    tab.to_csv(RES / "threshold_analysis.csv", index=False)
    wf = tab[(tab.jeu == "walk-forward 2020-2024") & (tab.source == "modèle")]
    chosen = float(wf[(wf.seuil > 0) & (wf.precision >= 2 / 3)].seuil.min())
    print(f"Seuil retenu (règle : plus petit seuil où les pronostics sont justes ≥ 2 fois sur 3 en walk-forward) : {chosen:.0%}\n")
    piv = tab.pivot_table(index=["jeu", "seuil"], columns="source", values=["part", "precision"], sort=False)
    print((piv * 100).round(1).to_string(), "\n")
    print("Nuls pronostiqués au seuil retenu :", int(tab[(tab.seuil == chosen) & (tab.source == "modèle")].nuls_pronostiques.sum()), "\n")

    # --- modèle de contre-performance
    feat = pd.read_pickle(cfg.PROCESSED_DIR / "features.pkl")
    stakes = pd.read_pickle(cfg.PROCESSED_DIR / "stakes.pkl")
    d = df[["match_id", "season", "y"]].copy()
    d[["pH", "pD", "pA"]] = P
    d[["bH", "bD", "bA"]] = Bk
    d = d.merge(feat, on=["match_id"], how="left", suffixes=("", "_f")).merge(stakes, on="match_id", how="left")
    home_fav = (d.pH >= d.pA).to_numpy()
    d["p_fav"] = np.where(home_fav, d.pH, d.pA)
    d["fav_home"] = home_fav.astype(float)
    d["upset"] = (d.y.to_numpy() != np.where(home_fav, 0, 2)).astype(int)
    d["b_fav"] = np.where(home_fav, d.bH, d.bA)

    def side(col):            # valeur côté favori, côté outsider
        h, a = d[f"h_{col}"].to_numpy(float), d[f"a_{col}"].to_numpy(float)
        return np.where(home_fav, h, a), np.where(home_fav, a, h)

    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    X = pd.DataFrame({"logit_fav": lg(d.p_fav), "fav_home": d.fav_home})
    for col in ("rest_days", "matches_14d", "roll3_pts", "ewm_pts", "season_ppg", "promoted", "st_title", "st_europe", "st_releg", "st_any", "st_dead"):
        X[f"fav_{col}"], X[f"dog_{col}"] = side(col)
    X["fav_form_dip"] = X.fav_roll3_pts - X.fav_season_ppg          # favori en méforme par rapport à sa saison
    X["dog_form_up"] = X.dog_roll3_pts - X.dog_season_ppg            # outsider en regain de forme
    X["fav_rest_days"], X["dog_rest_days"] = X.fav_rest_days.clip(upper=14), X.dog_rest_days.clip(upper=14)
    X["goals_expected"] = d.dc_lambda + d.dc_mu                      # match attendu fermé : plus de nuls
    X["h2h_fav"] = np.where(home_fav, d.h2h_pts_h, 3 - d.h2h_pts_h) * (d.h2h_n > 0)
    X["season_progress"] = d.season_progress
    X["fav_away_tired"] = (1 - d.fav_home) * (X.fav_matches_14d >= 4)
    for c in ("league_E0", "league_SP1", "league_I1", "league_D1", "league_F1"):
        X[c] = d[c]
    market = pd.DataFrame({"logit_book": lg(d.b_fav), "gap_market": d.p_fav - d.b_fav})
    X = X.fillna(X.median(numeric_only=True))
    up, seas, pf = d.upset.to_numpy(), d.season.to_numpy(), d.p_fav.to_numpy()
    has_book = ~np.isnan(d.b_fav.to_numpy())

    sets = {"référence : probabilité du favori seule": ["logit_fav"],
            "+ contexte (fatigue, enjeu, forme, style de match)": list(X.columns),
            "+ contexte + écart avec le marché": list(X.columns) + list(market.columns),
            "favori + marché seulement (retenu)": ["logit_fav", "logit_book", "gap_market", "fav_home"],
            "marché seul (bookmakers)": ["logit_book"]}
    XA = pd.concat([X, market], axis=1)
    preds = {k: np.full(len(d), np.nan) for k in sets}
    for s in WF + FT:
        tr = (seas < s) & has_book
        te = seas == s
        for k, cols in sets.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
            m.fit(XA.loc[tr, cols].fillna(0), up[tr])
            preds[k][te] = m.predict_proba(XA.loc[te, cols].fillna(0))[:, 1]

    rows = []
    for lab, sel in (("walk-forward 2020-2024", WF), ("test final 2025/26", FT)):
        for pop, extra in (("tous les matchs", np.ones(len(d), bool)), (f"pronostics émis (favori ≥ {chosen:.0%})", pf >= chosen)):
            m = np.isin(seas, sel) & has_book & extra
            ref = preds["référence : probabilité du favori seule"][m]
            for k in sets:
                p = preds[k][m]
                # écart de log loss avec la référence, intervalle par rééchantillonnage apparié
                l_ref = -(up[m] * np.log(ref) + (1 - up[m]) * np.log(1 - ref))
                l_k = -(up[m] * np.log(p) + (1 - up[m]) * np.log(1 - p))
                diff = l_k - l_ref
                boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
                top = p >= np.quantile(p, 0.8)                         # les 20 % de pronostics jugés les plus fragiles
                rows.append(dict(jeu=lab, population=pop, modele=k, matchs=int(m.sum()), taux_contre_perf=up[m].mean(),
                                 log_loss=log_loss(up[m], p), delta_vs_ref=diff.mean(),
                                 ic95_bas=np.quantile(boots, .025), ic95_haut=np.quantile(boots, .975),
                                 auc=roc_auc_score(up[m], p), contre_perf_top20=up[m][top].mean()))
    res = pd.DataFrame(rows)
    res.to_csv(RES / "upset_experiment.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))

    # variables les plus influentes (modèle complet, toutes saisons jusqu'à 2024)
    tr = (seas < 2025) & has_book
    cols = sets["favori + marché seulement (retenu)"]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(XA.loc[tr, cols].fillna(0), up[tr])
    coef = pd.Series(m[-1].coef_[0], index=cols).sort_values(key=abs, ascending=False)
    print("\nCoefficients standardisés (positif = plus de contre-performances) :\n", coef.head(12).round(3).to_string())

    # Modèle de production : appris sur toutes les saisons disponibles. Alerte « risque de contre-performance »
    # pour les 20 % de pronostics émis jugés les plus fragiles sur ces saisons.
    import joblib
    prod = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(XA.loc[has_book, cols], up[has_book])
    picks = has_book & (pf >= chosen)
    risk = prod.predict_proba(XA.loc[picks, cols])[:, 1]
    cut = float(np.quantile(risk, 0.8))
    joblib.dump({"model": prod, "features": cols, "alert": cut, "threshold": chosen, "mean": float(up[picks].mean()),
                 "alert_rate": float(up[picks][risk >= cut].mean())}, cfg.MODELS_DIR / "upset_model.joblib")
    print(f"\nModèle de contre-performance enregistré : alerte si risque ≥ {cut:.0%} ; contre-performances parmi les "
          f"alertes {up[picks][risk >= cut].mean():.0%}, parmi les autres pronostics {up[picks][risk < cut].mean():.0%}")


if __name__ == "__main__":
    main()
