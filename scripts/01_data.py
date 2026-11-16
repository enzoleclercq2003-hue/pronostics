"""Étape 1 : téléchargement, nettoyage et contrôle des données."""
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config as cfg
from src.data import ODDS_COLS, STAT_COLS, load_matches

pd.set_option("display.width", 200, "display.max_columns", 30)

df = load_matches(refresh_current="--refresh" in sys.argv)
df.to_pickle(cfg.PROCESSED_DIR / "matches.pkl")

rep = pd.read_csv(cfg.RESULTS_DIR / "data_report.csv")
print("\nSources utilisées :")
print(rep["source"].str.replace(r"cache \((.*)\)", r"\1", regex=True).value_counts().to_string())
print("\nFichiers absents ou colonnes manquantes :")
print(rep[(rep.n_matches == 0) | rep.missing_cols.notna()].to_string(index=False) or "aucun")

print(f"\nDimensions : {df.shape}")
print(f"Plage de dates : {df.Date.min().date()} -> {df.Date.max().date()}")
print("\nMatchs par championnat et saison :")
print(df.pivot_table(index="season", columns="league", values="match_id", aggfunc="count").to_string())
print("\nTaux de valeurs manquantes (stats) par championnat :")
print(df.groupby("league")[STAT_COLS].apply(lambda g: g.isna().mean()).round(3).to_string())
print("\nDisponibilité des cotes par saison (part des matchs) :")
print(df.assign(pre=df.odds_h.notna(), close=df.close_h.notna()).groupby("season")[["pre", "close"]].mean().round(3).T.to_string())
print("\nOrigine des cotes d'avant-match :", df.odds_src.value_counts().to_dict())
print("Origine des cotes de clôture :", df.close_src.value_counts().to_dict())
print("\nRépartition des résultats :", df.FTR.value_counts(normalize=True).round(4).to_dict())

# Contrôle des noms d'équipes : paires de noms très proches dans un même championnat
print("\nNoms d'équipes proches (à vérifier) :")
for lg in cfg.LEAGUES:
    teams = sorted(set(df.loc[df.league == lg, "HomeTeam"]))
    print(f"  {lg} : {len(teams)} équipes")
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            if SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.8:
                print(f"    ? {a!r} ~ {b!r}")

print("\n5 premières lignes :")
print(df[["Date", "league", "season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "AS", "odds_h", "odds_d", "odds_a"]].head().to_string())
