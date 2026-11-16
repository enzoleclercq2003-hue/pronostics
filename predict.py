"""Prédit les probabilités (domicile / nul / extérieur) pour une liste de matchs à venir.

Usage :
    python predict.py data/fixtures_example.csv [--out predictions.csv] [--with-odds] [--refresh]

Le CSV d'entrée contient les colonnes : date (AAAA-MM-JJ), home_team, away_team
(et odds_h, odds_d, odds_a si l'on utilise le modèle "avec cotes").
Les variables sont reconstruites à partir de l'historique (data/processed/matches.pkl).
`--refresh` retélécharge la saison en cours avant de prédire.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

import config as cfg
from src.data import load_matches, load_processed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixtures", help="CSV : date, home_team, away_team")
    parser.add_argument("--out", default=None, help="fichier CSV de sortie (facultatif)")
    parser.add_argument("--with-odds", action="store_true", help="utiliser le modèle 'avec cotes'")
    parser.add_argument("--refresh", action="store_true", help="mettre à jour l'historique de la saison en cours")
    args = parser.parse_args()

    if args.refresh:
        history = load_matches(refresh_current=True, verbose=False)
        history.to_pickle(cfg.PROCESSED_DIR / "matches.pkl")
    else:
        history = load_processed()
    name = "best_pipeline_with_odds.joblib" if args.with_odds else "best_pipeline.joblib"
    predictor = joblib.load(cfg.MODELS_DIR / name)
    fixtures = pd.read_csv(args.fixtures)
    preds = predictor.predict(fixtures, history)

    print(f"Modèle : {predictor.key} | historique jusqu'au {history.Date.max().date()} ({len(history)} matchs)")
    print(preds.to_string(index=False, float_format="%.3f"))
    if args.out:
        preds.to_csv(args.out, index=False)
        print(f"\nPrédictions enregistrées dans {args.out}")


if __name__ == "__main__":
    main()
