# Prédiction de l'issue des matchs de football (1 / N / 2)

Premier League, La Liga, Serie A, Bundesliga et Ligue 1 — probabilités calibrées, validation
chronologique stricte, comparaison honnête avec les bookmakers.

Le rapport complet (résultats, conclusions, limites) est dans [results/rapport.md](results/rapport.md).

## Installation

```bash
uv venv .venv --python 3.14
```

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

## Reproduire le projet (dans l'ordre)

| Étape | Commande | Rôle |
|---|---|---|
| 1 | `python scripts/01_data.py` | Téléchargement (cache dans `data/raw`), nettoyage |
| 2 | `python scripts/02_features.py` | Optimisation d'Elo par validation, construction des variables |
| – | `python -m pytest tests -q` | Tests anti-fuite |
| 3 | `python scripts/03_clustering.py` | Choix de k, K-means / GMM / hiérarchique, ACP |
| 4 | `python scripts/04_tune.py` | Optuna, 100 essais par modèle (puis `--variant odds logreg lgbm`) |
| 5 | `python scripts/05_walkforward.py` | Walk-forward 2020/21 → 2024/25, calibration, ensembles, sélection |
| 6 | `python scripts/06_ablations.py` | Ablations : clustering, Elo, cotes |
| 7 | `python scripts/07_final_test.py` | Test final 2025/26 (une seule fois) + sauvegarde du pipeline |
| 8 | `python scripts/08_figures_importance.py` | Graphiques, SHAP, importance par permutation |
| 9 | `python scripts/09_value_backtest.py` | Optionnel : backtest « value » (évaluation uniquement) |
| 10 | `python scripts/10_sota_extension.py` | Expérience : pi-ratings et forces sur tirs cadrés (état de l'art) |

| 11 | `python scripts/11_player_features.py` | Expérience : variables au niveau joueur (données dans `data/external/transfermarkt`) |

| 12 | `python scripts/12_player_features_live.py` | Expérience : gain avec les seules données maintenables en direct |
| 13 | `python scripts/13_live_check_2026.py` | Contrôle de la chaîne de production sur les matchs joués de 2026/27 |

| 14 | `python scripts/14_player_ext.py` | Expérience : indisponibilités, postes, âge, banc (non retenue) |
| 15 | `python scripts/15_stakes.py` | Expérience : enjeu sportif (titre, Europe, maintien) |
| 16 | `python scripts/16_retrain_with_stakes.py` | Modèle de production avec l'enjeu |

## Prédictions de la semaine (automatique)

Le modèle hebdomadaire ajoute au modèle de base la valeur marchande des onze alignés (compositions ESPN,
`src/espn.py`, `src/live_players.py`, `src/pipeline_players.py`) et la composition officielle quand elle
est publiée : relancer environ une heure avant un match pour l'intégrer.

**Application** : double-cliquez sur le raccourci « Pronostics Foot » du Bureau (ou sur `Pronostics Foot.bat`,
ou `python app.py`). Un petit serveur local (`app.py`, interface dans `app/index.html`) ouvre
http://127.0.0.1:8765 dans le navigateur : onglets Matchs, Top pronostics, Suivi et Aide. Les prédictions
sont recalculées à l'ouverture si elles ont plus de 3 heures, ou sur le bouton « Actualiser » ; le serveur
s'arrête seul dix minutes après la fermeture de l'onglet.

L'interface s'ouvre dans une fenêtre d'application (Edge ou Chrome en mode `--app`), sinon dans le navigateur
par défaut. Elle propose une fiche par match (forme, Elo, scores probables, absents de la dernière feuille de
match), un match à la une, la recherche, des équipes favorites, les derniers résultats, un suivi en graphique
et une version anglaise. Liens directs : `#top`, `#suivi`, `#modele`, `?open=N`, `?lang=en`.

**Actualité** : à chaque calcul, l'application locale lit neuf flux RSS publics (BBC, The Guardian, Sky, L'Équipe, AS, Mundo Deportivo, Gazzetta, Corriere dello Sport, Sportschau), repère les joueurs annoncés absents (`src/news.py`, sans IA payante) et recalcule les matchs concernés avec une composition probable. Désactivable avec `USE_NEWS` dans `config.py`. Les absences retenues sont consignées dans `results/news_absences_log.csv` pour mesurer leur apport après quelques semaines.

**Version hébergée** : `python build_site.py` fabrique un site statique dans `site/` (sans écussons ni
données joueurs), mis à jour automatiquement par `.github/workflows/site.yml`. Voir `DEPLOY.md`.

En ligne de commande, `python semaine.py` produit toujours la page simple `predictions_semaine.html`.
Chaque prédiction émise est aussi consignée dans `results/predictions_log.csv`.

Tests anti-fuite : `python -m pytest tests -q`. Conclusions : `results/rapport.md`.

Toutes les commandes se lancent avec l'interpréteur de l'environnement (`.venv/Scripts/python.exe`).
La graine aléatoire est fixée partout (`config.SEED = 42`).

## Prédire des matchs à venir

```bash
.venv/Scripts/python.exe predict.py data/fixtures_example.csv --out predictions.csv
```

Le CSV contient `date, home_team, away_team` (et `odds_h, odds_d, odds_a` avec `--with-odds`).
`--refresh` met d'abord à jour l'historique de la saison en cours. Les noms d'équipes sont ceux
de football-data.co.uk (`Man City`, `Paris SG`, `Ath Madrid`…) ; les variantes courantes et les
fautes de frappe légères sont reconnues.

## Organisation

```
config.py            paramètres (championnats, saisons, graine, chemins)
src/data.py          téléchargement, nettoyage, harmonisation
src/features.py      variables d'avant-match (forme, repos, classement, confrontations)
src/elo.py           Elo dynamique         src/poisson.py   Dixon-Coles sur fenêtre glissante
src/clustering.py    styles d'équipes      src/odds.py      probabilités implicites (Shin)
src/models.py        modèles, calibration, ensembles
src/evaluation.py    walk-forward, tableaux, tests appariés
src/pipeline.py      pipeline complet sauvegardé (MatchPredictor)
tests/               tests anti-fuite
results/             tableaux, graphiques, rapport.md
models/              pipeline final (joblib)
```
