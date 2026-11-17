# Prédiction de l'issue des matchs de football (H / D / A) — rapport

Championnats : Premier League, La Liga, Serie A, Bundesliga, Ligue 1. Saisons 2005/06 à 2026/27 (en cours).
Graine aléatoire fixée à 42 partout. Tous les chiffres ci-dessous sortent des scripts du dépôt et sont
reproductibles (`README.md`).

## 1. Résumé

| | Précision | Log loss | Brier | RPS |
|---|---|---|---|---|
| **Test final 2025/26 (1 752 matchs, utilisé une seule fois)** | | | | |
| Toujours victoire domicile (fréquences) | 44,0 % | 1,0722 | 0,6486 | 0,2294 |
| Elo seul | 50,9 % | 0,9953 | 0,5934 | 0,2029 |
| **Meilleur modèle sans cotes : régression logistique calibrée** | **52,5 %** [50,1 ; 54,7] | **0,9884** [0,969 ; 1,010] | 0,5890 | 0,2008 |
| Bookmakers (Bet365, méthode de Shin) | 53,3 % [51,0 ; 55,5] | 0,9784 [0,958 ; 1,000] | 0,5824 | 0,1976 |
| Meilleur modèle avec cotes : stacking | 53,1 % [50,9 ; 55,4] | 0,9778 | 0,5819 | 0,1972 |
| **Walk-forward 2020/21 à 2024/25 (8 981 matchs)** | | | | |
| Régression logistique calibrée (sans cotes) | 52,8 % [51,8 ; 53,9] | 0,9823 | 0,5849 | 0,1989 |
| Bookmakers (Bet365, Shin) | 53,8 % [52,7 ; 54,8] | 0,9716 | 0,5774 | 0,1954 |
| Bookmakers, cotes de clôture (Shin) | 54,0 % | 0,9682 | 0,5753 | 0,1945 |
| Stacking (avec cotes) | 53,7 % | 0,9717 | 0,5773 | 0,1954 |

Crochets : intervalles de confiance bootstrap à 95 %.

Conclusions, sans embellissement :

1. **Le modèle sans cotes ne bat pas les bookmakers.** Sa log loss est plus élevée de 0,0100 sur le test
   final (IC 95 % [+0,0041 ; +0,0165], p bootstrap < 0,001, p Diebold-Mariano = 0,002) et de 0,0107 en
   walk-forward (IC [+0,0078 ; +0,0138]). L'écart est petit mais net et stable.
2. **Avec les cotes en entrée, le modèle égale les bookmakers et ne les dépasse pas** : écart de −0,0006
   sur le test final (IC [−0,0044 ; +0,0031], p = 0,70) et de +0,0001 en walk-forward (p = 0,86). Nos
   variables n'ajoutent rien de mesurable à l'information contenue dans les cotes.
3. **La précision (52,5 à 52,8 %) est dans le bas de la fourchette attendue (52-56 %)**, à environ un point
   des bookmakers. Aucun signe de fuite : rien n'approche 60 %.
4. **Le clustering n'apporte pas de gain robuste** (détail en section 6).
5. **Aucun modèle ne prédit de match nul** : F1 de la classe « nul » proche de 0 pour tous, bookmakers
   compris. La probabilité de nul dépasse très rarement celles des deux autres issues.

## 2. Données

- Source : **football-data.co.uk pour 110 fichiers sur 110** (le secours GitHub n'a pas servi). Fichiers
  mis en cache dans `data/raw`.
- 38 523 matchs, du 2005-07-29 au 2026-11-08. Détail par championnat et saison : `data_report.csv`.
- Nettoyage : dates aux deux formats, encodages, lignes vides, doublons, noms d'équipes harmonisés.
- Colonnes absentes à la source : tirs cadrés en Bundesliga 2005/06 ; fautes et corners en Ligue 1 2005/06 ;
  fautes en Ligue 1 2006/07. Valeurs manquantes remplacées par la moyenne du championnat, calculée sur
  l'entraînement de chaque pli.
- Cotes : Bet365 d'avant-match partout ; cotes de clôture depuis 2012/13 (Pinnacle, sinon Bet365).
- La saison 2021/22 compte 1 825 matchs au lieu de 1 826 : une ligne manque à la source ou a été écartée
  au nettoyage (non investigué, sans effet sur les conclusions).

## 3. Absence de fuite

- Toutes les variables d'un match n'utilisent que des matchs strictement antérieurs. Les statistiques du
  match lui-même (buts, tirs, corners, cartons) ne sont jamais en entrée.
- Tests automatiques (`tests/test_no_leakage.py`) : on supprime le futur, puis on efface aussi le résultat
  et les statistiques du jour ; les variables du match doivent rester identiques. Vérifié pour les
  variables principales, le clustering et les variables supplémentaires de la section 8.
- Découpage uniquement chronologique. Hyperparamètres réglés sur 2017/18-2019/20 ; walk-forward sur
  2020/21-2024/25 ; choix du modèle sur le walk-forward ; test final 2025/26 calculé une seule fois ensuite.
- Calibration et ensembles appris, pour chaque saison, sur les prédictions hors échantillon des saisons
  antérieures.

## 4. Variables

123 variables sans cotes (127 avec) : Elo par championnat, forme pondérée exponentiellement et sur
fenêtres 3/5/10 (points, buts, tirs, tirs cadrés, corners, fautes, cartons, séparément domicile/extérieur),
forces Dixon-Coles réestimées chaque semaine, repos et calendrier, promu, confrontations directes,
classement, championnat, saison, clustering.

- Elo retenu par validation : K = 12, avantage domicile = 0, report inter-saisons 0,9, promu à −80
  (`elo_grid.csv`). L'avantage du terrain nul dans l'Elo n'est pas une anomalie : il est capté par
  la constante du modèle en aval.
- Importance par permutation (LightGBM, saison 2024/25) : trois groupes portent presque tout le signal,
  **Dixon-Coles (+0,023 de log loss si permuté), Elo (+0,021), forme pondérée (+0,019)**. Clustering
  +0,001. Repos, confrontations directes, classement et fenêtres 3/5/10 : effet nul.
  Graphiques : `feature_importance_permutation.png`, `feature_importance_shap.png`.

## 5. Modèles

Comparaison complète : `model_comparison_walkforward.csv`, `model_comparison_final_test.csv`,
`model_comparison.png`.

- Optuna, 100 essais par modèle, validation chronologique, arrêt précoce pour le boosting et le MLP.
  Pour la variante avec cotes, seuls la régression logistique et LightGBM ont été réglés à nouveau ; les
  autres modèles réutilisent leurs hyperparamètres « sans cotes ».
- **Les huit modèles sans cotes tiennent dans 0,002 de log loss** en walk-forward (0,9823 à 0,9845).
  La régression logistique fortement régularisée est en tête ; les modèles d'arbres n'apportent rien de
  plus. Le signal est essentiellement linéaire dans les notes de force.
- Les ensembles n'améliorent pas le meilleur modèle seul (moyenne pondérée 0,9824, stacking 0,9832).
- Sur le test final l'ordre change (XGBoost 0,9873, régression logistique 0,9884) : ces écarts sont du bruit.
- **Calibration** : la mise à l'échelle par température ne change rien (les modèles sont déjà calibrés) ;
  la régression isotonique dégrade la log loss de 0,002 à 0,004. Courbes de fiabilité :
  `calibration_walkforward.png`, `calibration_final_test.png`.
- Matrice de confusion du modèle retenu sur le test final (lignes = réel, colonnes = prédit, H/D/A) :
  612/0/159, 275/0/171, 226/1/308. F1 : domicile 0,65, nul 0,00, extérieur 0,53.
- Par championnat (test final, log loss sans cotes / bookmakers) : Bundesliga 0,965 / 0,950 ;
  Liga 0,974 / 0,962 ; Ligue 1 0,987 / 0,972 ; Serie A 0,991 / 0,982 ; Premier League 1,021 / 1,019.
  La Premier League 2025/26 a été difficile pour tout le monde (48,7 % de précision, 48,9 % pour les bookmakers).

## 6. Clustering : résultat de l'ablation

- K-means sur le profil de style des équipes (saison précédente), k = 4 retenu (silhouette 0,19, stabilité
  ARI 0,92). La structure est faible : les groupes reflètent surtout le niveau et l'agressivité.
  `clustering_k_selection.csv`, `clustering_pca.png`.
- Le clustering est réajusté dans chaque pli, sur les seules données d'entraînement.

Ablation en walk-forward (8 981 matchs, `ablations.csv`) :

| Configuration | Régression logistique | LightGBM |
|---|---|---|
| Complet (K-means) | 0,98231 | 0,98367 |
| Sans clustering | 0,98226 (−0,00005, p = 0,85) | 0,98491 (+0,00125, p = 0,04) |
| GMM à la place | 0,98249 (p = 0,57) | 0,98387 (p = 0,56) |
| Hiérarchique (Ward) | 0,98243 (p = 0,68) | 0,98387 (p = 0,53) |
| Sans Elo | 0,98280 (+0,00049, p = 0,04) | 0,98495 (+0,00128, p = 0,07) |
| Avec cotes | 0,97311 (−0,0092, p < 0,001) | 0,97243 (−0,0112, p < 0,001) |

**Le clustering n'améliore pas le modèle retenu** (effet nul pour la régression logistique). Pour LightGBM
le gain est de 0,001, à la limite de la significativité et sans correction pour comparaisons multiples :
il ne faut pas le tenir pour acquis. K-means, GMM et Ward sont équivalents. Retirer l'Elo coûte peu car
Dixon-Coles porte la même information. Seules les cotes changent réellement le résultat.

## 7. Comparaison aux bookmakers et backtest

- Les cotes de clôture sont meilleures que les cotes d'ouverture Bet365 en walk-forward (−0,0033, p < 0,001).
- Backtest « value » à mise fixe, cotes de clôture, modèle sans cotes (`value_backtest.csv`) :
  **rendement négatif à tous les seuils**, de −4,1 % à −10,2 % en walk-forward (IC entièrement négatifs) et
  de −7,0 % à −13,3 % sur le test final. Plus le modèle est en désaccord avec le marché, plus il perd :
  quand ils divergent, c'est le marché qui a raison.
- Ce backtest est une évaluation, **pas un conseil de pari**.

## 8. Comparaison à l'état de l'art

**Sommes-nous dans les clous ? Oui sur la méthode, et les résultats sont cohérents avec la littérature.**

- Les meilleurs modèles publiés sur données publiques combinent des notes de force d'équipe et du gradient
  boosting : vainqueur du Soccer Prediction Challenge 2017 (Hubáček et al., pi-ratings + XGBoost,
  RPS 0,2063), Berrar et al. (RPS 0,2054), puis CatBoost + pi-ratings (Razali et al., 55,8 % ; Yeung et
  al., 2023). C'est notre architecture : Elo + Dixon-Coles + forme, puis modèles linéaires et boosting.
- Ces RPS ne sont pas directement comparables aux nôtres (0,199 sans cotes) : ces défis couvrent des
  dizaines de championnats, plus difficiles que les cinq grands.
- Tous ces travaux font le même constat que nous : les modèles sur données publiques restent derrière
  les bookmakers, et les méthodes sophistiquées (y compris l'apprentissage profond) n'apportent presque
  rien par rapport à de bonnes notes de force.

**Deux idées publiées testées ici** (`scripts/10_sota_extension.py`, `sota_extension.csv`), en walk-forward :

| Ajout | Régression logistique | LightGBM |
|---|---|---|
| Pi-ratings (Constantinou & Fenton) | −0,00009 (p = 0,51) | +0,00031 (p = 0,61) |
| Forces d'attaque/défense sur tirs cadrés (esprit GAP, Wheatcroft) | +0,00023 (p = 0,29) | +0,00161 (p = 0,02, dégradation) |

Aucune amélioration : ces variables sont redondantes avec Elo, Dixon-Coles et la forme sur les tirs déjà
présents. **Avec les données gratuites de football-data.co.uk, le modèle est au plafond** ; changer
d'algorithme ou ajouter des notes de force ne rapportera plus rien.

**Ce qui sépare réellement notre modèle du marché, c'est l'information, pas l'algorithme.** Pistes par
ordre d'intérêt, aucune testée ici faute de données gratuites et historisées :

1. **Compositions, blessures, suspensions** : c'est l'essentiel de ce que les cotes intègrent et que nous
   ignorons ; les modèles professionnels notent les joueurs, pas les équipes.
2. **Expected goals (xG)** : mesure de la qualité des occasions, nettement moins bruitée que les buts ou
   les tirs (Understat, FBref ; disponibles surtout depuis 2014-2017).
3. **Valeurs marchandes des effectifs** (Transfermarkt) : très utiles en début de saison et pour les promus.
4. **Calendrier européen et enjeu** (rotation, matchs sans enjeu de fin de saison).
5. **Notes dérivées des cotes** (Wunderlich & Memmert) : meilleures que l'Elo classique, mais c'est utiliser
   le marché ; cela relève de la variante « avec cotes ».

Le gain à attendre de ces pistes n'est pas chiffré ici. L'écart à combler est de 0,010 de log loss et les
bookmakers disposent déjà de toutes ces informations : égaler le marché est un objectif réaliste, le
battre durablement ne l'est pas avec des données publiques. Les professionnels se mesurent d'ailleurs à
la cote de clôture (« closing line value ») et non à la précision.

Sources :
- Hubáček, Šourek, Železný (2019), *Machine Learning* : https://link.springer.com/article/10.1007/s10994-018-5763-8
- Yeung et al. (2023) : https://arxiv.org/abs/2309.14807
- Soccer Prediction Challenge 2023, synthèse : https://arxiv.org/pdf/2403.07669
- Wheatcroft (2020), ratings GAP : https://eprints.lse.ac.uk/103712 et https://arxiv.org/abs/2001.09097
- Wunderlich & Memmert (2018), *PLOS ONE* : https://doi.org/10.1371/journal.pone.0198668
- https://arxiv.org/pdf/1908.08980

## 8 bis. Données au niveau joueur : premier gain mesuré

Source : jeu de données ouvert « transfermarkt-datasets » (compositions, minutes, buts et passes par
joueur et par match, historique des valeurs marchandes ; toutes compétitions, 2012/13 à 2025/26,
couverture de 100 % de nos matchs sur cette période). Variables : `src/player_features.py`, test
anti-fuite dédié, expérience `scripts/11_player_features.py`, tableau `player_features_experiment.csv`.

- **Avant-match** (utilisables plusieurs jours avant) : valeur marchande du onze aligné au match
  précédent et des onze joueurs les plus chers de l'effectif récent, buts + passes par 90 minutes de ces
  joueurs sur un an, minutes jouées sur 14 jours (fatigue), repos réel toutes compétitions confondues.
- **Composition officielle** (connue environ une heure avant le coup d'envoi) : mêmes mesures sur le
  onze réellement aligné, nombre de changements, part des meilleurs joueurs présents.

Walk-forward 2020/21 à 2024/25, 8 981 matchs, régression logistique (écart de log loss par rapport au
modèle actuel ; bookmakers Bet365 : 0,9716) :

| Configuration | Précision | Log loss | Écart | p | Retard sur les bookmakers |
|---|---|---|---|---|---|
| Modèle actuel | 52,8 % | 0,9823 | | | +0,0107 |
| + joueurs avant-match | 53,1 % | 0,9809 | −0,0014 [−0,0027 ; −0,0001] | 0,04 | +0,0094 |
| + composition officielle | 53,2 % | 0,9788 | −0,0035 [−0,0052 ; −0,0016] | 0,002 | +0,0073 |
| Avec cotes | 53,3 % | 0,9757 | | | +0,0041 |
| Avec cotes + joueurs + composition | 53,2 % | 0,9756 | ≈ 0 | | +0,0040 |

- **C'est le premier ajout qui améliore le modèle** : la composition officielle comble environ un tiers du
  retard sur les bookmakers ; les variables d'avant-match seules, environ un huitième, à la limite de la
  significativité.
- Avec LightGBM, les variables d'avant-match n'apportent rien (+0,0002) ; seule la composition aide
  (−0,0026, p = 0,04). Le gain d'avant-match est donc fragile.
- **Ajoutées aux cotes, ces variables n'apportent rien** : le marché les intègre déjà.
- Le test final 2025/26 n'a pas été réutilisé pour cette expérience.

Autres sources examinées : Understat (xG par match et par joueur, gratuit) est hors ligne depuis le
2 novembre 2026 ; FBref aurait retiré ses statistiques avancées en janvier 2026 ; l'offre gratuite
d'API-Football ne couvre pas la saison en cours (2022 à 2024 seulement, d'après plusieurs retours
d'utilisateurs).

### Mise en production (prédictions hebdomadaires)

Le jeu de données Transfermarkt n'étant plus mis à jour depuis juillet 2026, la saison en cours est
prolongée ainsi : **compositions via l'API publique d'ESPN** (sans clé, `src/espn.py`), rattachées par
nom aux joueurs Transfermarkt (`src/live_players.py`), avec des **valeurs marchandes figées à juin 2026**.
Seules les variables maintenables dans ces conditions sont utilisées (valeur du onze précédent, des onze
joueurs les plus chers de l'effectif récent, et de la composition officielle quand elle est publiée).

Le gain survit à ces contraintes (`scripts/12_player_features_live.py`, walk-forward, 8 981 matchs) :

| Configuration | Log loss | Écart vs modèle actuel | p |
|---|---|---|---|
| Modèle actuel | 0,9823 | | |
| + valeur du onze avant-match, valeurs figées en juin | 0,9811 | −0,0012 [−0,0024 ; −0,0001] | 0,04 |
| + composition officielle, valeurs figées en juin | 0,9796 | −0,0027 [−0,0043 ; −0,0012] | < 0,001 |

Contrôle sur des données jamais utilisées, les 500 matchs déjà joués de 2026/27, avec la chaîne ESPN
réelle (`scripts/13_live_check_2026.py`) : modèle de base 0,9572 ; + effectifs 0,9550 (−0,0022, p = 0,25) ;
+ composition 0,9530 (−0,0042, p = 0,12) ; bookmakers 0,9447. Même sens et même ordre de grandeur que le
walk-forward, mais non significatif sur un échantillon aussi petit.

Limites de cette chaîne :
- 92 % des titulaires de 2026/27 sont identifiés. Les promus venus de divisions non couvertes (Coventry,
  Elversberg, Le Mans, Málaga, Paderborn, Troyes) ne le sont presque pas : leurs matchs retombent sur le
  modèle de base (76 matchs sur 500).
- Les valeurs marchandes vieillissent : les recrues et les progressions depuis juin 2026 sont ignorées.
- L'API d'ESPN n'est pas documentée officiellement et peut changer ; en cas d'échec, l'interface revient
  au modèle de base et le signale.
- Les modèles joueurs ne sont pas recalibrés (la calibration par température était sans effet).

### Indisponibilités, postes, âge, banc (expérience, non retenue)

Il n'existe pas de source gratuite de blessures avec historique (ESPN n'en publie pas pour le football,
Understat est toujours hors ligne). On a donc testé ce que les feuilles de match permettent de savoir
avant le coup d'envoi (`src/player_ext.py`, `scripts/14_player_ext.py`, `player_ext_experiment.csv`) :
joueurs importants absents de la dernière feuille de match, suspendus (rouge, cumul de jaunes), valeur
du onze par ligne, âge moyen, valeur du banc, stabilité du onze.

Walk-forward, 8 981 matchs, régression logistique, par rapport au modèle d'avant-match en production (0,9811) :

| Ajout | Écart de log loss | p |
|---|---|---|
| Indisponibilités (absents, suspendus) | −0,0006 [−0,0013 ; +0,0002] | 0,13 |
| Valeur par ligne (attaque, milieu, défense) | −0,0001 | 0,66 |
| Âge, banc, stabilité | −0,0005 [−0,0012 ; +0,0001] | 0,08 |
| Indisponibilités + âge, banc, stabilité | −0,0009 [−0,0019 ; +0,0001] | 0,08 |

Les signes sont cohérents (une équipe privée de joueurs importants gagne moins) et le gain apparaît sur
quatre saisons sur cinq, mais **aucun n'est statistiquement établi** ; avec LightGBM l'effet est encore
plus faible. Ces variables ne sont donc pas intégrées au modèle en production. La forme individuelle
(buts et passes par 90 minutes) et la fatigue avaient déjà été testées en 8 bis, sans gain au-delà de la
valeur marchande.

### Enjeu sportif des matchs (retenu)

Variables `src/stakes.py`, calculées avant chaque match à partir du classement : pour le titre, la quatrième
place et le maintien, l'objectif est-il encore ouvert compte tenu des points restant en jeu, et à quel point
est-ce serré ; plus un indicateur « plus rien à jouer ». Test anti-fuite dédié ; expérience
`scripts/15_stakes.py` (`stakes_experiment.csv`), modèle de production `scripts/16_retrain_with_stakes.py`.

Walk-forward 2020/21 à 2024/25 (écart de log loss, par rapport au modèle sans enjeu) :

| Matchs | Régression logistique | LightGBM |
|---|---|---|
| Tous (8 981) | −0,0009 [−0,0020 ; +0,0001], p = 0,07 | −0,0001 (p = 0,90) |
| Dernier quart de saison (2 140) | **−0,0049** [−0,0088 ; −0,0008], p = 0,02 | **−0,0047** [−0,0085 ; −0,0008], p = 0,02 |

L'effet se concentre en fin de saison, comme attendu : une équipe sans objectif (31 % des matchs du dernier
quart concernent au moins une telle équipe) obtient de moins bons résultats que sa force ne le laisse prévoir,
une équipe menacée de relégation légèrement meilleurs. Le modèle de production intègre désormais ces
variables. Sur la saison 2025/26, déjà utilisée comme test final et regardée ici à titre de simple contrôle,
le gain est du même ordre (−0,0009, non significatif sur 1 752 matchs). La précision brute varie peu
(52,6 % contre 52,8 % en walk-forward) : le gain porte sur la qualité des probabilités. L'ancien modèle est
conservé sous `models/best_pipeline_v1.joblib`.

## 9. Limites

- Le test final ne compte que 1 752 matchs : les intervalles de confiance sont larges (± 2,3 points de précision).
- Les saisons 2020/21 et 2021/22 (huis clos) pèsent sur le walk-forward : l'avantage du terrain y a changé.
- La comparaison « avec cotes » sous-estime peut-être légèrement ce que le boosting pourrait faire,
  six modèles sur huit n'ayant pas été réglés à nouveau avec les cotes.
- Sur le test final, la marge moyenne des cotes de clôture est de 4,3 % contre 2,6 % en walk-forward :
  la source de la cote de clôture y est vraisemblablement Bet365 et non Pinnacle.
- Le modèle ne connaît ni les compositions, ni les blessures, ni les transferts en cours de saison.
- Les matchs nuls ne sont jamais prédits comme issue la plus probable.

## 10. Livrables

- `results/` : tableaux CSV, graphiques PNG, ce rapport.
- `models/best_pipeline.joblib` (sans cotes) et `models/best_pipeline_with_odds.joblib`, réentraînés sur
  tout l'historique jusqu'au 2026-11-08.
- `predict.py` : prédit des matchs à venir à partir d'un CSV (`data/fixtures_example.csv`).
