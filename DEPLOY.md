# Mettre le site en ligne

La version hébergée est un site **statique** : un dossier `site/` (page, police, icône, `data.json`)
fabriqué par `python build_site.py`. Il n'y a pas de serveur à maintenir. Les prédictions sont
recalculées par une tâche planifiée gratuite (GitHub Actions) qui republie le dossier.

## Ce que contient la version publique

| | Application locale | Site public |
|---|---|---|
| Écussons des clubs | oui | **non** (pastille avec l'initiale) |
| Compositions ESPN, valeurs Transfermarkt | oui | **non** (`PUBLIC_USE_PLAYER_DATA = False` dans `config.py`) |
| Modèle | base + valeur des effectifs | modèle de base (52,5 % au test final) |
| Fiche de match | forme, Elo, scores, absents | forme, Elo, scores |
| Mise à jour | à l'ouverture / bouton | automatique, trois fois par jour |
| Suivi | `results/predictions_log.csv` | `results/predictions_log_public.csv` |

L'onglet « Modèle » du site ne décrit que ce qui est réellement utilisé (la section sur les effectifs est retirée).

## Essayer en local

```bash
python build_site.py
python -m http.server 8000 --directory site
```

Puis ouvrir http://127.0.0.1:8000.

## Publier avec GitHub Pages (gratuit)

1. Créer un compte GitHub et un dépôt vide (public, ou privé avec une offre qui inclut Pages).
2. Dans le dossier du projet :
   ```bash
   git init -b main
   git add .
   git commit -m "Première version"
   git remote add origin https://github.com/<compte>/<depot>.git
   git push -u origin main
   ```
   Le fichier `.gitignore` écarte les données téléchargées, les écussons et les fichiers volumineux.
3. Sur GitHub : **Settings → Pages → Build and deployment → Source : GitHub Actions**.
4. Onglet **Actions** : la tâche « Mettre à jour le site » se lance toute seule après l'envoi (ou via
   « Run workflow »). Comptez trois à cinq minutes ; l'adresse du site s'affiche à la fin.

Ensuite, rien à faire : la tâche tourne à 5 h 15, 11 h 15 et 17 h 15 UTC
(`.github/workflows/site.yml`), recalcule les prédictions, enregistre le journal de suivi dans le
dépôt et republie le site.

Autres hébergeurs : tout service de fichiers statiques convient (Netlify, Cloudflare Pages…). Il suffit
d'y déposer le contenu de `site/` ; seule la planification change.

## Nom de domaine

Acheter le domaine chez un registraire, puis le déclarer dans **Settings → Pages → Custom domain** et
créer l'enregistrement DNS indiqué par GitHub. Le HTTPS est fourni.

## À vérifier avant d'ouvrir au public

Ce guide ne remplace pas un avis juridique.

- **Sources de données.** Le site public s'appuie sur football-data.co.uk (résultats, cotes) et
  fixturedownload.com (calendrier). Lire leurs conditions d'utilisation pour un usage public ou
  commercial, et les créditer si elles le demandent.
- **Cotes des bookmakers.** Elles sont affichées quand elles sont publiées. Les retirer si le
  positionnement choisi doit rester à distance des paris.
- **Noms de clubs.** Ils sont utilisés pour désigner les équipes, sans logo ni habillage officiel.
- **Mentions obligatoires.** Mentions légales, politique de confidentialité, et selon le pays les
  messages de prévention liés aux jeux d'argent. Le pied de page indique déjà « estimation statistique,
  pas un conseil de pari ».
- **Modèle sauvegardé.** `models/best_pipeline.joblib` doit être relu avec les versions de
  `requirements-site.txt`. Après un réentraînement, republier le fichier.
- **Changement de saison.** Mettre à jour `CURRENT_SEASON` dans `config.py` chaque été.
