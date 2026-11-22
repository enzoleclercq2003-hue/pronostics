"""Configuration centrale du projet (chemins, saisons, hyperparamètres fixes)."""
from pathlib import Path

# --- Reproductibilité -------------------------------------------------------
SEED = 42

# --- Chemins ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Championnats -----------------------------------------------------------
LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
}
INCLUDE_LIGUE1 = True  # option : ajouter la Ligue 1 (code F1)
if INCLUDE_LIGUE1:
    LEAGUES["F1"] = "Ligue 1"

# --- Saisons (identifiées par l'année de début : 2005 = saison 2005/06) -----
FIRST_SEASON = 2005
CURRENT_SEASON = 2026          # saison en cours (2026/27), incomplète
FINAL_TEST_SEASON = 2025       # dernière saison complète : test final, utilisé une seule fois
WALK_FORWARD_SEASONS = [2020, 2021, 2022, 2023, 2024]   # saisons de test walk-forward
TUNING_SEASONS = [2017, 2018, 2019]   # plis de validation pour Optuna / Elo / choix de k
ALL_SEASONS = list(range(FIRST_SEASON, CURRENT_SEASON + 1))

# --- Version hébergée (build_site.py) ----------------------------------------
# False : le site public n'utilise aucune donnée joueur (compositions ESPN, valeurs Transfermarkt),
# dont les conditions n'autorisent pas un usage commercial. À passer à True seulement avec des
# données sous licence. L'application locale, elle, n'est pas concernée par ce réglage.
PUBLIC_USE_PLAYER_DATA = True

# Application locale : absences annoncées dans l'actualité (flux RSS publics, src/news.py) prises en
# compte à chaque calcul via une composition probable. Mettre à False pour revenir au modèle sans actualité.
USE_NEWS = True

# --- Sources ----------------------------------------------------------------
PRIMARY_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/{league}.csv"
# Source de secours (sans cotes) : dépôt GitHub datasets/football-datasets
FALLBACK_URL = (
    "https://raw.githubusercontent.com/datasets/football-datasets/master/"
    "datasets/{slug}/season-{season_code}.csv"
)
FALLBACK_SLUGS = {
    "E0": "premier-league", "SP1": "la-liga", "I1": "serie-a",
    "D1": "bundesliga", "F1": "ligue-1",
}

# --- Elo (K et avantage du terrain sont optimisés par validation, cf. scripts/02) ---
ELO_INIT = 1500.0
ELO_K = 20.0                   # valeur par défaut, remplacée par la valeur optimisée
ELO_HFA = 60.0                 # idem
ELO_SEASON_CARRY = 0.80        # retour partiel vers la moyenne entre deux saisons
ELO_PROMOTED_OFFSET = -80.0    # note initiale d'une équipe qui entre dans la division
ELO_PARAMS_FILE = PROCESSED_DIR / "elo_params.json"

# --- Forme récente ----------------------------------------------------------
EWM_HALFLIFE = 5
ROLL_WINDOWS = (3, 5, 10)

# --- Dixon-Coles sur fenêtre glissante --------------------------------------
DC_WINDOW_DAYS = 3 * 365       # fenêtre glissante
DC_HALFLIFE_DAYS = 240         # pondération temporelle exponentielle dans la fenêtre
DC_PRIOR_MATCHES = 4.0         # rétrécissement vers l'équipe moyenne (pseudo-matchs)
DC_MAX_GOALS = 10

# --- Clustering -------------------------------------------------------------
CLUSTER_PROFILE_WINDOW = 38    # matchs passés utilisés pour le profil de style
CLUSTER_MIN_MATCHES = 10
CLUSTER_K_RANGE = range(3, 9)
CLUSTER_PARAMS_FILE = PROCESSED_DIR / "cluster_params.json"

# --- Optuna -----------------------------------------------------------------
N_TRIALS = 100
N_JOBS = -1

# --- Divers -----------------------------------------------------------------
CLASSES = ["H", "D", "A"]      # ordre des classes : domicile, nul, extérieur
N_BOOTSTRAP = 2000


def season_code(season: int) -> str:
    """2005 -> '0506' (format de football-data.co.uk)."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def season_label(season: int) -> str:
    """2005 -> '2005/06'."""
    return f"{season}/{(season + 1) % 100:02d}"
