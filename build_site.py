"""Construit la version hébergée du site : un dossier `site/` entièrement statique.

    python build_site.py              calcule les prédictions puis écrit site/
    python build_site.py --no-compute réutilise le dernier calcul (data/processed/week_public.json)

Le dossier `site/` se dépose tel quel sur n'importe quel hébergeur de fichiers statiques
(GitHub Pages, Netlify, Cloudflare Pages...). Il n'y a pas de serveur : la page lit `data.json`,
régénéré par ce script. La mise à jour automatique est décrite dans DEPLOY.md.

Différences avec l'application locale (réglées dans config.py) :
- aucun écusson de club (marques déposées) : une pastille avec l'initiale à la place ;
- par défaut aucune donnée joueur (compositions ESPN, valeurs Transfermarkt : non utilisables
  commercialement) : le modèle de base seul, à partir des résultats et statistiques de match.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config as cfg

SITE = ROOT / "site"
CACHE = cfg.PROCESSED_DIR / "week_public.json"
PUBLIC_LOG = cfg.RESULTS_DIR / "predictions_log_public.csv"     # journal de suivi du site (versionné, cf. DEPLOY.md)


def compute_payload(use_players: bool) -> dict:
    import app
    from semaine import compute, log_predictions
    df, info = compute(refresh=True, progress=lambda m: print(" ", m, flush=True), use_players=use_players)
    if len(df):
        log_predictions(df, info, PUBLIC_LOG)
    payload = app.build_payload(df, info, PUBLIC_LOG)
    payload["public"] = True
    CACHE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def build_page(use_players: bool) -> str:
    """Page de l'application adaptée à l'hébergement : chemins relatifs, mode « site », sans écran local."""
    html = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    for old, new in (('href="/icon.png"', 'href="icon.png"'), ('url("/fonts/', 'url("fonts/')):
        assert old in html, f"repère introuvable dans la page : {old}"
        html = html.replace(old, new)
    flag = f"<script>window.SITE = {json.dumps({'players': use_players})};</script>\n"
    meta = ('<meta name="description" content="Probabilités des matchs de football des cinq grands championnats, '
            'calculées par un modèle statistique documenté. Estimation, pas un conseil de pari.">\n')
    assert "<style>" in html
    return html.replace("<style>", meta + flag + "<style>", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-compute", action="store_true", help="réutiliser le dernier calcul")
    args = parser.parse_args()
    use_players = bool(getattr(cfg, "PUBLIC_USE_PLAYER_DATA", False))
    if args.no_compute and CACHE.exists():
        payload = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        payload = compute_payload(use_players)

    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "fonts").mkdir(parents=True)
    (SITE / "index.html").write_text(build_page(use_players), encoding="utf-8")
    (SITE / "data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    shutil.copy(ROOT / "app" / "team_colors.json", SITE / "colors.json")     # couleurs des pastilles d'équipes
    shutil.copy(ROOT / "app" / "icon.png", SITE / "icon.png")
    shutil.copy(ROOT / "app" / "icon.ico", SITE / "favicon.ico")
    shutil.copy(ROOT / "app" / "fonts" / "InterVariable.woff2", SITE / "fonts" / "InterVariable.woff2")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")          # GitHub Pages : servir les fichiers tels quels
    (SITE / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) / 1024
    print(f"Site construit dans {SITE} : {len(payload['matches'])} matchs, {size:.0f} Ko, "
          f"données joueurs {'incluses' if use_players else 'exclues'}, sans écussons.")


if __name__ == "__main__":
    main()
