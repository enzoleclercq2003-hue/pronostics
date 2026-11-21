"""Prédictions des matchs de la semaine : tout est automatique.

    python semaine.py            (ou double-clic sur « Predictions de la semaine.bat »)

1. met à jour l'historique des résultats (football-data.co.uk) ;
2. récupère le calendrier des cinq championnats et garde les matchs de la semaine en cours ;
3. calcule les probabilités avec le modèle sauvegardé (sans cotes) ;
4. écrit la page `predictions_semaine.html` et l'ouvre dans le navigateur.

Options : --no-open (ne pas ouvrir le navigateur), --no-refresh (ne rien télécharger côté résultats).
"""
import argparse
import html
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

import config as cfg
from src.data import load_matches, load_processed
from src.fixtures import LOCAL_TZ, load_calendar, load_odds, upcoming_week
from src.odds import implied_shin
from src.pipeline_players import LEVELS, predict_week

OUT_HTML = ROOT / "predictions_semaine.html"
LOG = cfg.RESULTS_DIR / "predictions_log.csv"
DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]


def french_date(ts: pd.Timestamp) -> str:
    return f"{DAYS[ts.day_of_week]} {ts.day} {MONTHS[ts.month - 1]}"


def compute(refresh: bool = True, progress=lambda msg: None, use_players: bool = True) -> tuple[pd.DataFrame, dict]:
    notes = []
    cached = cfg.PROCESSED_DIR / "matches.pkl"
    history = load_processed() if cached.exists() else None
    progress("Mise à jour des derniers résultats…")
    if refresh or history is None:
        try:
            history = load_matches(refresh_current=True, verbose=False)
            history.to_pickle(cached)
        except Exception as exc:                        # pas de réseau : on garde l'historique en cache
            if history is None:
                raise
            notes.append(f"Mise à jour des résultats impossible ({type(exc).__name__}) : historique en cache utilisé.")
    predictor = joblib.load(cfg.MODELS_DIR / "best_pipeline.joblib")
    progress("Récupération du calendrier…")
    calendar = load_calendar(history)
    week, label = upcoming_week(calendar, history)
    info = {"label": label, "notes": notes, "history_end": history.Date.max(), "n_history": len(history),
            "train_end": predictor.train_end, "generated": pd.Timestamp.now(tz=LOCAL_TZ),
            "recent_results": calendar.loc[calendar.played, ["league", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]}
    if week.empty:
        return week, info
    fixtures = pd.DataFrame({"date": week.Date.dt.strftime("%Y-%m-%d"), "home_team": week.HomeTeam, "away_team": week.AwayTeam})
    progress("Compositions des équipes et calcul des probabilités…")
    try:
        pred, report = predict_week(predictor, fixtures, history, use_players=use_players)
        info["player_share"] = report.get("matched_share")
    except Exception as exc:                            # données joueurs indisponibles : modèle de base seul
        notes.append(f"Variables joueurs indisponibles ({type(exc).__name__}: {exc}) : modèle de base utilisé.")
        pred = predictor.predict(fixtures, history).assign(level="base")
    if "detail" not in pred.columns:
        pred["detail"] = None
    out = week.reset_index(drop=True).join(pred[["p_home", "p_draw", "p_away", "level", "elo_home", "elo_away", "detail"]])
    out = out.merge(load_odds(), on=["league", "HomeTeam", "AwayTeam"], how="left")
    has = out["odds_h"].notna().to_numpy()
    for c in ("book_h", "book_d", "book_a"):
        out[c] = np.nan
    if has.any():
        out.loc[has, ["book_h", "book_d", "book_a"]] = implied_shin(out.loc[has, ["odds_h", "odds_d", "odds_a"]].to_numpy(float))
    out["local"] = out.kickoff.dt.tz_convert(LOCAL_TZ)
    return out, info


def log_predictions(df: pd.DataFrame, info: dict, path: Path | None = None):
    """Journal des prédictions émises (une ligne par match, la plus récente gagne) pour un suivi ultérieur.
    Le site public tient son propre journal (`path`) : ses prédictions diffèrent de celles de l'application locale."""
    LOG = path or globals()["LOG"]
    new = df[["league", "Date", "HomeTeam", "AwayTeam", "p_home", "p_draw", "p_away", "level", "book_h", "book_d", "book_a"]].copy()
    new["Date"] = new["Date"].dt.strftime("%Y-%m-%d")
    new["generated"] = info["generated"].strftime("%Y-%m-%d %H:%M")
    if LOG.exists():
        new = pd.concat([pd.read_csv(LOG), new], ignore_index=True)
    new.drop_duplicates(["league", "Date", "HomeTeam", "AwayTeam"], keep="last").to_csv(LOG, index=False)


CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#1c1c1a;--ink2:#5f5e5a;--line:#e6e5e0;--home:#2a78d6;--draw:#b9b8b2;--away:#eb6834;--chip:#f1f0ec}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#f1f0ec;--ink2:#a8a7a1;--line:#33322f;--home:#4a94ec;--draw:#6f6e69;--away:#f08454;--chip:#2a2927}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}.sub{color:var(--ink2);margin:0 0 18px}
h2{font-size:17px;margin:30px 0 10px;display:flex;align-items:baseline;gap:8px}h2 small{color:var(--ink2);font-weight:400;font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--ink2);font-size:13px;margin:0 0 6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.note{background:var(--chip);border-radius:8px;padding:10px 12px;color:var(--ink2);font-size:13px;margin:10px 0}
.match{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:8px 0}
.top{display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap}
.teams{font-weight:600;font-size:16px}.teams .fav{text-decoration:underline;text-decoration-color:var(--line);text-underline-offset:4px}
.when{color:var(--ink2);font-size:13px;white-space:nowrap}
.bar{display:flex;height:26px;border-radius:6px;overflow:hidden;margin:9px 0 6px;gap:2px}
.bar div{display:flex;align-items:center;justify-content:center;color:#fff;font-size:12.5px;font-weight:600;font-variant-numeric:tabular-nums;min-width:0;white-space:nowrap;overflow:hidden}
.bar .h{background:var(--home)}.bar .d{background:var(--draw);color:var(--ink)}.bar .a{background:var(--away)}
.meta{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:var(--ink2);font-size:12.5px;font-variant-numeric:tabular-nums}
.chip{background:var(--chip);border-radius:20px;padding:1px 9px;font-size:12px;color:var(--ink2)}
footer{margin-top:34px;color:var(--ink2);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}
"""


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def render(df: pd.DataFrame, info: dict) -> str:
    e = html.escape
    parts = [f"<!doctype html><html lang='fr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Prédictions de la semaine</title><style>{CSS}</style></head><body><main>",
             "<h1>Prédictions de la semaine</h1>"]
    if df.empty:
        parts.append(f"<p class='sub'>{e(info['label'])}.</p>")
    else:
        first, last = df.local.min(), df.local.max()
        parts.append(f"<p class='sub'>{e(info['label'])} · du {french_date(first)} au {french_date(last)} {last.year} · {len(df)} matchs</p>")
        parts.append("<div class='legend'><span><i style='background:var(--home)'></i>Victoire domicile</span>"
                     "<span><i style='background:var(--draw)'></i>Nul</span><span><i style='background:var(--away)'></i>Victoire extérieur</span></div>")
    for n in info["notes"]:
        parts.append(f"<div class='note'>{e(n)}</div>")
    for league, grp in df.groupby("league", sort=False) if not df.empty else []:
        parts.append(f"<h2>{e(cfg.LEAGUES[league])} <small>{len(grp)} matchs</small></h2>")
        for r in grp.sort_values("kickoff").itertuples():
            p = np.array([r.p_home, r.p_draw, r.p_away])
            fav = int(p.argmax())
            home = f"<span class='fav'>{e(r.HomeTeam)}</span>" if fav == 0 else e(r.HomeTeam)
            away = f"<span class='fav'>{e(r.AwayTeam)}</span>" if fav == 2 else e(r.AwayTeam)
            verdict = ["Favori : " + r.HomeTeam, "Issue la plus probable : nul", "Favori : " + r.AwayTeam][fav]
            if p.max() < 0.42:
                verdict = "Match ouvert, pas de favori net"
            book = (f"Bookmakers : {pct(r.book_h)} · {pct(r.book_d)} · {pct(r.book_a)}" if not np.isnan(r.book_h)
                    else "Cotes pas encore publiées")
            parts.append(
                f"<div class='match'><div class='top'><div class='teams'>{home} – {away}</div>"
                f"<div class='when'>{french_date(r.local)} · {r.local:%H:%M}</div></div>"
                f"<div class='bar' role='img' aria-label='domicile {pct(p[0])}, nul {pct(p[1])}, extérieur {pct(p[2])}'>"
                + "".join(f"<div class='{c}' style='flex:{v:.4f}' title='{pct(v)}'>{pct(v) if v >= 0.07 else ''}</div>"
                          for c, v in zip("hda", p)) + "</div>"
                f"<div class='meta'><span>{e(verdict)}</span><span>{e(book)}</span></div>"
                f"<div class='meta' style='margin-top:6px'><span class='chip'>{e(LEVELS[r.level])}</span></div></div>")
    parts.append(
        f"<footer>Page générée le {info['generated']:%d/%m/%Y à %H:%M}. Résultats connus jusqu'au {info['history_end']:%d/%m/%Y} "
        f"({info['n_history']} matchs) ; modèle entraîné jusqu'au {info['train_end']:%d/%m/%Y}.<br>"
        "Modèle sans cotes (régression logistique) : environ 53 % de bons pronostics en test, un peu moins précis que "
        "les bookmakers. Il tient compte de la valeur marchande des onze alignés (valeurs figées à juin 2026) et, "
        "si elle est publiée au moment du calcul, de la composition officielle : relancez environ une heure avant "
        "un match pour l'intégrer. Il ignore les blessures et suspensions annoncées. "
        "Les probabilités sont une estimation statistique, pas un conseil de pari.<br>Heures : Europe/Paris.</footer></main></body></html>")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args()
    df, info = compute(refresh=not args.no_refresh)
    OUT_HTML.write_text(render(df, info), encoding="utf-8")
    if not df.empty:
        log_predictions(df, info)
        show = df.assign(quand=df.local.dt.strftime("%a %d/%m %H:%M"))[["league", "quand", "HomeTeam", "AwayTeam", "p_home", "p_draw", "p_away"]]
        print(f"{info['label']} : {len(df)} matchs")
        print(show.to_string(index=False, float_format="%.3f"))
    else:
        print(info["label"])
    print(f"\nPage écrite : {OUT_HTML}")
    if not args.no_open:
        webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
