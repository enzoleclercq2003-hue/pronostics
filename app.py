"""Application locale « Pronostics Foot » : un petit serveur web sur cet ordinateur.

Lancement : double-clic sur le raccourci « Pronostics Foot » (ou `python app.py`).
- ouvre l'interface dans le navigateur (http://127.0.0.1:8765) ;
- affiche tout de suite les dernières prédictions calculées ;
- les recalcule automatiquement si elles datent de plus de 3 heures, ou sur le bouton « Actualiser » ;
- s'arrête tout seul quelques minutes après la fermeture de l'onglet.

Le serveur n'écoute que sur cet ordinateur (127.0.0.1) : rien n'est accessible depuis l'extérieur.
"""
import json
import re
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
HOST, PORT = "127.0.0.1", 8765
URL = f"http://{HOST}:{PORT}/"
CACHE = ROOT / "data" / "processed" / "week.json"
STALE_HOURS = 3            # au-delà, on recalcule à l'ouverture
IDLE_SECONDS = 600         # arrêt automatique sans onglet ouvert
VALUE_EDGE = 0.05          # écart minimal affiché entre modèle et bookmakers

STATE = {"running": False, "message": "", "error": None, "started": None, "last_seen": time.time()}
LOCK = threading.Lock()


def already_running() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def open_ui():
    """Ouvre l'interface dans une fenêtre d'application (sans barre d'adresse ni onglets) si Edge ou
    Chrome est installé, sinon dans le navigateur par défaut."""
    import os
    import subprocess
    candidates = [os.path.join(os.environ.get(v, ""), p) for v in ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData")
                  for p in (r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe")]
    for exe in candidates:
        if os.path.isfile(exe):
            try:
                subprocess.Popen([exe, f"--app={URL}", "--window-size=1320,900"])
                return
            except OSError:
                pass
    webbrowser.open(URL)


def build_payload(df, info, log_path=None) -> dict:
    """Transforme le résultat du calcul en données pour l'interface."""
    import numpy as np
    import pandas as pd

    import config as cfg
    from semaine import LOG, french_date
    LOG = log_path or LOG
    from src.data import load_processed
    from src.pipeline_players import LEVELS

    matches = []
    for r in df.itertuples() if len(df) else []:
        p = [float(r.p_home), float(r.p_draw), float(r.p_away)]
        has_odds = not np.isnan(r.odds_h)
        odds = [float(r.odds_h), float(r.odds_d), float(r.odds_a)] if has_odds else None
        matches.append({
            "league": r.league, "league_name": cfg.LEAGUES[r.league], "kickoff": r.kickoff.isoformat(),
            "day": french_date(r.local), "day_key": f"{r.local:%Y-%m-%d}", "time": f"{r.local:%H:%M}",
            "home": r.HomeTeam, "away": r.AwayTeam, "p": p, "level": r.level, "level_label": LEVELS[r.level],
            "book": [float(r.book_h), float(r.book_d), float(r.book_a)] if has_odds else None, "odds": odds,
            "edge": [p[i] * odds[i] - 1 for i in range(3)] if has_odds else None,
            "elo": [None if np.isnan(r.elo_home) else round(float(r.elo_home)), None if np.isnan(r.elo_away) else round(float(r.elo_away))],
            "detail": r.detail if isinstance(r.detail, dict) else None})

    # Suivi : prédictions passées confrontées aux résultats réels
    tracking = {"rows": [], "n": 0}
    if LOG.exists():
        log = pd.read_csv(LOG, parse_dates=["Date"])
        hist = load_processed()[["league", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]
        recent = info.get("recent_results")
        if recent is not None and len(recent):
            # football-data.co.uk publie les résultats avec un à trois jours de retard : on complète avec le calendrier
            recent = recent.dropna(subset=["FTHG", "FTAG"]).copy()
            recent["FTR"] = np.where(recent.FTHG > recent.FTAG, "H", np.where(recent.FTHG < recent.FTAG, "A", "D"))
            season = lambda d: d.year if d.month >= 7 else d.year - 1          # une affiche par saison et par terrain
            known = {(l, h, a, season(d)) for l, h, a, d in zip(hist.league, hist.HomeTeam, hist.AwayTeam, hist.Date)}
            recent = recent[[(l, h, a, season(d)) not in known for l, h, a, d in zip(recent.league, recent.HomeTeam, recent.AwayTeam, recent.Date)]]
            hist = pd.concat([hist, recent[hist.columns]], ignore_index=True)
        j = log.merge(hist, on=["league", "HomeTeam", "AwayTeam"], suffixes=("", "_res"))
        j = j[(j.Date_res - j.Date).abs().dt.days <= 3].sort_values("Date_res", ascending=False)
        if len(j):
            p = j[["p_home", "p_draw", "p_away"]].to_numpy(float)
            y = j.FTR.map({"H": 0, "D": 1, "A": 2}).to_numpy(int)
            hit = p.argmax(axis=1) == y
            tracking = {"n": int(len(j)), "hits": int(hit.sum()), "accuracy": float(hit.mean()),
                        "logloss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())}
            b = j[["book_h", "book_d", "book_a"]].to_numpy(float)
            ok = ~np.isnan(b).any(axis=1)
            if ok.any():
                tracking["book_n"] = int(ok.sum())
                tracking["book_accuracy"] = float((b[ok].argmax(axis=1) == y[ok]).mean())
                tracking["model_accuracy_same"] = float(hit[ok].mean())
            book_hit = np.where(ok, np.nan_to_num(b).argmax(axis=1) == y, False)
            tracking["rows"] = [{"date": f"{d:%d/%m}", "iso": f"{d:%Y-%m-%d}", "league": cfg.LEAGUES[l], "home": h, "away": a,
                                 "score": f"{int(hg)}-{int(ag)}", "p": [float(x) for x in pr], "result": int(res), "hit": bool(ok_),
                                 "book_hit": bool(bh) if has_b else None, "book": [float(x) for x in bp] if has_b else None}
                                for d, l, h, a, hg, ag, pr, res, ok_, bh, has_b, bp in zip(j.Date_res, j.league, j.HomeTeam, j.AwayTeam,
                                                                                          j.FTHG, j.FTAG, p, y, hit, book_hit, ok, b)][:400]

    backtest = None
    path = cfg.RESULTS_DIR / "value_backtest.csv"
    if path.exists():
        bt = pd.read_csv(path)
        backtest = {"roi_min": float(bt.roi.min()), "roi_max": float(bt.roi.max())}
    return {"label": info["label"], "notes": info["notes"], "generated": info["generated"].isoformat(),
            "generated_text": f"{info['generated']:%d/%m/%Y à %H:%M}", "history_end": f"{info['history_end']:%d/%m/%Y}",
            "n_history": int(info["n_history"]), "player_share": info.get("player_share"),
            "matches": matches, "tracking": tracking, "backtest": backtest, "value_edge": VALUE_EDGE,
            "leagues": [{"code": c, "name": n} for c, n in cfg.LEAGUES.items()]}


def refresh_job():
    def progress(msg):
        STATE["message"] = msg
    try:
        from semaine import compute, log_predictions
        df, info = compute(refresh=True, progress=progress)
        if len(df):
            log_predictions(df, info)
        progress("Mise en forme…")
        try:                                              # écussons des nouvelles équipes (sans gravité si échec)
            from src.assets import ensure_team_logos
            from src.data import load_processed
            ensure_team_logos(load_processed())
        except Exception:
            traceback.print_exc()
        CACHE.write_text(json.dumps(build_payload(df, info), ensure_ascii=False), encoding="utf-8")
        STATE["error"] = None
    except Exception as exc:
        traceback.print_exc()
        STATE["error"] = f"{type(exc).__name__} : {exc}"
    finally:
        STATE["running"], STATE["message"] = False, ""


def start_refresh() -> bool:
    with LOCK:
        if STATE["running"]:
            return False
        STATE.update(running=True, message="Démarrage…", error=None, started=time.time())
    threading.Thread(target=refresh_job, daemon=True).start()
    return True


def cache_age_hours() -> float | None:
    return (time.time() - CACHE.stat().st_mtime) / 3600 if CACHE.exists() else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200, cache: str = "no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", code)

    def _local(self) -> bool:
        return self.headers.get("Host", "") in (f"{HOST}:{PORT}", f"localhost:{PORT}")

    def do_GET(self):
        if not self._local():
            return self._json({"error": "hôte refusé"}, 403)
        STATE["last_seen"] = time.time()
        self.path = self.path.split("?", 1)[0]
        if self.path in ("/", "/index.html"):
            return self._send((ROOT / "app" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path in ("/icon.png", "/favicon.ico"):
            icon = ROOT / "app" / ("icon.png" if self.path == "/icon.png" else "icon.ico")
            if icon.exists():
                return self._send(icon.read_bytes(), "image/png" if icon.suffix == ".png" else "image/x-icon", cache="max-age=3600")
        m = re.fullmatch(r"/(logos|fonts)/([\w-]+\.(png|json|woff2))", self.path)
        if m and (ROOT / "app" / m.group(1) / m.group(2)).exists():       # nom de fichier simple uniquement
            ctype = {"png": "image/png", "json": "application/json; charset=utf-8", "woff2": "font/woff2"}[m.group(3)]
            return self._send((ROOT / "app" / m.group(1) / m.group(2)).read_bytes(), ctype,
                              cache="no-store" if m.group(3) == "json" else "max-age=86400")
        if self.path == "/api/data":
            if not CACHE.exists():
                return self._json({"empty": True})
            return self._send(CACHE.read_bytes(), "application/json; charset=utf-8")
        if self.path == "/api/status":
            age = cache_age_hours()
            return self._json({"running": STATE["running"], "message": STATE["message"], "error": STATE["error"],
                               "elapsed": time.time() - STATE["started"] if STATE["running"] else None,
                               "age_hours": age, "stale_hours": STALE_HOURS})
        self._json({"error": "introuvable"}, 404)

    def do_POST(self):
        # En-tête personnalisé exigé : une page d'un autre site ne peut pas déclencher ces actions
        if not self._local() or self.headers.get("X-App") != "pronostics":
            return self._json({"error": "refusé"}, 403)
        STATE["last_seen"] = time.time()
        if self.path == "/api/refresh":
            return self._json({"started": start_refresh()})
        if self.path == "/api/quit":
            self._json({"bye": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self._json({"error": "introuvable"}, 404)


def watchdog(server):
    while True:
        time.sleep(30)
        if not STATE["running"] and time.time() - STATE["last_seen"] > IDLE_SECONDS:
            server.shutdown()
            return


def main():
    if sys.stdout is None or sys.stderr is None:          # lancé sans console (pythonw) : journal dans un fichier
        sys.stdout = sys.stderr = open(ROOT / "app.log", "a", encoding="utf-8", buffering=1)
    if already_running():
        open_ui()
        return
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    age = cache_age_hours()
    if age is None or age > STALE_HOURS:
        start_refresh()
    threading.Thread(target=watchdog, args=(server,), daemon=True).start()
    if "--no-open" not in sys.argv:
        open_ui()
    print(f"Pronostics Foot : {URL} (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
