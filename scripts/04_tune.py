"""Étape 4 : réglage des hyperparamètres avec Optuna (validation chronologique, log loss).

Plis : entraînement sur les saisons < N, validation sur la saison N, pour N dans TUNING_SEASONS.
Les saisons de test walk-forward (>= 2020) et le test final ne sont jamais consultés.
Usage : python scripts/04_tune.py [modèles...] [--variant no_odds|odds] [--threads 6] [--trials 100]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna

import config as cfg
from src.features import select_features
from src.folds import load_fold, split
from src.models import EARLY_STOP_MODELS, MODEL_NAMES, PARAMS_DIR, LeagueMeanImputer, Model

parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="*", default=MODEL_NAMES)
parser.add_argument("--variant", default="no_odds", choices=["no_odds", "odds"])
parser.add_argument("--threads", type=int, default=6)
parser.add_argument("--trials", type=int, default=cfg.N_TRIALS)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial: optuna.Trial, name: str) -> dict:
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-5, 10, log=True)}
    if name in ("rf", "et"):
        return {"n_estimators": 250,
                "max_depth": trial.suggest_int("max_depth", 3, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 300, log=True),
                "max_features": trial.suggest_float("max_features", 0.05, 0.7),
                "max_samples": trial.suggest_float("max_samples", 0.2, 1.0)}
    if name == "hgb":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 3, 40),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 400, log=True),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 100, log=True),
                "max_features": trial.suggest_float("max_features", 0.2, 1.0)}
    if name == "lgbm":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 3, 48),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True)}
    if name == "xgb":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 1, 7),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 200, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100, log=True),
                "gamma": trial.suggest_float("gamma", 1e-3, 5, log=True)}
    if name == "catboost":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
                "depth": trial.suggest_int("depth", 2, 7),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100, log=True),
                "rsm": trial.suggest_float("rsm", 0.2, 1.0)}
    if name == "mlp":
        return {"n_layers": trial.suggest_int("n_layers", 1, 2),
                "units": trial.suggest_int("units", 8, 128, log=True),
                "alpha": trial.suggest_float("alpha", 1e-4, 30, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 3e-4, 5e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512])}
    raise ValueError(name)


# Données des plis, imputées une seule fois (imputeur ajusté sur l'entraînement de chaque pli)
folds = []
for season in cfg.TUNING_SEASONS:
    fold = load_fold(season)
    features = select_features(fold.columns, with_odds=args.variant == "odds")
    train, val = split(fold, season)
    imp = LeagueMeanImputer().fit(train[features])
    folds.append((imp.transform(train[features]), train["y"].to_numpy(int), imp.transform(val[features]), val["y"].to_numpy(int)))
print(f"Variante {args.variant} : {len(features)} variables, plis de validation {cfg.TUNING_SEASONS}, "
      f"tailles entraînement {[len(f[1]) for f in folds]}, validation {[len(f[3]) for f in folds]}", flush=True)

for name in args.models:
    out_path = PARAMS_DIR / f"{name}_{args.variant}.json"
    if out_path.exists() and not args.force:
        print(f"[{name}] déjà réglé, ignoré", flush=True)
        continue

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, name)
        losses, iters = [], []
        for i, (Xtr, ytr, Xva, yva) in enumerate(folds):
            model = Model(name, params, features, n_threads=args.threads).fit_arrays(Xtr, ytr, Xva, yva)
            p = np.clip(model.predict_proba_arrays(Xva), 1e-12, 1)
            losses.append(float(-np.log(p[np.arange(len(yva)), yva]).mean()))
            iters.append(model.best_iter_)
            trial.report(float(np.mean(losses)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if name in EARLY_STOP_MODELS:
            trial.set_user_attr("n_iter", int(np.median(iters)))
        trial.set_user_attr("fold_losses", losses)
        return float(np.mean(losses))

    t0 = time.time()
    # Étude persistée (SQLite) : une exécution interrompue reprend là où elle s'était arrêtée
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0),
                                study_name=f"{name}_{args.variant}", load_if_exists=True,
                                storage=f"sqlite:///{(PARAMS_DIR / f'optuna_{name}_{args.variant}.db').as_posix()}")
    done = sum(t.state.is_finished() for t in study.trials)
    if done < args.trials:
        study.optimize(objective, n_trials=args.trials - done)
    best = study.best_trial
    params = suggest(optuna.trial.FixedTrial(best.params), name)
    if name in ("rf", "et"):
        params["n_estimators"] = 500           # davantage d'arbres pour le modèle final
    if "n_iter" in best.user_attrs:
        params["n_iter"] = best.user_attrs["n_iter"]   # nombre d'itérations fixé par l'arrêt précoce
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    out_path.write_text(json.dumps({
        "model": name, "variant": args.variant, "val_logloss": best.value,
        "fold_losses": best.user_attrs["fold_losses"], "n_trials": len(study.trials), "n_pruned": n_pruned,
        "minutes": (time.time() - t0) / 60, "params": params}, indent=2))
    print(f"[{name}] {len(study.trials)} essais ({n_pruned} élagués) en {(time.time() - t0) / 60:.1f} min | "
          f"log loss validation = {best.value:.5f} | {params}", flush=True)
