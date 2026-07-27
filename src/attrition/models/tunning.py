# src/attrition/models/tunning.py
import json
import logging
import os

import optuna
import xgboost as xgb
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y):
    """
    Função objetivo para o Optuna otimizar o XGBoost dentro de um Pipeline
    com SMOTEENN, evitando Data Leakage entre os folds de validação cruzada.
    """

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
    }

    model = xgb.XGBClassifier(**params)

    # Encapsula SMOTEENN no Pipeline para ser executado apenas nos folds de treino (evita Data Leakage)
    pipeline = Pipeline(
        [("smoteenn", SMOTEENN(random_state=42)), ("xgb", model)]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(
        pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1
    ).mean()

    return score


def run_tuning(X_train, y_train, n_trials: int, output_path: str):
    """
    Executa a otimização de hiperparâmetros sem Data Leakage e salva os melhores parâmetros.
    Garante que X_train não foi previamente resampleado.
    """
    logging.info(
        f"Iniciando otimização com {n_trials} tentativas (Search Space ajustado e sem Data Leakage)..."
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), n_trials=n_trials
    )

    best_params = study.best_params
    logging.info(f"Melhores parâmetros encontrados: {best_params}")
    logging.info(f"Melhor F1-Score (CV Stratified): {study.best_value:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params, f)
    logging.info(f"✅ Melhores parâmetros salvos em: {output_path}")
