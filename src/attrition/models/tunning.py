# src/attrition/models/tunning.py
import json
import logging
import os
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y):
    """Define a função objetivo para o Optuna otimizar o XGBoost com foco no recall."""

    count_neg, count_pos = y.value_counts()
    scale_pos_weight_value = count_neg / count_pos

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight_value,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
    }
    model = xgb.XGBClassifier(**params)

    pipeline = model

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    score = cross_val_score(pipeline, X, y, cv=cv, scoring="recall", n_jobs=-1).mean()
    return score


def run_tuning(X_train, y_train, n_trials: int, output_path: str):
    """Executa a otimização e salva os melhores parâmetros."""
    logging.info(
        f"Iniciando a otimização de hiperparâmetros com {n_trials} tentativas, focando em RECALL..."
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    best_params = study.best_params
    logging.info(f"Melhores parâmetros encontrados: {best_params}")
    logging.info(f"Melhor Recall (Validação Cruzada): {study.best_value:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params, f)
    logging.info(f"✅ Melhores parâmetros salvos em: {output_path}")
