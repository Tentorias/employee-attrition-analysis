# src/attrition/models/tunning.py (Ajustado para SMOTE)
import argparse
import json
import logging
import os
import joblib
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE # <-- MUDANÇA: Voltando para SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, X, y):
    """Define a função objetivo para o Optuna otimizar o XGBoost com SMOTE."""
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }
    model = xgb.XGBClassifier(**params)
    
    pipeline = ImbPipeline([
        ('balancer', SMOTE(random_state=42)), # <-- MUDANÇA: Usando SMOTE
        ('classifier', model)
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=-1).mean()
    return score

def run_tuning(data_path: str, features_path: str, target_col: str, n_trials: int, output_path: str):
    """Executa a otimização e salva os melhores parâmetros."""
    logging.info("Iniciando a otimização de hiperparâmetros para XGBoost com SMOTE...")
    df = pd.read_csv(data_path)
    features = joblib.load(features_path)
    X = df[features]
    y = df[target_col]

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    best_params = study.best_params
    logging.info(f"Melhores parâmetros encontrados: {best_params}")
    logging.info(f"Melhor F1-Score (CV): {study.best_value:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params, f)
    logging.info(f"✅ Melhores parâmetros salvos em: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Otimização de Hiperparâmetros com Optuna para XGBoost.")
    parser.add_argument("--data-path", required=True, help="Caminho para o features_matrix.csv")
    parser.add_argument("--features-path", required=True, help="Caminho para o features.pkl")
    parser.add_argument("--output-path", required=True, help="Caminho para salvar o ficheiro JSON com os melhores parâmetros.")
    parser.add_argument("--target-col", default="Attrition", help="Nome da coluna alvo.")
    parser.add_argument("--n-trials", type=int, default=100, help="Número de tentativas do Optuna.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parser.parse_args()
    run_tuning(
        data_path=args.data_path,
        features_path=args.features_path,
        target_col=args.target_col,
        n_trials=args.n_trials,
        output_path=args.output_path,
    )