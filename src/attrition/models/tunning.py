# src/attrition/models/tunning.py

import argparse
import logging
import warnings

import joblib
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def run_tuning(
    data_path: str,
    features_path: str,
    # --- CORREÇÃO 1: Alinhar o nome da coluna alvo ---
    target_col: str = "Attrition", # Alterado de "Attrition_Yes"
    n_trials: int = 50,
):
    """
    Executa a otimização de hiperparâmetros com Optuna para o modelo XGBoost.
    """
    logger.info("--- Iniciando Otimização de Hiperparâmetros com Optuna ---")

    df = pd.read_csv(data_path)
    features = joblib.load(features_path)
    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    def objective(trial):
        """Função objetivo que o Optuna tentará maximizar."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        }

        # --- CORREÇÃO 2: Adicionar parâmetros essenciais ao XGBClassifier ---
        model = XGBClassifier(
            objective="binary:logistic",
            base_score=0.5,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            **params
        )

        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)

        return f1_score(y_test, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("\n--- Otimização Concluída ---")
    logger.info(f"Melhor F1-score encontrado: {study.best_value:.4f}")
    logger.info("Melhores parâmetros encontrados:")
    logger.info(study.best_params)
    logger.info("\nCOPIE o dicionário de parâmetros acima e cole no seu script 'train.py'.")

    return study.best_params

# --- MELHORIA: Adicionar interface de linha de comando ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Roda a otimização de hiperparâmetros com Optuna."
    )
    parser.add_argument("--data-path", required=True, help="Caminho para o features_matrix.csv")
    parser.add_argument("--features-path", required=True, help="Caminho para o features.pkl")
    parser.add_argument("--target-col", default="Attrition", help="Nome da coluna alvo")
    parser.add_argument("--n-trials", type=int, default=50, help="Número de tentativas do Optuna")
    
    args = parser.parse_args()

    run_tuning(
        data_path=args.data_path,
        features_path=args.features_path,
        target_col=args.target_col,
        n_trials=args.n_trials,
    )
