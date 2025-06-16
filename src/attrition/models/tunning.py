# src/attrition/models/tunning.py

import warnings

import joblib
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def run_tuning(
    data_path: str,
    features_path: str,
    target_col: str = "Attrition_Yes",
    n_trials: int = 50,
):
    """
    Executa a otimização de hiperparâmetros com Optuna para o modelo XGBoost.
    """
    print("--- Iniciando Otimização de Hiperparâmetros com Optuna ---")

    # Carregar e preparar os dados
    df = pd.read_csv(data_path)
    features = joblib.load(features_path)
    X = df[features]
    y = df[target_col]

    # Dividir em treino e teste (essencial para o Optuna avaliar no conjunto de teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Aplicar SMOTE apenas nos dados de treino
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

        model = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42, **params
        )

        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)

        return f1_score(y_test, y_pred)

    # Criar e executar o estudo do Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n--- Otimização Concluída ---")
    print(f"Melhor F1-score encontrado: {study.best_value:.4f}")
    print("Melhores parâmetros encontrados:")
    print(study.best_params)
    print("\nCOPIE o dicionário de parâmetros acima e cole no seu script 'train.py'.")

    return study.best_params
