import argparse
import logging
import os
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")

def train_model(X, y, random_state=42):
    """
    Treina um modelo de classificação XGBoost com balanceamento de dados SMOTE.
    """
    k_neighbors_smote = 5
    n_minority_samples = y.value_counts().min()

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': random_state,
        'n_jobs': -1
    }

    if n_minority_samples > k_neighbors_smote:
        logging.info("Amostras suficientes. Treinando o modelo com XGBoost e SMOTE...")
        model = Pipeline([
            ("smote", SMOTE(random_state=random_state, k_neighbors=k_neighbors_smote)),
            ("classifier", XGBClassifier(**xgb_params))
        ])
    else:
        logging.warning(f"Não foi possível usar SMOTE. Treinando XGBoost sem SMOTE.")
        model = XGBClassifier(**xgb_params)

    model.fit(X, y)
    return model

def optimize_threshold(model, X_val, y_val):
    logging.info("Otimizando o threshold de classificação...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0, 1.01, 0.01)
    f1_scores = [f1_score(y_val, y_pred_proba >= t) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    logging.info(f"Threshold otimizado encontrado: {optimal_threshold:.2f} (F1-Score: {f1_scores[optimal_idx]:.4f})")
    return optimal_threshold

def main(in_path, features_path, model_path, threshold_path=None, target_col="Attrition", x_test_out=None, y_test_out=None, test_size=0.2, random_state=42, retrain_full_data=False, params_path=None):
    logging.info(f"Lendo a matriz de features de: {in_path}")
    df = pd.read_csv(in_path)
    logging.info(f"Lendo a lista de features de: {features_path}")
    features = joblib.load(features_path)
    X = df[features]
    y = df[target_col]

    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Diretório criado: {directory}")

    if retrain_full_data:
        logging.info("Retreinando o modelo com todos os dados disponíveis...")
        model = train_model(X, y, random_state)
        ensure_dir(model_path)
        joblib.dump(model, model_path)
        logging.info(f"Modelo final salvo em: {model_path}")
    else:
        logging.info("Dividindo os dados em conjuntos de treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        model = train_model(X_train, y_train, random_state)
        ensure_dir(model_path)
        joblib.dump(model, model_path)
        logging.info(f"Modelo treinado salvo em: {model_path}")
        if threshold_path:
            optimal_thr = optimize_threshold(model, X_test, y_test)
            ensure_dir(threshold_path)
            joblib.dump(optimal_thr, threshold_path)
            logging.info(f"Threshold otimizado salvo em: {threshold_path}")
        if x_test_out and y_test_out:
            ensure_dir(x_test_out)
            X_test.to_csv(x_test_out, index=False)
            ensure_dir(y_test_out)
            y_test.to_csv(y_test_out, index=False)
            logging.info(f"Dados de teste salvos em: {x_test_out} e {y_test_out}")

def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento do Modelo de Attrition")
    parser.add_argument("--in-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--threshold-path", type=str)
    parser.add_argument("--target-col", type=str, default="Attrition")
    parser.add_argument("--x-test-out", type=str)
    parser.add_argument("--y-test-out", type=str)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--retrain-full-data", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))