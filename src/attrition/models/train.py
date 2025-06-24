# src/attrition/models/train.py

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")

def train_model(X, y, random_state=42):
    """
    Treina um modelo de classificação com balanceamento de dados.
    Utiliza SMOTE para lidar com o desbalanceamento, somente se houver amostras suficientes.
    """
    k_neighbors_smote = 5
    
    n_minority_samples = y.value_counts().min()

    if n_minority_samples > k_neighbors_smote:
        logging.info("Amostras suficientes. Treinando o modelo com SMOTE...")
        model = Pipeline([
            ("smote", SMOTE(random_state=random_state, k_neighbors=k_neighbors_smote)),
            ("classifier", LGBMClassifier(random_state=random_state))
        ])
    else:
        logging.warning(
            f"Não foi possível usar SMOTE (amostras da classe minoritária: {n_minority_samples} <= k_neighbors: {k_neighbors_smote}). "
            "Treinando sem SMOTE."
        )
        model = LGBMClassifier(random_state=random_state)

    model.fit(X, y)
    return model


def optimize_threshold(model, X_val, y_val):
    """Otimiza o threshold de classificação com base no F1-score."""
    logging.info("Otimizando o threshold de classificação...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0, 1.01, 0.01)
    f1_scores = [f1_score(y_val, y_pred_proba >= t) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    logging.info(f"Threshold otimizado encontrado: {optimal_threshold:.2f} (F1-Score: {f1_scores[optimal_idx]:.4f})")
    return optimal_threshold


def main(in_path, features_path, model_path, threshold_path=None, target_col="Attrition", x_test_out=None, y_test_out=None, test_size=0.2, random_state=42, retrain_full_data=False):
    """Função principal para executar o passo de treinamento do modelo."""
    logging.info(f"Lendo a matriz de features de: {in_path}")
    df = pd.read_csv(in_path)
    
    logging.info(f"Lendo a lista de features de: {features_path}")
    features = joblib.load(features_path)
    
    X = df[features]
    y = df[target_col]

    if retrain_full_data:
        logging.info("Retreinando o modelo com todos os dados disponíveis...")
        model = train_model(X, y, random_state)
        joblib.dump(model, model_path)
        logging.info(f"Modelo final salvo em: {model_path}")
        if threshold_path:
            joblib.dump(0.5, threshold_path) 
    else:
        logging.info("Dividindo os dados em conjuntos de treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        model = train_model(X_train, y_train, random_state)

        joblib.dump(model, model_path)
        logging.info(f"Modelo treinado salvo em: {model_path}")

        if threshold_path:
            optimal_thr = optimize_threshold(model, X_test, y_test)
            joblib.dump(optimal_thr, threshold_path)
            logging.info(f"Threshold otimizado salvo em: {threshold_path}")

        if x_test_out and y_test_out:
            X_test.to_csv(x_test_out, index=False)
            y_test.to_csv(y_test_out, index=False)
            logging.info(f"Dados de teste salvos em: {x_test_out} e {y_test_out}")


def parse_args():
    """Analisa os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Treinamento do Modelo de Attrition")
    parser.add_argument("--in-path", type=str, required=True, help="Caminho para a matriz de features (CSV).")
    parser.add_argument("--features-path", type=str, required=True, help="Caminho para a lista de features (PKL).")
    parser.add_argument("--model-path", type=str, required=True, help="Caminho para salvar o modelo treinado (PKL).")
    parser.add_argument("--threshold-path", type=str, help="Caminho para salvar o threshold otimizado (PKL).")
    parser.add_argument("--target-col", type=str, default="Attrition", help="Nome da coluna alvo.")
    parser.add_argument("--x-test-out", type=str, help="Caminho para salvar o X_test (CSV).")
    parser.add_argument("--y-test-out", type=str, help="Caminho para salvar o y_test (CSV).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporção do conjunto de teste.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed para reprodutibilidade.")
    parser.add_argument("--retrain-full-data", action="store_true", help="Flag para retreinar o modelo com todos os dados.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        in_path=args.in_path,
        features_path=args.features_path,
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        target_col=args.target_col,
        x_test_out=args.x_test_out,
        y_test_out=args.y_test_out,
        test_size=args.test_size,
        random_state=args.random_state,
        retrain_full_data=args.retrain_full_data,
    )