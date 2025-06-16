# src/attrition/models/evaluate.py (VERSÃO CORRETA)

import argparse
import logging

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, threshold: float):
    logger.info(f"Aplicando threshold de {threshold:.2f} para fazer as previsões.")
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    f1 = f1_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    logger.info("--- Relatório de Avaliação no Conjunto de Teste ---")
    logger.info("\n" + report)
    logger.info("--- Matriz de Confusão ---")
    logger.info("\n" + str(cm))
    logger.info("--------------------------------------------------")
    logger.info(f"✅ F1-Score Final (Teste): {f1:.4f}")
    return f1, report, cm


def main(model_path: str, threshold_path: str, x_test_path: str, y_test_path: str):
    try:
        logger.info("Iniciando a avaliação do modelo...")
        logger.info(f"Carregando modelo de: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Carregando threshold de: {threshold_path}")
        threshold = joblib.load(threshold_path)
        logger.info(f"Carregando dados de teste X de: {x_test_path}")
        X_test = pd.read_csv(x_test_path)
        logger.info(f"Carregando dados de teste y de: {y_test_path}")
        y_test = pd.read_csv(y_test_path).squeeze("columns")
        evaluate_model(model=model, X_test=X_test, y_test=y_test, threshold=threshold)
        logger.info("Avaliação concluída com sucesso.")
    except FileNotFoundError as e:
        logger.error(
            f"Erro: Arquivo não encontrado. Verifique os caminhos. Detalhes: {e}"
        )
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado durante a avaliação: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avalia o modelo treinado usando os dados de teste e o threshold otimizado."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Caminho para o arquivo do modelo treinado (.pkl).",
    )
    parser.add_argument(
        "--threshold-path",
        type=str,
        required=True,
        help="Caminho para o arquivo do threshold otimizado (.pkl).",
    )
    parser.add_argument(
        "--x-test-path",
        type=str,
        required=True,
        help="Caminho para o arquivo de dados de teste X_test.csv.",
    )
    parser.add_argument(
        "--y-test-path",
        type=str,
        required=True,
        help="Caminho para o arquivo de dados de teste y_test.csv.",
    )
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        x_test_path=args.x_test_path,
        y_test_path=args.y_test_path,
    )
