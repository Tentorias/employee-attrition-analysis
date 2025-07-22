# src/attrition/models/evaluate.py (CORRIGIDO)
import argparse
import logging
import sys
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, threshold: float):
    logger.info(f"Aplicando threshold de {threshold:.2f} para fazer as previsões.")
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    logger.info("--- Relatório de Avaliação no Conjunto de Teste ---")
    logger.info("\n" + report)
    logger.info("--- Matriz de Confusão ---")
    logger.info("\n" + str(cm))

    return report, cm

def main(model_path: str, x_test_path: str, y_test_path: str): # REMOVIDO o threshold_path
    try:
        logger.info("Iniciando a avaliação do modelo...")
        model = joblib.load(model_path)

        # USA um threshold fixo de 0.5
        threshold = 0.5

        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze("columns")

        evaluate_model(model=model, X_test=X_test, y_test=y_test, threshold=threshold)

        logger.info("Avaliação concluída com sucesso.")

    except FileNotFoundError as e:
        logger.error(f"Erro: Arquivo não encontrado. Detalhes: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado durante a avaliação: {e}")
        sys.exit(1) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    # REMOVIDO o argumento --threshold-path
    parser.add_argument("--x-test-path", required=True)
    parser.add_argument("--y-test-path", required=True)
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        x_test_path=args.x_test_path,
        y_test_path=args.y_test_path,
    )