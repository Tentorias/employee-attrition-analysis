import argparse
import logging

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_model(path: str):
    """Carrega o modelo treinado salvo em disco."""
    return joblib.load(path)


def evaluate_model(model, X_test, y_test):
    """Avalia o modelo nos dados de teste e exibe métricas."""
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    f1 = f1_score(y_test, preds)
    return report, cm, f1


def main(model_path: str, test_data_path: str):
    """
    Script principal para avaliar o modelo.
    Recebe:
      - model_path: caminho para o modelo treinado
      - test_data_path: caminho para CSV de teste
    """
    logger.info(f"🔄 Carregando modelo de {model_path}")
    model = load_model(model_path)

    logger.info(f"🔄 Carregando dados de teste de {test_data_path}")
    df_test = pd.read_csv(test_data_path)
    y_test = (
        df_test["Attrition_Yes"] if "Attrition_Yes" in df_test else df_test["Attrition"]
    )
    X_test = df_test.drop(
        columns=[col for col in ["Attrition", "Attrition_Yes"] if col in df_test]
    )

    logger.info("🔧 Avaliando modelo")
    report, cm, f1 = evaluate_model(model, X_test, y_test)

    logger.info("\n" + report)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"F1-score: {f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avalia o modelo treinado em dados de teste."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Caminho para o modelo treinado",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        required=True,
        help="Caminho para os dados de teste",
    )
    args = parser.parse_args()
    main(model_path=args.model_path, test_data_path=args.test_data_path)
