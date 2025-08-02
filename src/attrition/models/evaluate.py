# src/attrition/models/evaluate.py
import argparse
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_optimal_threshold_f1_constrained_precision(
    model, X_val, y_val, min_precision_target=0.60
):
    """
    Encontra o threshold que maximiza o Recall, garantindo uma precisão mínima.
    """
    logger.info(
        f"Otimizando o threshold de decisão para maximizar o F1-score com Precisão mínima de {min_precision_target:.0%}..."
    )
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    max_f1 = -1.0

    results = []
    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)

        current_precision = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        current_recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        current_f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)

        results.append(
            {
                "threshold": threshold,
                "precision": current_precision,
                "recall": current_recall,
                "f1_score": current_f1,
            }
        )

        if current_precision >= min_precision_target and current_f1 > max_f1:
            max_f1 = current_f1
            best_threshold = threshold

    if max_f1 == -1.0:
        logger.warning(
            f"⚠️ Nenhum threshold atingiu a precisão mínima de {min_precision_target:.0%}. Escolhendo o threshold com o maior F1-Score geral."
        )
        if results:
            best_threshold = max(results, key=lambda x: x["f1_score"])["threshold"]
        else:
            best_threshold = 0.5

    y_pred_final = (y_pred_proba >= best_threshold).astype(int)
    final_precision = precision_score(y_val, y_pred_final, pos_label=1, zero_division=0)
    final_recall = recall_score(y_val, y_pred_final, pos_label=1, zero_division=0)
    final_f1 = f1_score(y_val, y_pred_final, pos_label=1, zero_division=0)

    logger.info(
        f"✅ Threshold ótimo encontrado: {best_threshold:.2f} (Precisão: {final_precision:.2f}, Recall: {final_recall:.2f}, F1-Score: {final_f1:.2f})"
    )

    threshold_output_path = "artifacts/models/optimal_threshold.pkl"
    os.makedirs(os.path.dirname(threshold_output_path), exist_ok=True)
    joblib.dump(best_threshold, threshold_output_path)
    logger.info(f"✅ Threshold ótimo salvo em: {threshold_output_path}")

    return best_threshold


def evaluate_with_threshold(model, X_test, y_test, threshold: float):
    """Aplica o threshold e imprime o relatório final."""
    logger.info(f"Aplicando threshold de {threshold:.2f} para fazer as previsões.")
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    report = classification_report(
        y_test, predictions, target_names=["Fica (0)", "Sai (1)"]
    )
    cm = confusion_matrix(y_test, predictions)

    logger.info("--- Relatório de Avaliação no Conjunto de Teste ---")
    logger.info("\n" + report)
    logger.info("--- Matriz de Confusão ---")
    logger.info("\n" + str(cm))


def main(
    model_path: str,
    x_test_path: str,
    y_test_path: str,
    threshold_output_path: str = "artifacts/models/optimal_threshold.pkl",
    min_precision_target: float = 0.60,
):
    try:
        logger.info("Iniciando a avaliação do modelo...")
        model = joblib.load(model_path)
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze("columns")

        optimal_threshold = find_optimal_threshold_f1_constrained_precision(
            model, X_test, y_test, min_precision_target=min_precision_target
        )

        evaluate_with_threshold(model, X_test, y_test, optimal_threshold)

        logger.info("Avaliação concluída com sucesso.")

    except Exception as e:
        logger.error(
            f"Ocorreu um erro inesperado durante a avaliação: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="artifacts/models/model.pkl")
    parser.add_argument("--x-test-path", default="artifacts/features/X_test.csv")
    parser.add_argument("--y-test-path", default="artifacts/features/y_test.csv")
    parser.add_argument(
        "--threshold-output-path",
        default="artifacts/models/optimal_threshold.pkl",
        help="Caminho para salvar o threshold ótimo.",
    )
    parser.add_argument(
        "--min-precision-target",
        type=float,
        default=0.60,
        help="Precisão mínima desejada para otimizar o threshold.",
    )
    args = parser.parse_args()
    main(**vars(args))
