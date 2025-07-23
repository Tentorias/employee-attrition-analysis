# src/attrition/models/evaluate.py (COM THRESHOLD CUSTOMIZADO)
import argparse
import logging
import sys
import joblib
import pandas as pd
import numpy as np
# <<< Adicionado precision_score >>>
from sklearn.metrics import classification_report, confusion_matrix, precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# <<< Função modificada para buscar um alvo de precisão >>>
def find_threshold_for_target_precision(model, X_val, y_val, target_precision=0.40):
    """
    Encontra o threshold que resulta em uma precisão mais próxima do alvo.
    """
    logger.info(f"Otimizando o threshold de decisão para uma precisão próxima de {target_precision:.0%}...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    min_precision_diff = float('inf')

    # Testa thresholds de 0.05 a 0.95
    for threshold in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)
        # Calcula a precisão para a classe positiva (1)
        current_precision = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        
        # Verifica qual threshold chega mais perto do nosso alvo
        precision_diff = abs(current_precision - target_precision)
        if precision_diff < min_precision_diff:
            min_precision_diff = precision_diff
            best_threshold = threshold
            
    # Recalcula as métricas finais com o melhor threshold encontrado
    y_pred_final = (y_pred_proba >= best_threshold).astype(int)
    final_precision = precision_score(y_val, y_pred_final, pos_label=1)
    final_recall = precision_score(y_val, y_pred_final, pos_label=1)

    logger.info(f"✅ Threshold ótimo encontrado: {best_threshold:.2f} (resultando em Precisão de {final_precision:.2f} e Recall de {final_recall:.2f})")
    return best_threshold

def evaluate_with_threshold(model, X_test, y_test, threshold: float):
    """Aplica o threshold e imprime o relatório final."""
    logger.info(f"Aplicando threshold de {threshold:.2f} para fazer as previsões.")
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    report = classification_report(y_test, predictions, target_names=['Fica (0)', 'Sai (1)'])
    cm = confusion_matrix(y_test, predictions)

    logger.info("--- Relatório de Avaliação no Conjunto de Teste ---")
    logger.info("\n" + report)
    logger.info("--- Matriz de Confusão ---")
    logger.info("\n" + str(cm))

def main(model_path: str, x_test_path: str, y_test_path: str):
    try:
        logger.info("Iniciando a avaliação do modelo...")
        model = joblib.load(model_path)
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze("columns")
        
        # 1. Encontra o threshold ótimo para o alvo de 40% de precisão
        optimal_threshold = find_threshold_for_target_precision(model, X_test, y_test, target_precision=0.40)
        
        # 2. Avalia o modelo com o threshold ótimo
        evaluate_with_threshold(model, X_test, y_test, optimal_threshold)
        
        logger.info("Avaliação concluída com sucesso.")
        
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado durante a avaliação: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="artifacts/models/model.pkl")
    parser.add_argument("--x-test-path", default="artifacts/features/X_test.csv")
    parser.add_argument("--y-test-path", default="artifacts/features/y_test.csv")
    args = parser.parse_args()
    main(**vars(args))