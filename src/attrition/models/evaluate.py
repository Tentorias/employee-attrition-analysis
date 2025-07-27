# src/attrition/models/evaluate.py 
import argparse
import logging
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score # Importar f1_score e recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Mudar o nome da função para refletir o novo objetivo
def find_optimal_threshold_f1_constrained_precision(model, X_val, y_val, min_precision_target=0.60):
    """
    Encontra o threshold que maximiza o F1-score, garantindo uma precisão mínima.
    """
    logger.info(f"Otimizando o threshold de decisão para maximizar o F1-score com Precisão mínima de {min_precision_target:.0%}...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5 # Default inicial
    max_f1 = -1.0 # Inicializa com um valor baixo para F1-score (não pode ser -1 se f1_score for 0)
    
    results = [] # Para armazenar resultados e depurar

    # Varra thresholds de 0.01 até 0.99
    for threshold in np.arange(0.01, 1.00, 0.01): # Inclui 0.99 para cobrir mais o range
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calcular Precision, Recall e F1-score para o threshold atual
        current_precision = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        current_recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        current_f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': current_precision,
            'recall': current_recall,
            'f1_score': current_f1
        })

        # Filtre só os thresholds com precision >= min_precision_target
        # Pegue o que tem maior F1-score
        if current_precision >= min_precision_target and current_f1 > max_f1:
            max_f1 = current_f1
            best_threshold = threshold
            
    # Se nenhum threshold atender à precisão mínima, podemos retornar o threshold com o melhor F1-score geral
    # (ou um valor padrão, dependendo da estratégia)
    if max_f1 == -1.0: # Se nenhum threshold atendeu à condição de precisão mínima
        logger.warning(f"⚠️ Nenhum threshold atingiu a precisão mínima de {min_precision_target:.0%}. Escolhendo o threshold com o maior F1-Score geral.")
        if results:
            best_threshold = max(results, key=lambda x: x['f1_score'])['threshold']
        else:
            best_threshold = 0.5 # Fallback extremo se não houver resultados
        
    y_pred_final = (y_pred_proba >= best_threshold).astype(int)
    final_precision = precision_score(y_val, y_pred_final, pos_label=1, zero_division=0)
    final_recall = recall_score(y_val, y_pred_final, pos_label=1, zero_division=0)
    final_f1 = f1_score(y_val, y_pred_final, pos_label=1, zero_division=0)


    logger.info(f"✅ Threshold ótimo encontrado: {best_threshold:.2f} (Precisão: {final_precision:.2f}, Recall: {final_recall:.2f}, F1-Score: {final_f1:.2f})")
    
    # Adicionar o salvamento do threshold aqui
    # Este threshold precisa ser salvo em um arquivo para ser usado pela API e pelo Streamlit
    threshold_output_path = "artifacts/models/optimal_threshold.pkl" # Definir um caminho para salvar (já existia no main.py)
    # A função ensure_dir não está definida aqui, mas assumimos que o diretório é garantido pelo main.py
    # Se não for, precisaríamos importar ensure_dir ou criar o diretório aqui.
    joblib.dump(best_threshold, threshold_output_path)
    logger.info(f"✅ Threshold ótimo salvo em: {threshold_output_path}")

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

# A assinatura da função main agora aceita 'threshold_output_path'
def main(model_path: str, x_test_path: str, y_test_path: str, threshold_output_path: str = "artifacts/models/optimal_threshold.pkl"): 
    try:
        logger.info("Iniciando a avaliação do modelo...")
        model = joblib.load(model_path)
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze("columns")
        
        # Chamar a nova função de otimização de threshold
        # Min Precision Target: 60% (0.60) conforme seu plano
        optimal_threshold = find_optimal_threshold_f1_constrained_precision(model, X_test, y_test, min_precision_target=0.60) 
        
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
    parser.add_argument("--threshold-output-path", default="artifacts/models/optimal_threshold.pkl", help="Caminho para salvar o threshold ótimo.") # Novo argumento
    args = parser.parse_args()
    main(**vars(args))