import argparse
import logging
import os
import sys
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Configuração do logging para um feedback claro e profissional
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def explain_model(model, X_test: pd.DataFrame, output_path: str):
    """
    Usa o SHAP para gerar e salvar um gráfico de importância das features.
    Lida corretamente com modelos que estão dentro de um Pipeline.
    """
    logger.info("📊 Gerando explicações do modelo com SHAP...")

    # Se o modelo for um pipeline (como o nosso, que usa SMOTE),
    # precisamos extrair o classificador real de dentro dele.
    if hasattr(model, 'steps'):
        logger.info("Modelo do tipo Pipeline detetado. Extraindo o classificador...")
        actual_model = model.named_steps['classifier']
    else:
        actual_model = model

    # O TreeExplainer é otimizado para modelos baseados em árvore como LightGBM/XGBoost
    explainer = shap.TreeExplainer(actual_model)
    shap_values = explainer(X_test)

    logger.info(f"💾 Salvando gráfico de importância das features em {output_path}")
    # Garante que o diretório de saída exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Gera e salva o gráfico
    shap.summary_plot(shap_values, X_test, show=False, plot_size="auto")
    plt.title("Importância das Features (SHAP Summary Plot)", size=16)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close() # Fecha a figura para liberar memória
    logger.info("✅ Gráfico SHAP salvo com sucesso.")

def main(model_path: str, x_test_path: str, output_path: str = "reports/figures/shap_summary_plot.png"):
    """
    Função principal corrigida para aceitar argumentos diretamente do orquestrador.
    """
    try:
        logger.info("Iniciando a análise de explicabilidade do modelo...")
        model = joblib.load(model_path)
        X_test = pd.read_csv(x_test_path)
        
        explain_model(model=model, X_test=X_test, output_path=output_path)
        
        logger.info("Análise de explicabilidade concluída.")
    except FileNotFoundError as e:
        logger.error(f"❌ Erro: Arquivo não encontrado. Detalhes: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Ocorreu um erro inesperado: {e}")
        sys.exit(1)

def build_parser():
    """
    Cria o parser de argumentos para permitir a execução standalone do script.
    """
    parser = argparse.ArgumentParser(description="Gera explicações SHAP para o modelo treinado.")
    parser.add_argument("--model-path", type=str, required=True, help="Caminho para o arquivo do modelo treinado (.pkl).")
    parser.add_argument("--x-test-path", type=str, required=True, help="Caminho para os dados de teste X_test.csv.")
    parser.add_argument("--output-path", type=str, default="reports/figures/shap_summary_plot.png", help="Caminho para salvar o gráfico SHAP.")
    return parser

# Bloco para permitir que o script seja executado tanto pelo orquestrador quanto de forma independente
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(model_path=args.model_path, x_test_path=args.x_test_path, output_path=args.output_path)