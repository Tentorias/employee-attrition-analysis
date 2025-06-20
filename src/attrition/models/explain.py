# src/attrition/models/explain.py (versão com sugestões)

import argparse
import logging
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Configuração do logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def explain_model(model, X_test: pd.DataFrame, output_path: str):
    """
    Usa o SHAP para gerar e salvar um gráfico de importância das features.
    """
    logger.info("📊 Gerando explicações do modelo com SHAP...")

    # 1. Cria o explainer de forma agnóstica ao modelo
    # shap.Explainer seleciona o algoritmo ideal (Tree, Kernel, etc.)
    explainer = shap.Explainer(model, X_test)

    # 2. Calcula os valores SHAP para o conjunto de teste
    shap_values = explainer(X_test)

    # 3. Gera e salva o gráfico de resumo (beeswarm)
    logger.info(f"💾 Salvando gráfico de importância das features em {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure()  # shap.summary_plot cria sua própria figura, mas é bom ter o controle
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Importância das Features (SHAP Summary Plot)", size=16)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info("✅ Gráfico SHAP salvo com sucesso.")


def main():
    """
    Função principal para orquestrar a explicação do modelo.
    """
    parser = argparse.ArgumentParser(
        description="Gera explicações SHAP para o modelo treinado."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Caminho para o arquivo do modelo treinado (.pkl).",
    )
    parser.add_argument(
        "--x-test-path",
        type=str,
        required=True,
        help="Caminho para o arquivo de dados de teste X_test.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="reports/figures/shap_summary_plot.png",
        help="Caminho para salvar o gráfico SHAP gerado.",
    )
    args = parser.parse_args()

    try:
        logger.info("Iniciando a análise de explicabilidade do modelo...")

        logger.info(f"Carregando modelo de: {args.model_path}")
        model = joblib.load(args.model_path)

        logger.info(f"Carregando dados de teste X de: {args.x_test_path}")
        X_test = pd.read_csv(args.x_test_path)

        explain_model(model=model, X_test=X_test, output_path=args.output_path)

        logger.info("Análise de explicabilidade concluída.")

    except FileNotFoundError as e:
        logger.error(
            f"Erro: Arquivo não encontrado. Verifique os caminhos. Detalhes: {e}"
        )
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado: {e}")


if __name__ == "__main__":
    main()
