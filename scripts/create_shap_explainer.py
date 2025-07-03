# scripts/create_shap_explainer.py

import argparse
import logging
import os
import joblib
import pandas as pd
import shap

# Configuração do logging para um feedback claro
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_and_save_explainer(model_path: str, output_path: str):
    """
    Carrega um modelo treinado, cria um explicador SHAP e o salva em um arquivo .pkl.
    """
    try:
        logger.info(f"Carregando modelo de: {model_path}")
        model = joblib.load(model_path)

        logger.info("Criando o objeto explicador SHAP...")
        
        # O modelo real dentro do pipeline do imblearn é o segundo passo, 'classifier'
        if hasattr(model, 'steps'):
            actual_model = model.named_steps['classifier']
        else:
            actual_model = model

        # CORREÇÃO FINAL: Inicializar o TreeExplainer apenas com o modelo.
        # Esta é a forma mais robusta e evita o erro de 'AttributeError'.
        explainer = shap.TreeExplainer(actual_model)

        # Garante que o diretório de saída exista antes de salvar
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Diretório criado: {output_dir}")
        
        logger.info(f"Salvando o explicador SHAP em: {output_path}")
        joblib.dump(explainer, output_path)
        
        logger.info(f"✅ Explicador SHAP salvo com sucesso em: {output_path}")

    except FileNotFoundError as e:
        logger.error(f"❌ Erro: Arquivo não encontrado. Detalhes: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Ocorreu um erro inesperado: {e}")
        raise

def main():
    """
    Função principal para analisar os argumentos da linha de comando e executar o script.
    """
    parser = argparse.ArgumentParser(
        description="Cria e salva um objeto explicador SHAP para um modelo treinado."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    create_and_save_explainer(
        model_path=args.model_path,
        output_path=args.output_path,
    )

if __name__ == "__main__":
    main()
