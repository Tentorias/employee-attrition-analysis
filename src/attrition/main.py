import argparse
import logging

# Importa apenas os módulos que ainda existem e são necessários
from attrition.models import train, evaluate

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    """
    Orquestra o pipeline de ML refatorado através de uma interface de linha de comando.
    """
    # Configura o parser principal
    parser = argparse.ArgumentParser(description="Pipeline de ML para Employee Attrition")
    
    # Define o único comando que estamos usando: 'run-pipeline'
    # Esta é a forma correta de adicionar um subcomando
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_pipeline_parser = subparsers.add_parser("run-pipeline", help="Executa o pipeline completo.")

    # Argumentos específicos para o 'run-pipeline' (opcionais, com valores padrão)
    run_pipeline_parser.add_argument("--raw-data-path", default="data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    run_pipeline_parser.add_argument("--prod-model-path", default="models/production_model.pkl")
    
    args = parser.parse_args()

    # Executa o pipeline se o comando for 'run-pipeline'
    if args.command == "run-pipeline":
        logging.info("--- 🚀 EXECUTANDO O PIPELINE COMPLETO (À PROVA DE DATA LEAKAGE) 🚀 ---")
        
        # Define os caminhos intermediários para os artefatos
        model_path = "artifacts/models/model.pkl"
        features_path = "artifacts/features/features.pkl"
        x_test_path = "artifacts/features/X_test.csv"
        y_test_path = "artifacts/features/y_test.csv"

        # ETAPA 1: Treinar o modelo de avaliação (o novo train.py faz o pré-processamento)
        logging.info("\n[ETAPA 1/3] Processando dados e treinando modelo de avaliação...")
        train.main(
            raw_data_path=args.raw_data_path,
            model_path=model_path,
            features_path=features_path,
            x_test_out=x_test_path,
            y_test_out=y_test_path,
            retrain_full_data=False # Garante que o split de teste seja criado
        )

        # ETAPA 2: Avaliar o modelo
        logging.info("\n[ETAPA 2/3] Avaliando o modelo treinado...")
        evaluate.main(
            model_path=model_path,
            x_test_path=x_test_path,
            y_test_path=y_test_path
        )

        # ETAPA 3: Retreinar com todos os dados para produção
        logging.info("\n[ETAPA 3/3] Retreinando o modelo com todos os dados para produção...")
        train.main(
            raw_data_path=args.raw_data_path,
            model_path=args.prod_model_path,
            features_path=features_path,
            x_test_out=None, # Define como None para não salvar
            y_test_out=None, # Define como None para não salvar
            retrain_full_data=True # Agora retreina com todos os dados
        )
        
        logging.info("\n--- ✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO! ---")

if __name__ == "__main__":
    main()