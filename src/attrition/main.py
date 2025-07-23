# src/attrition/main.py (Final)
import argparse
import logging
from attrition.models import train, evaluate

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de ML para Employee Attrition")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_pipeline_parser = subparsers.add_parser("run-pipeline", help="Executa o pipeline completo.")
    run_pipeline_parser.add_argument("--tune", action="store_true", help="Ativa a otimização com Optuna.")
    
    args = parser.parse_args()

    if args.command == "run-pipeline":
        logging.info("--- 🚀 EXECUTANDO O PIPELINE COMPLETO 🚀 ---")
        
        
        raw_data_path = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
        model_path = "artifacts/models/model.pkl"
        features_path = "artifacts/features/features.pkl"
        params_path = "artifacts/models/best_params.json"
        x_test_path = "artifacts/features/X_test.csv"
        y_test_path = "artifacts/features/y_test.csv"
        prod_model_path = "models/production_model.pkl"

        
        logging.info("\n[ETAPA 1/3] Processando dados e treinando modelo de avaliação...")
        train.main(
            raw_data_path=raw_data_path,
            model_path=model_path,
            features_path=features_path,
            params_path=params_path,
            x_test_out=x_test_path,
            y_test_out=y_test_path,
            retrain_full_data=False,
            run_optuna_tuning=args.tune
        )

        
        logging.info("\n[ETAPA 2/3] Avaliando o modelo treinado...")
        evaluate.main(
            model_path=model_path,
            x_test_path=x_test_path,
            y_test_path=y_test_path
        )

        
        logging.info("\n[ETAPA 3/3] Retreinando o modelo com todos os dados para produção...")
        train.main(
            raw_data_path=raw_data_path,
            model_path=prod_model_path,
            features_path=features_path,
            params_path=params_path,
            x_test_out=None,
            y_test_out=None,
            retrain_full_data=True,
            run_optuna_tuning=False 
        )
        
        logging.info("\n--- ✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO! ---")

if __name__ == "__main__":
    main()