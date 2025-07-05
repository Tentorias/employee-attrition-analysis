import argparse
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(message)s")

from attrition.data import process
from attrition.features import engineer
from attrition.models import evaluate, explain, train, tunning, predict

def main():
    """Pipeline de ML para Employee Attrition via CLI."""
    parser = argparse.ArgumentParser(description="Pipeline de ML para Employee Attrition")
    subparsers = parser.add_subparsers(dest="command", help="Subcomandos disponíveis", required=True)

    # Subcomando 'process'
    process_parser = subparsers.add_parser("process", help="Processar dados brutos")
    process_parser.add_argument("--raw-path", type=str, required=True)
    process_parser.add_argument("--out-path", type=str, required=True)

    # Subcomando 'engineer'
    engineer_parser = subparsers.add_parser("engineer", help="Realizar engenharia de features")
    engineer_parser.add_argument("--input-path", type=str, required=True)
    engineer_parser.add_argument("--output-path", type=str, required=True)
    engineer_parser.add_argument("--features-out-path", type=str, required=True)

    # Subcomando 'train'
    train_parser = subparsers.add_parser("train", help="Treinar o modelo")
    train_parser.add_argument("--in-path", type=str, required=True)
    train_parser.add_argument("--features-path", type=str, required=True)
    train_parser.add_argument("--model-path", type=str, required=True)
    train_parser.add_argument("--threshold-path", type=str)
    train_parser.add_argument("--target-col", type=str, default="Attrition")
    train_parser.add_argument("--x-test-out", type=str)
    train_parser.add_argument("--y-test-out", type=str)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--retrain-full-data", action="store_true")

    # Subcomando 'tune'
    tune_parser = subparsers.add_parser("tune", help="Otimizar hiperparâmetros")
    tune_parser.add_argument("--data-path", type=str, required=True)
    tune_parser.add_argument("--features-path", type=str, required=True)
    tune_parser.add_argument("--output-path", type=str, required=True)
    tune_parser.add_argument("--target-col", type=str, default="Attrition")
    tune_parser.add_argument("--n-trials", type=int, default=100)

    # Subcomando 'evaluate'
    evaluate_parser = subparsers.add_parser("evaluate", help="Avaliar o modelo")
    evaluate_parser.add_argument("--model-path", type=str, required=True)
    evaluate_parser.add_argument("--threshold-path", type=str, required=True)
    evaluate_parser.add_argument("--x-test-path", type=str, required=True)
    evaluate_parser.add_argument("--y-test-path", type=str, required=True)

    # Subcomando 'explain'
    explain_parser = subparsers.add_parser("explain", help="Gerar explicações do modelo")
    explain_parser.add_argument("--model-path", type=str, required=True)
    explain_parser.add_argument("--x-test-path", type=str, required=True)
    explain_parser.add_argument("--output-path", type=str, default="reports/figures/shap_summary_plot.png")

    # Subcomando 'predict'
    predict_parser = subparsers.add_parser("predict", help="Fazer uma predição única")
    # ... (adicione os argumentos do predict se necessário)

    # Subcomando 'run-pipeline'
    run_pipeline_parser = subparsers.add_parser("run-pipeline", help="Executa o pipeline de ponta a ponta.")
    run_pipeline_parser.add_argument("--raw-data-path", type=str, default="data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    run_pipeline_parser.add_argument("--final-model-path", type=str, default="models/production_model.pkl")

    args = parser.parse_args()

    # Lógica para executar o comando selecionado
    command_functions = {
        "process": process.main,
        "engineer": engineer.main,
        "train": train.main,
        "tune": tunning.run_tuning,
        "evaluate": evaluate.main,
        "explain": explain.main,
        "predict": predict.main,
    }

    if args.command == "run-pipeline":
        logging.info("--- 🚀 EXECUTANDO O PIPELINE COMPLETO 🚀 ---")
        processed_data_path = "data/processed/employee_attrition_processed.csv"
        features_matrix_path = "artifacts/features/features_matrix.csv"
        features_list_path = "artifacts/features/features.pkl"
        model_path = "artifacts/models/model.pkl"
        threshold_path = "artifacts/models/threshold_optimizado.pkl"
        x_test_path = "artifacts/features/X_test.csv"
        y_test_path = "artifacts/features/y_test.csv"

        logging.info("\n[ETAPA 1/4] Processando dados...")
        process.main(raw_path=args.raw_data_path, out_path=processed_data_path)

        logging.info("\n[ETAPA 2/4] Criando features...")
        engineer.main(
            input_path=processed_data_path,
            output_path=features_matrix_path,
            features_out_path=features_list_path,
        )

        logging.info("\n[ETAPA 3/4] Treinando modelo...")
        train.main(
            in_path=features_matrix_path,
            features_path=features_list_path,
            model_path=model_path,
            threshold_path=threshold_path,
            target_col="Attrition",
            x_test_out=x_test_path,
            y_test_out=y_test_path,
        )

        logging.info("\n[ETAPA 4/4] Avaliando modelo final...")
        evaluate.main(
            model_path=model_path,
            threshold_path=threshold_path,
            x_test_path=x_test_path,
            y_test_path=y_test_path,
        )

        logging.info("\n[ETAPA 5/5] Retreinando o modelo com todos os dados para produção...")
        train.main(
            in_path=features_matrix_path,
            features_path=features_list_path,
            model_path=args.final_model_path,
            retrain_full_data=True
        )
        logging.info("\n--- ✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO! ---")
    
    else:
        # Executa o comando individual
        main_args = vars(args)
        command_to_run = main_args.pop("command")
        if command_to_run in command_functions:
            # Chama a função correspondente ao comando com os argumentos restantes
            command_functions[command_to_run](**main_args)
        else:
            logging.error(f"Comando '{command_to_run}' não reconhecido.")

if __name__ == "__main__":
    main()