# src/attrition/main.py

import argparse
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(message)s")

from attrition.data import process
from attrition.features import engineer
from attrition.models import evaluate, explain, train, tunning, predict

def main():
    """Pipeline de ML para Employee Attrition via CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline de ML para Employee Attrition"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcomandos disponíveis", required=True)

    process_parser = subparsers.add_parser("process", help="Processar dados brutos")
    process_parser.add_argument("--raw-path", type=str, required=True)
    process_parser.add_argument("--out-path", type=str, required=True)

    engineer_parser = subparsers.add_parser("engineer", help="Realizar engenharia de features")
    engineer_parser.add_argument("--input-path", type=str, required=True)
    engineer_parser.add_argument("--output-path", type=str, required=True)
    engineer_parser.add_argument("--features-out-path", type=str, required=True)

    train_parser = subparsers.add_parser("train", help="Treinar o modelo")
    train_parser.add_argument("--in-path", type=str, required=True)
    train_parser.add_argument("--features-path", type=str, required=True)
    train_parser.add_argument("--model-path", type=str, required=True)
    train_parser.add_argument("--threshold-path", type=str, required=False)
    train_parser.add_argument("--target-col", type=str, default="Attrition")
    train_parser.add_argument("--x-test-out", type=str, required=False)
    train_parser.add_argument("--y-test-out", type=str, required=False)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--retrain-full-data", action="store_true")

    tune_parser = subparsers.add_parser("tune", help="Otimizar hiperparâmetros")
    tune_parser.add_argument("--data-path", type=str, required=True)
    tune_parser.add_argument("--features-path", type=str, required=True)
    tune_parser.add_argument("--target-col", type=str, default="Attrition")
    tune_parser.add_argument("--n-trials", type=int, default=50)

    evaluate_parser = subparsers.add_parser("evaluate", help="Avaliar o modelo")
    evaluate_parser.add_argument("--model-path", type=str, required=True)
    evaluate_parser.add_argument("--threshold-path", type=str, required=True)
    evaluate_parser.add_argument("--x-test-path", type=str, required=True)
    evaluate_parser.add_argument("--y-test-path", type=str, required=True)


    explain_parser = subparsers.add_parser("explain", help="Gerar explicações do modelo")
    explain_parser.add_argument("--model-path", type=str, required=True)
    explain_parser.add_argument("--x-test-path", type=str, required=True)
    explain_parser.add_argument("--output-path", type=str, required=False)

    predict_parser = subparsers.add_parser("predict", help="Fazer uma predição única")
    predict_parser.add_argument("--model-path", type=str, required=True)
    predict_parser.add_argument("--threshold-path", type=str, required=True)
    predict_parser.add_argument("--features-path", type=str, required=True)
    predict_parser.add_argument("--input-data", type=str, required=True, help='String JSON com os dados de entrada')

    run_pipeline_parser = subparsers.add_parser("run-pipeline", help="Executa o pipeline de ponta a ponta.")
    run_pipeline_parser.add_argument("--raw-data-path", type=str, default="data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    run_pipeline_parser.add_argument("--final-model-path", type=str, default="models/production_model.pkl")

    args = parser.parse_args()

    if args.command == "process":
        process.main(raw_path=args.raw_path, out_path=args.out_path)
    elif args.command == "engineer":
        engineer.main(
            input_path=args.input_path,
            output_path=args.output_path,
            features_out_path=args.features_out_path,
        )
    elif args.command == "train":
        train.main(
            in_path=args.in_path,
            features_path=args.features_path,
            model_path=args.model_path,
            threshold_path=args.threshold_path,
            target_col=args.target_col,
            x_test_out=args.x_test_out,
            y_test_out=args.y_test_out,
            test_size=args.test_size,
            random_state=args.random_state,
            retrain_full_data=args.retrain_full_data,
        )
    elif args.command == "tune":
        tunning.run_tuning(
            data_path=args.data_path,
            features_path=args.features_path,
            target_col=args.target_col,
            n_trials=args.n_trials,
        )
    elif args.command == "evaluate":
        evaluate.main(
            model_path=args.model_path,
            threshold_path=args.threshold_path,
            x_test_path=args.x_test_path,
            y_test_path=args.y_test_path,
        )
    elif args.command == "explain":
        explain_args = {"model_path": args.model_path, "x_test_path": args.x_test_path}
        if hasattr(args, 'output_path') and args.output_path:
            explain_args["output_path"] = args.output_path
        explain.main(**explain_args)
    elif args.command == "predict":
        try:
            input_data_dict = json.loads(args.input_data)
            predict.main(
                model_path=args.model_path,
                threshold_path=args.threshold_path,
                features_path=args.features_path,
                input_data=input_data_dict,
            )
        except json.JSONDecodeError:
            logging.error("Erro: A string de --input-data não é um JSON válido.")
    elif args.command == "run-pipeline":
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

if __name__ == "__main__":
    main()


