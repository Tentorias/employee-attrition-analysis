# src/attrition/main.py (CORRIGIDO E COMPLETO)

import argparse

from attrition.data import process
from attrition.features import engineer
from attrition.models import evaluate, explain, predict, train, tunning

# import os <-- Removido, pois não era utilizado


def main():
    """Pipeline de ML para Employee Attrition via CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline de ML para Employee Attrition"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcomandos disponíveis")

    # --- Todos os parsers de argumentos ---
    process_parser = subparsers.add_parser("process", help="Processar dados brutos")
    process_parser.add_argument(
        "--raw-path", type=str, required=True, help="Caminho para o CSV bruto"
    )
    process_parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Caminho para salvar o CSV processado",
    )

    engineer_parser = subparsers.add_parser(
        "engineer", help="Realizar engenharia de features"
    )
    engineer_parser.add_argument(
        "--input-path", type=str, required=True, help="Caminho para o CSV de entrada"
    )
    engineer_parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Caminho para salvar o CSV com features",
    )

    train_parser = subparsers.add_parser("train", help="Treinar o modelo")
    train_parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Caminho para os dados de treinamento",
    )
    train_parser.add_argument(
        "--features-path",
        type=str,
        required=True,
        help="Pickle com lista de features X",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Caminho para salvar o modelo treinado",
    )
    train_parser.add_argument(
        "--thr-path",
        type=str,
        required=False,
        help="Caminho para salvar o threshold otimizado",
    )
    train_parser.add_argument(
        "--target-col",
        type=str,
        required=False,
        default="Attrition_Yes",
        help="Nome da coluna alvo",
    )
    train_parser.add_argument(
        "--x-test-out",
        type=str,
        required=False,
        help="Caminho para salvar X_test (CSV)",
    )
    train_parser.add_argument(
        "--y-test-out",
        type=str,
        required=False,
        help="Caminho para salvar y_test (CSV)",
    )
    train_parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proporção de teste para split"
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed para reprodutibilidade",
    )
    train_parser.add_argument(
        "--retrain-full-data",
        action="store_true",
        help="Pula o split e treina com todos os dados.",
    )

    tune_parser = subparsers.add_parser(
        "tune", help="Executar otimização de hiperparâmetros"
    )
    tune_parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Caminho para o features_matrix.csv",
    )
    tune_parser.add_argument(
        "--features-path", type=str, required=True, help="Caminho para o features.pkl"
    )
    tune_parser.add_argument(
        "--target-col", type=str, default="Attrition_Yes", help="Nome da coluna alvo"
    )
    tune_parser.add_argument(
        "--n-trials", type=int, default=50, help="Número de tentativas para o Optuna"
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Avaliar o modelo")
    evaluate_parser.add_argument(
        "--model-path", type=str, required=True, help="Caminho para o modelo .pkl"
    )
    evaluate_parser.add_argument(
        "--threshold-path",
        type=str,
        required=True,
        help="Caminho para o threshold .pkl",
    )
    evaluate_parser.add_argument(
        "--x-test-path", type=str, required=True, help="Caminho para X_test.csv"
    )
    evaluate_parser.add_argument(
        "--y-test-path", type=str, required=True, help="Caminho para y_test.csv"
    )

    explain_parser = subparsers.add_parser(
        "explain", help="Gerar explicações do modelo"
    )
    explain_parser.add_argument(
        "--model-path", type=str, required=True, help="Caminho para o modelo treinado"
    )
    explain_parser.add_argument(
        "--x-test-path", type=str, required=True, help="Caminho para X_test.csv"
    )
    explain_parser.add_argument(
        "--y-test-path", type=str, required=True, help="Caminho para y_test.csv"
    )
    explain_parser.add_argument(
        "--threshold-path",
        type=str,
        required=True,
        help="Caminho para o threshold .pkl",
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Faz a predição para um novo funcionário"
    )
    predict_parser.add_argument(
        "--model-path", type=str, required=True, help="Caminho para o modelo .pkl"
    )
    predict_parser.add_argument(
        "--threshold-path",
        type=str,
        required=True,
        help="Caminho para o threshold .pkl",
    )
    predict_parser.add_argument(
        "--features-path",
        type=str,
        required=True,
        help="Caminho para a lista de features .pkl",
    )
    predict_parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Caminho para o arquivo JSON com dados de entrada",
    )

    _ = subparsers.add_parser(
        "run-pipeline",
        help="Executa todo o pipeline de processamento, treino e avaliação.",
    )

    args = parser.parse_args()

    # --- Lógica de Execução dos Comandos ---
    if args.command == "process":
        process.main(raw_path=args.raw_path, out_path=args.out_path)
    elif args.command == "engineer":
        engineer.main(input_path=args.input_path, output_path=args.output_path)
    elif args.command == "train":
        train.main(
            in_path=args.data_path,
            features_path=args.features_path,
            model_path=args.model_path,
            threshold_path=args.thr_path,
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
        explain.main(
            model_path=args.model_path,
            x_test_path=args.x_test_path,
            y_test_path=args.y_test_path,
            threshold_path=args.threshold_path,
        )
    elif args.command == "predict":
        predict.main(
            model_path=args.model_path,
            threshold_path=args.threshold_path,
            features_path=args.features_path,
            input_file_path=args.input_file,
        )
    elif args.command == "run-pipeline":
        print("--- EXECUTANDO O PIPELINE COMPLETO ---")
        raw_data_path = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
        processed_data_path = "data/processed/employee_attrition_processed.csv"
        features_matrix_path = "artifacts/features/features_matrix.csv"
        features_list_path = "artifacts/features/features.pkl"
        model_path = "artifacts/models/model.pkl"
        threshold_path = "artifacts/models/threshold_optimizado.pkl"
        x_test_path = "artifacts/features/X_test.csv"
        y_test_path = "artifacts/features/y_test.csv"

        print("\n[ETAPA 1/4] Processando dados...")
        process.main(raw_path=raw_data_path, out_path=processed_data_path)

        print("\n[ETAPA 2/4] Criando features...")
        engineer.main(
            input_path=processed_data_path,
            output_path=features_matrix_path,
            features_out_path=features_list_path,
        )

        print("\n[ETAPA 3/4] Treinando modelo...")
        train.main(
            in_path=features_matrix_path,
            features_path=features_list_path,
            model_path=model_path,
            threshold_path=threshold_path,
            target_col="Attrition_Yes",
            x_test_out=x_test_path,
            y_test_out=y_test_path,
        )

        print("\n[ETAPA 4/4] Avaliando modelo final...")
        evaluate.main(
            model_path=model_path,
            threshold_path=threshold_path,
            x_test_path=x_test_path,
            y_test_path=y_test_path,
        )
        print("\n--- PIPELINE COMPLETO EXECUTADO COM SUCESSO! ---")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
