# src/attrition/main.py
import argparse
import logging
import os

from attrition.models import evaluate, train

logging.basicConfig(level=logging.INFO, format="%(message)s")


def ensure_dir(file_path):
    """Garante que o diretório para um arquivo exista."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de ML para Employee Attrition"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Comando run-pipeline ---
    p = subparsers.add_parser("run-pipeline", help="Executa o pipeline completo.")
    p.add_argument(
        "--raw-data-path",
        default="data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        help="Caminho dos dados brutos.",
    )
    p.add_argument(
        "--model-path",
        default="artifacts/models/model.pkl",
        help="Caminho para salvar modelo de avaliação.",
    )
    p.add_argument(
        "--features-path",
        default="artifacts/features/features.pkl",
        help="Caminho para salvar lista de features.",
    )
    p.add_argument(
        "--params-path",
        default="artifacts/models/best_params.json",
        help="Caminho para parâmetros do Optuna.",
    )
    p.add_argument(
        "--x-test-path",
        default="artifacts/features/X_test.csv",
        help="Caminho para salvar X_test.",
    )
    p.add_argument(
        "--y-test-path",
        default="artifacts/features/y_test.csv",
        help="Caminho para salvar y_test.",
    )
    p.add_argument(
        "--prod-model-path",
        default="models/production_model.pkl",
        help="Caminho para salvar modelo de produção.",
    )
    p.add_argument("--tune", action="store_true", help="Ativa a otimização com Optuna.")
    p.add_argument(
        "--threshold-output-path",
        default="artifacts/models/optimal_threshold.pkl",
        help="Caminho para salvar o threshold ótimo.",
    )
    p.add_argument(
        "--min-precision-target",
        type=float,
        default=0.60,
        help="Precisão mínima desejada para otimizar o threshold.",
    )

    args = parser.parse_args()

    if args.command == "run-pipeline":
        logging.info("--- 🚀 EXECUTANDO O PIPELINE COMPLETO 🚀 ---")

        ensure_dir(args.model_path)
        ensure_dir(args.features_path)
        ensure_dir(args.params_path)
        ensure_dir(args.x_test_path)
        ensure_dir(args.y_test_path)
        ensure_dir(args.prod_model_path)
        os.makedirs(os.path.dirname(args.threshold_output_path), exist_ok=True)

        logging.info(
            "\n[ETAPA 1/3] Processando dados e treinando modelo de avaliação..."
        )
        train.main(
            raw_data_path=args.raw_data_path,
            model_path=args.model_path,
            features_path=args.features_path,
            params_path=args.params_path,
            x_test_out=args.x_test_path,
            y_test_out=args.y_test_path,
            retrain_full_data=False,
            run_optuna_tuning=args.tune,
        )

        logging.info("\n[ETAPA 2/3] Avaliando o modelo treinado...")
        evaluate.main(
            model_path=args.model_path,
            x_test_path=args.x_test_path,
            y_test_path=args.y_test_path,
            threshold_output_path=args.threshold_output_path,
            min_precision_target=args.min_precision_target,
        )

        logging.info(
            "\n[ETAPA 3/3] Retreinando o modelo com todos os dados para produção..."
        )
        train.main(
            raw_data_path=args.raw_data_path,
            model_path=args.prod_model_path,
            features_path=args.features_path,
            params_path=args.params_path,
            x_test_out=None,
            y_test_out=None,
            retrain_full_data=True,
            run_optuna_tuning=False,
        )

        logging.info("\n--- ✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO! ---")


if __name__ == "__main__":
    main()
