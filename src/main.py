import argparse
import os
from src.data import process
from src.features import engineer
from src.models import train, evaluate, explain

def main():
    parser = argparse.ArgumentParser(description="Pipeline de ML para Employee Attrition")
    subparsers = parser.add_subparsers(dest="command", help="Subcomandos disponíveis")

    process_parser = subparsers.add_parser("process", help="Processar dados brutos")
    process_parser.add_argument("--raw-path", type=str, required=True, help="Caminho para o CSV bruto")
    process_parser.add_argument("--out-path", type=str, required=True, help="Caminho para salvar o CSV processado")

    engineer_parser = subparsers.add_parser("engineer", help="Realizar engenharia de features")
    engineer_parser.add_argument("--input-path", type=str, required=True, help="Caminho para o CSV de entrada")
    engineer_parser.add_argument("--output-path", type=str, required=True, help="Caminho para salvar o CSV com features")

    train_parser = subparsers.add_parser("train", help="Treinar o modelo")
    train_parser.add_argument("--data-path", type=str, required=True, help="Caminho para os dados de treinamento")
    train_parser.add_argument("--model-path", type=str, required=True, help="Caminho para salvar o modelo treinado")

    evaluate_parser = subparsers.add_parser("evaluate", help="Avaliar o modelo")
    evaluate_parser.add_argument("--model-path", type=str, required=True, help="Caminho para o modelo treinado")
    evaluate_parser.add_argument("--test-data-path", type=str, required=True, help="Caminho para os dados de teste")

    explain_parser = subparsers.add_parser("explain", help="Gerar explicações do modelo")
    explain_parser.add_argument("--model-path", type=str, required=True, help="Caminho para o modelo treinado")
    explain_parser.add_argument("--data-path", type=str, required=True, help="Caminho para os dados de entrada")

    args = parser.parse_args()

    if args.command == "process":
        process.main(args.raw_path, args.out_path)
    elif args.command == "engineer":
        engineer.main(args.input_path, args.output_path)
    elif args.command == "train":
        train.main(args.data_path, args.model_path)
    elif args.command == "evaluate":
        evaluate.main(args.model_path, args.test_data_path)
    elif args.command == "explain":
        explain.main(args.model_path, args.data_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
