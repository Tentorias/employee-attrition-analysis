# src/attrition/models/predict.py (AJUSTADO PARA LER ARQUIVO JSON)

import argparse
import json

import joblib
import pandas as pd


def main(
    model_path: str, threshold_path: str, features_path: str, input_file_path: str
):
    """
    Carrega o modelo e faz uma predição para um novo funcionário fornecido via arquivo JSON.
    """
    try:
        # Carregar os artefatos (modelo, threshold, lista de features)
        model = joblib.load(model_path)
        threshold = joblib.load(threshold_path)
        feature_names = joblib.load(features_path)

        # --- MUDANÇA PRINCIPAL AQUI ---
        # Abre e lê o arquivo JSON do caminho fornecido
        print(f"Lendo dados do arquivo: {input_file_path}")
        with open(input_file_path, "r") as f:
            input_data = json.load(f)

        # O resto do código continua igual
        X_new = pd.DataFrame([input_data])
        X_new_aligned = X_new.reindex(columns=feature_names, fill_value=0)

        probability = model.predict_proba(X_new_aligned)[:, 1][0]
        prediction = int((probability >= threshold).astype(int))

        return prediction, probability

    except FileNotFoundError:
        print(
            "ERRO: Arquivo não encontrado em um dos caminhos. Verifique se os caminhos estão corretos."
        )
        return None, None
    except Exception as e:
        print(f"Ocorreu um erro na predição: {e}")
        return None, None


def cli_main():
    """Função para executar o script via linha de comando."""
    parser = argparse.ArgumentParser(
        description="Faz a predição para um novo funcionário a partir de um arquivo JSON."
    )
    parser.add_argument(
        "--model-path", required=True, help="Caminho para o modelo .pkl"
    )
    parser.add_argument(
        "--threshold-path", required=True, help="Caminho para o threshold .pkl"
    )
    parser.add_argument(
        "--features-path", required=True, help="Caminho para a lista de features .pkl"
    )
    # Altera o argumento para esperar um caminho de arquivo
    parser.add_argument(
        "--input-file",
        required=True,
        help="Caminho para o arquivo JSON com os dados do funcionário.",
    )
    args = parser.parse_args()

    prediction, probability = main(
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        features_path=args.features_path,
        input_file_path=args.input_file,  # Usa o novo argumento
    )

    if prediction is not None:
        print("\n--- Resultado da Predição ---")
        print(f"Probabilidade de Saída (Attrition): {probability:.4f}")
        print(
            f"Decisão Final: {'Funcionário Sai' if prediction == 1 else 'Funcionário Fica'}"
        )
        print("-----------------------------")


if __name__ == "__main__":
    cli_main()
