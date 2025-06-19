# src/attrition/models/predict.py (VERSÃO FINAL E INTELIGENTE)

import argparse
import json

import joblib
import numpy as np
import pandas as pd


def main(model_path: str, threshold_path: str, features_path: str, input_data: dict):
    """
    Recebe um dicionário com dados brutos, faz todo o pré-processamento,
    carrega o modelo e retorna a predição.
    """
    try:
        # Carregar os artefatos
        model = joblib.load(model_path)
        threshold = joblib.load(threshold_path)
        feature_names = joblib.load(features_path)

        # --- INTELIGÊNCIA DE PRÉ-PROCESSAMENTO CENTRALIZADA AQUI ---
        X_new = pd.DataFrame([input_data])

        # 1. Cria features derivadas que o modelo espera
        if (
            "TotalWorkingYears" in X_new.columns
            and "NumCompaniesWorked" in X_new.columns
        ):
            denominator = (
                X_new["NumCompaniesWorked"].iloc[0]
                if X_new["NumCompaniesWorked"].iloc[0] > 0
                else 1
            )
            X_new["YearsPerCompany"] = X_new["TotalWorkingYears"].iloc[0] / denominator
        if "MonthlyIncome" in X_new.columns:
            X_new["MonthlyIncome_log"] = np.log1p(X_new["MonthlyIncome"])
        if "TotalWorkingYears" in X_new.columns:
            X_new["TotalWorkingYears_log"] = np.log1p(X_new["TotalWorkingYears"])

        # 2. Aplica o one-hot encoding
        X_new_encoded = pd.get_dummies(X_new, drop_first=True, dtype=float)

        # 3. Alinha as colunas com o que o modelo foi treinado
        X_new_aligned = X_new_encoded.reindex(columns=feature_names, fill_value=0)

        # 4. Faz a predição
        probability = model.predict_proba(X_new_aligned)[:, 1][0]
        prediction = int((probability >= threshold).astype(int))

        return prediction, probability

    except Exception as e:
        print(f"Ocorreu um erro na predição: {e}")
        return None, None


def cli_main():
    """Função para executar o script via linha de comando."""
    parser = argparse.ArgumentParser(
        description="Faz a predição a partir de um arquivo JSON."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--threshold-path", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument(
        "--input-file",
        required=True,
        help="Caminho para o arquivo JSON com os dados do funcionário.",
    )
    args = parser.parse_args()

    # Lê os dados brutos do arquivo para simular a entrada
    with open(args.input_file, "r") as f:
        input_data = json.load(f)

    prediction, probability = main(
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        features_path=args.features_path,
        input_data=input_data,  # Passa o dicionário para a função principal
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
