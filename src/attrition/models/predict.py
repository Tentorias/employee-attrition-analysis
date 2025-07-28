# src/attrition/models/predict.py

import argparse
import json
import os  # Importar os

import joblib
import numpy as np
import pandas as pd


def main(model_path: str, threshold_path: str, features_path: str, input_data: dict):
    """
    Recebe um dicionário com dados brutos, faz todo o pré-processamento,
    carrega o modelo e retorna a predição.
    """
    try:
        model = joblib.load(model_path)
        threshold = joblib.load(threshold_path)
        feature_names = joblib.load(features_path)

        X_new = pd.DataFrame([input_data])

        # As transformações aqui precisam ser as mesmas de data_processing.py e train.py!
        cols_to_drop = ["EmployeeCount", "StandardHours", "Over18"]
        X_new.drop(
            columns=[col for col in cols_to_drop if col in X_new.columns],
            errors="ignore",
            inplace=True,
        )

        if (
            "TotalWorkingYears" in X_new.columns
            and "NumCompaniesWorked" in X_new.columns
        ):
            X_new["YearsPerCompany"] = X_new["TotalWorkingYears"] / X_new[
                "NumCompaniesWorked"
            ].replace(
                0, 1
            )  # Usar .replace(0,1)
        if "MonthlyIncome" in X_new.columns:
            X_new["MonthlyIncome_log"] = np.log1p(X_new["MonthlyIncome"])
        if "TotalWorkingYears" in X_new.columns:
            X_new["TotalWorkingYears_log"] = np.log1p(X_new["TotalWorkingYears"])

        X_new_encoded = pd.get_dummies(X_new, drop_first=True, dtype=float)

        X_new_aligned = X_new_encoded.reindex(columns=feature_names, fill_value=0.0)

        probability = model.predict_proba(X_new_aligned)[:, 1][0]
        prediction = int((probability >= threshold).astype(int))

        return prediction, probability

    except Exception as e:
        # F541: Corrigir a f-string para ter a variável `e` dentro das chaves {}
        print(f"Ocorreu um erro na predição: {e}")
        return None, None


def cli_main():
    """Função para executar o script via linha de comando."""
    parser = argparse.ArgumentParser(
        description="Faz a predição a partir de um arquivo JSON."
    )
    # Default paths for cli_main
    parser.add_argument(
        "--model-path",
        default=os.path.join("models", "production_model.pkl"),
        help="Caminho para o modelo treinado.",
    )
    parser.add_argument(
        "--threshold-path",
        default=os.path.join("artifacts", "models", "optimal_threshold.pkl"),
        help="Caminho para o threshold salvo.",
    )
    parser.add_argument(
        "--features-path",
        default=os.path.join("artifacts", "features", "features.pkl"),
        help="Caminho para o arquivo de features.",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Caminho para o arquivo JSON com os dados do funcionário.",
    )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        input_data = json.load(f)

    prediction, probability = main(
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        features_path=args.features_path,
        input_data=input_data,
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
