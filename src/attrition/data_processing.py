# src/attrition/data_processing.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from attrition.features import preprocess_employee_data


def load_and_preprocess_data(model_features_list=None):
    """
    Carrega os dados dos funcionários e aplica transformações IDÊNTICAS
    ao preprocess centralizado para compatibilidade com o modelo treinado.
    Retorna dois DataFrames: um para o modelo de ML e outro para exibição no UI.

    Args:
        model_features_list (list, optional): Lista de nomes das features que o modelo espera.

    Retorna:
        tuple: (pd.DataFrame para modelo, pd.DataFrame para UI)
    """
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")

    df = pd.DataFrame()
    if DATABASE_URL:
        try:
            engine = create_engine(DATABASE_URL)
            df = pd.read_sql("SELECT * FROM employees", engine)
            print("Dados carregados do PostgreSQL.")
        except Exception as e:
            print(
                f"Erro ao carregar dados do PostgreSQL: {e}. Tentando carregar do CSV..."
            )
            pass

    if df.empty:
        try:
            project_root_temp = Path(__file__).resolve().parent.parent.parent
            csv_path = (
                project_root_temp
                / "data"
                / "raw"
                / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
            )
            df = pd.read_csv(csv_path)
            print("Dados carregados do CSV.")
        except FileNotFoundError:
            print(f"Erro: Arquivo CSV não encontrado em {csv_path}.")
            return pd.DataFrame(), pd.DataFrame()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "EmployeeNumber" in df.columns:
        df["EmployeeNumber"] = df["EmployeeNumber"].astype(int)

    # --- DataFrame para UI (manter colunas originais relevantes) ---
    df_for_ui = df.copy()

    if "Attrition" in df_for_ui.columns:
        df_for_ui["Attrition"] = df_for_ui["Attrition"].map({"Yes": 1, "No": 0})
    if "JobSatisfaction" in df_for_ui.columns:
        df_for_ui["high_job_satisfaction"] = (df_for_ui["JobSatisfaction"] >= 3).astype(
            int
        )
    if "OverTime" in df_for_ui.columns:
        df_for_ui["OverTime_Yes"] = (
            df_for_ui["OverTime"].map({"Yes": 1, "No": 0}).astype(int)
        )

    ui_cols = [
        "EmployeeNumber",
        "Age",
        "Department",
        "JobRole",
        "Attrition",
        "MonthlyIncome",
        "JobSatisfaction",
        "OverTime",
        "high_job_satisfaction",
        "OverTime_Yes",
    ]
    df_for_ui = df_for_ui[[col for col in ui_cols if col in df_for_ui.columns]].copy()

    # --- DataFrame para o Modelo de ML (Aplicar preprocess_employee_data centralizado) ---
    df_model = preprocess_employee_data(df, model_features=model_features_list)

    if model_features_list is not None:
        print("DataFrame reindexado para corresponder às features do modelo.")

    return df_model, df_for_ui


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent.parent.parent
    MODEL_FEATURES_PATH_FOR_TEST = (
        current_dir / "artifacts" / "features" / "features.pkl"
    )

    try:
        if MODEL_FEATURES_PATH_FOR_TEST.exists():
            import joblib

            features_list_for_test = joblib.load(MODEL_FEATURES_PATH_FOR_TEST)
            print(
                f"model_features carregado para teste: {len(features_list_for_test)} features."
            )
            df_model, df_ui = load_and_preprocess_data(
                model_features_list=features_list_for_test
            )
        else:
            print(
                f"Aviso: {MODEL_FEATURES_PATH_FOR_TEST} não encontrado. Rodando sem model_features_list para teste."
            )
            df_model, df_ui = load_and_preprocess_data()

        if not df_model.empty:
            print("\n--- DataFrame para Modelo (df_model) ---")
            print(df_model.head())
            print(df_model.info())
            if MODEL_FEATURES_PATH_FOR_TEST.exists():
                features_list_for_test = joblib.load(MODEL_FEATURES_PATH_FOR_TEST)
                missing_in_df = set(features_list_for_test) - set(df_model.columns)
                extra_in_df = set(df_model.columns) - set(features_list_for_test)
                print(f"Faltando em df_model (mas em model_features): {missing_in_df}")
                print(f"Extra em df_model (mas não em model_features): {extra_in_df}")
                print(f"Número de colunas em df_model: {len(df_model.columns)}")
                print(f"Número de features no modelo: {len(features_list_for_test)}")

        if not df_ui.empty:
            print("\n--- DataFrame para UI (df_ui) ---")
            print(df_ui.head())
            print(df_ui.info())

    except Exception as e:
        print(f"Erro durante o teste de load_and_preprocess_data: {e}")
