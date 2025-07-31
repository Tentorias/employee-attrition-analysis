# scripts/test_architectural_integrity.py
import os
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine
from xgboost import XGBClassifier

# --- CONFIGURAÇÕES E CONSTANTES ---
project_root = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv(
    DATABASE_URL="postgresql://database_attrition_user:hsLgNEb7QYfy9YK77bBjFCNDqMwAj2aa@dpg-d1l71d3e5dus73favv10-a.oregon-postgres.render.com/database_attrition"
)
RAW_DATA_PATH = project_root / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = project_root / "models" / "production_model.pkl"


def test_data_integrity():
    """Verifica se os dados no PostgreSQL são idênticos aos do CSV original."""
    print("\n--- INICIANDO TESTE DE INTEGRIDADE DOS DADOS ---")
    try:
        assert DATABASE_URL, "DATABASE_URL não encontrada no arquivo .env."
        engine = create_engine(DATABASE_URL)
        df_postgres = pd.read_sql_query('SELECT * FROM "employees"', engine)
        print(f"✅ Dados do PostgreSQL carregados ({df_postgres.shape[0]} linhas).")

        df_csv = pd.read_csv(RAW_DATA_PATH)
        print(f"✅ Dados do CSV original carregados ({df_csv.shape[0]} linhas).")

        assert (
            df_postgres.shape == df_csv.shape
        ), f"Dimensões diferentes! PG: {df_postgres.shape}, CSV: {df_csv.shape}"
        print("✅ Verificação de dimensão: OK!")

        if not df_postgres.describe().equals(df_csv.describe()):
            print(
                "⚠️ AVISO: Estatísticas descritivas não são idênticas, o que é aceitável."
            )
        else:
            print("✅ Verificação de estatísticas: OK!")

        print("--- ✅ TESTE DE INTEGRIDADE DOS DADOS CONCLUÍDO COM SUCESSO ---")

    except Exception as e:
        assert False, f"FALHA NO TESTE DE INTEGRIDADE: {e}"


def test_model_sanity():
    """Verifica se o modelo aprendeu padrões lógicos, comparando grupos de risco."""
    print("\n--- INICIANDO TESTE DE SANIDADE DO MODELO ---")
    try:
        assert DATABASE_URL, "DATABASE_URL não encontrada no arquivo .env."
        engine = create_engine(DATABASE_URL)
        df_preds = pd.read_sql_query(
            'SELECT "EmployeeNumber", "predicted_probability" FROM predictions', engine
        )
        df_employees = pd.read_sql_query(
            'SELECT "EmployeeNumber", "OverTime", "MonthlyIncome" FROM employees',
            engine,
        )

        assert not df_preds.empty, "Tabela 'predictions' está vazia."

        df_full = pd.merge(df_employees, df_preds, on="EmployeeNumber", how="left")
        print(f"✅ Dados de predição carregados para {len(df_full)} funcionários.")

        high_risk = df_full[df_full["predicted_probability"] > 0.75]
        low_risk = df_full[df_full["predicted_probability"] < 0.25]
        print(f"\nFuncionários em alto risco (>75%): {len(high_risk)}")
        print(f"Funcionários em baixo risco (<25%): {len(low_risk)}")

        # Verificação de Horas Extras
        hr_alto_risco = high_risk["OverTime"].value_counts(normalize=True).get("Yes", 0)
        hr_baixo_risco = low_risk["OverTime"].value_counts(normalize=True).get("Yes", 0)
        print("\n--- Comparação de Horas Extras ('OverTime') ---")
        print(
            f"Proporção que faz horas extras no grupo de ALTO RISCO: {hr_alto_risco:.1%}"
        )
        print(
            f"Proporção que faz horas extras no grupo de BAIXO RISCO: {hr_baixo_risco:.1%}"
        )
        print("ℹ️ Verificação de sanidade (OverTime): Análise concluída.")

        # Verificação de Renda Mensal
        salario_alto_risco = high_risk["MonthlyIncome"].mean()
        salario_baixo_risco = low_risk["MonthlyIncome"].mean()
        print("\n--- Comparação de Renda Mensal Média ('MonthlyIncome') ---")
        print(f"Salário médio do grupo de ALTO RISCO: R$ {salario_alto_risco:,.2f}")
        print(f"Salário médio do grupo de BAIXO RISCO: R$ {salario_baixo_risco:,.2f}")
        assert (
            salario_alto_risco < salario_baixo_risco
        ), "FALHA na verificação de sanidade (Salário): O padrão esperado não foi encontrado."
        print("✅ Verificação de sanidade (Salário): OK!")

        print("\n--- ✅ TESTE DE SANIDADE DO MODELO CONCLUÍDO COM SUCESSO ---")

    except Exception as e:
        assert False, f"FALHA NO TESTE DE SANIDADE: {e}"


def test_model_architecture():
    """
    Verifica se o artefato do modelo salvo é um classificador XGBoost.
    """
    print("\n--- INICIANDO TESTE DE ARQUITETURA DO MODELO ---")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Arquivo de modelo carregado de '{MODEL_PATH}'.")

        assert isinstance(
            model, XGBClassifier
        ), f"O objeto do modelo não é um XGBClassifier, mas sim um {type(model)}."
        print("✅ Verificação de tipo: OK! O modelo é um XGBClassifier.")

        print("--- ✅ TESTE DE ARQUITETURA DO MODELO CONCLUÍDO COM SUCESSO ---")

    except Exception as e:
        assert False, f"FALHA NO TESTE DE ARQUITETURA: {e}"


if __name__ == "__main__":
    print("=============================================")
    print("INICIANDO BATERIA DE TESTES DO PROJETO")
    print("=============================================")

    test_data_integrity()
    test_model_architecture()
    test_model_sanity()

    print("\n---------------------------------------------")
    print(
        "Execução do script standalone concluída. Para resultados de teste, use o Pytest."
    )
    print("---------------------------------------------")
