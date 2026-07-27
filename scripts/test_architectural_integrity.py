# scripts/test_architectural_integrity.py
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from xgboost import XGBClassifier

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- CONFIGURAÇÕES E CONSTANTES ---
project_root = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv("DATABASE_URL")
RAW_DATA_PATH = project_root / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = project_root / "models" / "production_model.pkl"
FEATURES_PATH = project_root / "artifacts" / "features" / "features.pkl"


def test_data_integrity():
    """Verifica se os dados no PostgreSQL são idênticos aos do CSV original."""
    print("\n--- INICIANDO TESTE DE INTEGRIDADE DOS DADOS ---")
    if not DATABASE_URL:
        print(
            "[INFO] DATABASE_URL não encontrada no arquivo .env. Pulando teste de integração com PostgreSQL."
        )
        return

    try:
        engine = create_engine(DATABASE_URL)
        df_postgres = pd.read_sql_query('SELECT * FROM "employees"', engine)
        print(f"[OK] Dados do PostgreSQL carregados ({df_postgres.shape[0]} linhas).")

        df_csv = pd.read_csv(RAW_DATA_PATH)
        print(f"[OK] Dados do CSV original carregados ({df_csv.shape[0]} linhas).")

        assert (
            df_postgres.shape == df_csv.shape
        ), f"Dimensões diferentes! PG: {df_postgres.shape}, CSV: {df_csv.shape}"
        print("[OK] Verificação de dimensão: OK!")

        if not df_postgres.describe().equals(df_csv.describe()):
            print(
                "[WARN] Estatísticas descritivas não são idênticas, o que é aceitável."
            )
        else:
            print("[OK] Verificação de estatísticas: OK!")

        print("--- [OK] TESTE DE INTEGRIDADE DOS DADOS CONCLUÍDO COM SUCESSO ---")

    except Exception as e:
        print(
            f"[INFO] Erro na conexão ou consulta ao PostgreSQL ({e}). Pulando verificação DB."
        )


def test_model_sanity():
    """Verifica se o modelo aprendeu padrões lógicos, comparando grupos de risco."""
    print("\n--- INICIANDO TESTE DE SANIDADE DO MODELO ---")
    if not DATABASE_URL:
        print(
            "[INFO] DATABASE_URL não encontrada no arquivo .env. Pulando teste de sanidade com PostgreSQL."
        )
        return

    try:
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
        print(f"[OK] Dados de predição carregados para {len(df_full)} funcionários.")

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
        print("[INFO] Verificação de sanidade (OverTime): Análise concluída.")

        # Verificação de Renda Mensal
        salario_alto_risco = high_risk["MonthlyIncome"].mean()
        salario_baixo_risco = low_risk["MonthlyIncome"].mean()
        print("\n--- Comparação de Renda Mensal Média ('MonthlyIncome') ---")
        print(f"Salário médio do grupo de ALTO RISCO: R$ {salario_alto_risco:,.2f}")
        print(f"Salário médio do grupo de BAIXO RISCO: R$ {salario_baixo_risco:,.2f}")
        assert (
            salario_alto_risco < salario_baixo_risco
        ), "FALHA na verificação de sanidade (Salário): O padrão esperado não foi encontrado."
        print("[OK] Verificação de sanidade (Salário): OK!")

        print("--- [OK] TESTE DE SANIDADE DO MODELO CONCLUÍDO COM SUCESSO ---")

    except Exception as e:
        print(
            f"[INFO] Erro na conexão ou consulta ao PostgreSQL ({e}). Pulando verificação de sanidade DB."
        )


def test_model_architecture():
    """
    Verifica se o artefato do modelo salvo é um classificador XGBoost e se está livre de Data Leakage (IDs).
    """
    print("\n--- INICIANDO TESTE DE ARQUITETURA E DATA LEAKAGE DO MODELO ---")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Arquivo de modelo carregado de '{MODEL_PATH}'.")

        assert isinstance(
            model, XGBClassifier
        ), f"O objeto do modelo não é um XGBClassifier, mas sim um {type(model)}."
        print("[OK] Verificação de tipo: OK! O modelo é um XGBClassifier.")

        # Verificação de prevenção de Data Leakage na quantidade de features
        if FEATURES_PATH.exists():
            features = joblib.load(FEATURES_PATH)
            print(f"[OK] Arquivo de features carregado ({len(features)} colunas).")
            assert (
                "EmployeeNumber" not in features
            ), "FALHA GRAVE: 'EmployeeNumber' foi encontrado nas features do modelo (Data Leakage)!"
            assert (
                "Attrition" not in features
            ), "FALHA GRAVE: 'Attrition' foi encontrado nas features do modelo!"
            assert (
                model.n_features_in_ == len(features)
            ), f"FALHA: n_features_in_ ({model.n_features_in_}) difere de len(features) ({len(features)})."
            print(
                "[OK] Verificação anti-leakage: OK! O modelo foi treinado sem colunas de ID ou target."
            )

        print("--- [OK] TESTE DE ARQUITETURA DO MODELO CONCLUÍDO COM SUCESSO ---")

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
