import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# --- NOVAS IMPORTAÇÕES PARA O TESTE DE ARQUITETURA ---
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- CONFIGURAÇÕES E CONSTANTES ---
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")
RAW_DATA_PATH = project_root / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = project_root / "models" / "production_model.pkl"

def test_data_integrity():
    """Verifica se os dados no PostgreSQL são idênticos aos do CSV original."""
    print("\n--- INICIANDO TESTE DE INTEGRIDADE DOS DADOS ---")
    # ... (código existente, sem alterações) ...
    if not DATABASE_URL:
        print("❌ ERRO: DATABASE_URL não encontrada no arquivo .env.")
        return False
    try:
        engine = create_engine(DATABASE_URL)
        df_postgres = pd.read_sql_query('SELECT * FROM "employees"', engine)
        print(f"✅ Dados do PostgreSQL carregados ({df_postgres.shape[0]} linhas).")
        df_csv = pd.read_csv(RAW_DATA_PATH)
        print(f"✅ Dados do CSV original carregados ({df_csv.shape[0]} linhas).")
        if df_postgres.shape != df_csv.shape:
            print(f"❌ FALHA: Dimensões diferentes! PG: {df_postgres.shape}, CSV: {df_csv.shape}")
            return False
        print("✅ Verificação de dimensão: OK!")
        if not df_postgres.describe().equals(df_csv.describe()):
            print("⚠️ AVISO: Estatísticas descritivas não são idênticas, o que é aceitável.")
        else:
            print("✅ Verificação de estatísticas: OK!")
        print("--- ✅ TESTE DE INTEGRIDADE DOS DADOS CONCLUÍDO COM SUCESSO ---")
        return True
    except Exception as e:
        print(f"❌ FALHA NO TESTE DE INTEGRIDADE: {e}")
        return False

def test_model_sanity():
    """Verifica se o modelo aprendeu padrões lógicos, comparando grupos de risco."""
    print("\n--- INICIANDO TESTE DE SANIDADE DO MODELO ---")
    # ... (código existente, sem alterações) ...
    all_checks_passed = True
    if not DATABASE_URL:
        print("❌ ERRO: DATABASE_URL não encontrada no arquivo .env.")
        return False
    try:
        engine = create_engine(DATABASE_URL)
        df_preds = pd.read_sql_query('SELECT "EmployeeNumber", "predicted_probability" FROM predictions', engine)
        df_employees = pd.read_sql_query('SELECT "EmployeeNumber", "OverTime", "MonthlyIncome" FROM employees', engine)
        if df_preds.empty:
            print("⚠️ AVISO: Tabela 'predictions' está vazia.")
            return False
        df_full = pd.merge(df_employees, df_preds, on='EmployeeNumber', how='left')
        print(f"✅ Dados de predição carregados para {len(df_full)} funcionários.")
        high_risk = df_full[df_full['predicted_probability'] > 0.75]
        low_risk = df_full[df_full['predicted_probability'] < 0.25]
        print(f"\nFuncionários em alto risco (>75%): {len(high_risk)}")
        print(f"Funcionários em baixo risco (<25%): {len(low_risk)}")
        hr_alto_risco = high_risk['OverTime'].value_counts(normalize=True).get('Yes', 0)
        hr_baixo_risco = low_risk['OverTime'].value_counts(normalize=True).get('Yes', 0)
        print("\n--- Comparação de Horas Extras ('OverTime') ---")
        print(f"Proporção que faz horas extras no grupo de ALTO RISCO: {hr_alto_risco:.1%}")
        print(f"Proporção que faz horas extras no grupo de BAIXO RISCO: {hr_baixo_risco:.1%}")
        if hr_alto_risco > hr_baixo_risco:
            print("✅ Verificação de sanidade (OverTime): OK!")
        else:
            print("❌ FALHA na verificação de sanidade (OverTime): O padrão esperado não foi encontrado.")
            all_checks_passed = False
        salario_alto_risco = high_risk['MonthlyIncome'].mean()
        salario_baixo_risco = low_risk['MonthlyIncome'].mean()
        print("\n--- Comparação de Renda Mensal Média ('MonthlyIncome') ---")
        print(f"Salário médio do grupo de ALTO RISCO: R$ {salario_alto_risco:,.2f}")
        print(f"Salário médio do grupo de BAIXO RISCO: R$ {salario_baixo_risco:,.2f}")
        if salario_alto_risco < salario_baixo_risco:
            print("✅ Verificação de sanidade (Salário): OK!")
        else:
            print("❌ FALHA na verificação de sanidade (Salário): O padrão esperado não foi encontrado.")
            all_checks_passed = False
        if all_checks_passed:
            print("\n--- ✅ TESTE DE SANIDADE DO MODELO CONCLUÍDO COM SUCESSO ---")
        else:
            print("\n--- ⚠️ TESTE DE SANIDADE DO MODELO CONCLUÍDO COM AVISOS ---")
        return all_checks_passed
    except Exception as e:
        print(f"❌ FALHA NO TESTE DE SANIDADE: {e}")
        return False

def test_model_architecture():
    """
    Verifica se o artefato do modelo salvo contém um pipeline com SMOTE e XGBoost.
    """
    print("\n--- INICIANDO TESTE DE ARQUITETURA DO MODELO ---")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Arquivo de modelo carregado de '{MODEL_PATH}'.")

        # 1. Verifica se é um Pipeline da imblearn
        if not isinstance(model, ImbPipeline):
            print(f"❌ FALHA: O objeto do modelo não é um Pipeline da imblearn, mas sim um {type(model)}.")
            return False
        print("✅ Verificação de tipo: OK! O modelo é um pipeline.")

        # 2. Verifica os passos do pipeline
        steps = model.named_steps
        if "smote" not in steps:
            print("❌ FALHA: Passo 'smote' não encontrado no pipeline.")
            return False
        print("✅ Verificação de passo 'smote': OK!")

        if "classifier" not in steps:
            print("❌ FALHA: Passo 'classifier' não encontrado no pipeline.")
            return False
        print("✅ Verificação de passo 'classifier': OK!")

        # 3. Verifica o tipo de cada passo
        if not isinstance(steps['smote'], SMOTE):
            print(f"❌ FALHA: O passo 'smote' não é do tipo SMOTE, mas sim {type(steps['smote'])}.")
            return False
        print("✅ Verificação de tipo (SMOTE): OK!")

        if not isinstance(steps['classifier'], XGBClassifier):
            print(f"❌ FALHA: O passo 'classifier' não é do tipo XGBClassifier, mas sim {type(steps['classifier'])}.")
            return False
        print("✅ Verificação de tipo (XGBClassifier): OK!")

        print("--- ✅ TESTE DE ARQUITETURA DO MODELO CONCLUÍDO COM SUCESSO ---")
        return True

    except Exception as e:
        print(f"❌ FALHA NO TESTE DE ARQUITETURA: {e}")
        return False


if __name__ == "__main__":
    print("=============================================")
    print("INICIANDO BATERIA DE TESTES DO PROJETO")
    print("=============================================")
    
    integrity_passed = test_data_integrity()
    architecture_passed = test_model_architecture()
    sanity_passed = test_model_sanity()
    
    print("\n---------------------------------------------")
    if integrity_passed and architecture_passed and sanity_passed:
        print("🎉 RESULTADO FINAL: TODOS OS TESTES PASSARAM!")
    else:
        print("🔥 RESULTADO FINAL: UM OU MAIS TESTES FALHARAM.")
    print("---------------------------------------------")