import sqlite3
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# --- Configuração de Caminhos ---
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "hr_analytics.db"
MODEL_PATH = BASE_DIR / "models" / "production_model.pkl"

EMPLOYEES_TABLE = "employees"
PREDICTIONS_TABLE = "predictions"


def generate_predictions():
    """
    Script completo para carregar os dados dos funcionários, recriar a engenharia de
    features, alinhar com as features exatas do modelo e salvar as predições.
    """
    print("🚀 Iniciando script de geração de predições em massa...")

    # --- Validação dos Artefatos ---
    if not MODEL_PATH.exists():
        print(f"❌ ERRO: Modelo de produção não encontrado em '{MODEL_PATH}'.")
        return

    try:
        # --- 1. Carregar Dados e Modelo ---
        print(f"Conectando ao banco de dados em '{DB_PATH}'...")
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {EMPLOYEES_TABLE}", conn)
        print(f"✔ {len(df)} registros de funcionários carregados.")

        print("Carregando modelo de produção...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Pega a lista de features diretamente do modelo treinado.
        # Esta é a fonte única da verdade para as colunas esperadas.
        model_feature_names = model.get_booster().feature_names
        print(f"✔ Modelo carregado. O modelo espera {len(model_feature_names)} features.")

        # --- 2. Engenharia de Features Idêntica ao Treinamento ---
        print("Aplicando engenharia de features nos dados...")
        
        df_processed = df.copy()

        # Criação de features de engenharia
        # Adiciona 1 para evitar divisão por zero se NumCompaniesWorked for 0.
        df_processed['YearsPerCompany'] = (df_processed['YearsAtCompany'] / (df_processed['NumCompaniesWorked'] + 1)).round(2)
        
        # Transformações logarítmicas
        df_processed['MonthlyIncome_log'] = np.log(df_processed['MonthlyIncome'] + 1)
        df_processed['TotalWorkingYears_log'] = np.log(df_processed['TotalWorkingYears'] + 1)

        # Identifica colunas categóricas para One-Hot Encoding
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.drop('Attrition', errors='ignore')

        # Aplica o One-Hot Encoding
        df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        print("✔ Engenharia de features aplicada.")
        
        # --- 3. Alinhamento Final com o Modelo ---
        # Garante que o DataFrame final tenha exatamente as mesmas colunas (e na mesma ordem)
        # que o modelo usou para ser treinado. Colunas que o modelo espera, mas não
        # existem no df_encoded (raro), são preenchidas com 0. Colunas no df_encoded
        # que o modelo não espera são descartadas.
        print("Alinhando colunas do DataFrame com as features do modelo...")
        X_final = df_encoded.reindex(columns=model_feature_names, fill_value=0)
        print(f"✔ DataFrame final preparado com {X_final.shape[1]} colunas.")

        # --- 4. Gerar Predições de Probabilidade ---
        print("Gerando predições de probabilidade de turnover...")
        probabilities = model.predict_proba(X_final)[:, 1]
        print("✔ Predições geradas.")

        # --- 5. Preparar o DataFrame de Resultados ---
        print("Montando tabela de resultados...")
        results_df = pd.DataFrame({
            'EmployeeNumber': df['EmployeeNumber'],
            'Department': df['Department'],
            'JobRole': df['JobRole'],
            'Attrition_Actual': df['Attrition'],
            'predicted_probability': probabilities
        })
        results_df['predicted_probability'] = results_df['predicted_probability'].round(4)
        print("✔ Tabela de resultados pronta.")

        # --- 6. Salvar os Resultados no Banco de Dados ---
        print(f"Salvando predições na tabela '{PREDICTIONS_TABLE}'...")
        with sqlite3.connect(DB_PATH) as conn:
            results_df.to_sql(PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
        print(f"✔ Predições salvas com sucesso no banco de dados!")
        print("\n🏛️ - Marco Concluído: A camada preditiva foi adicionada à base de dados.")

    except Exception as e:
        print(f"\n❌ Ocorreu um erro inesperado durante o processo: {e}")
        print("Verifique se as colunas usadas no script correspondem às do seu dataset.")


if __name__ == "__main__":
    generate_predictions()
