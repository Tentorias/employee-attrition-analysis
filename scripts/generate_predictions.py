import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Carregar Configurações do Ambiente ---
load_dotenv()

# --- Configuração de Caminhos e Variáveis ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "production_model.pkl"
FEATURES_PATH = BASE_DIR / "artifacts" / "features" / "features.pkl"
POSTGRES_URL = os.getenv("DATABASE_URL")
EMPLOYEES_TABLE = "employees"
PREDICTIONS_TABLE = "predictions_batch" 

def load_data_from_postgres(db_url: str, table_name: str) -> pd.DataFrame:
    """Carrega dados de uma tabela específica do banco de dados PostgreSQL."""
    if not db_url:
        raise ValueError("A variável de ambiente DATABASE_URL não foi encontrada. Verifique o seu ficheiro .env.")
    
    logging.info(f"Conectando ao PostgreSQL e lendo a tabela '{table_name}'...")
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        logging.info(f"✔ {len(df)} registos carregados com sucesso.")
        return df
    except Exception as e:
        logging.error(f"Não foi possível ler a tabela '{table_name}'. Erro: {e}")
        return pd.DataFrame()

def prepare_data_for_model(df, features_list):
    """Prepara um DataFrame para ser usado pelo modelo, aplicando a mesma engenharia de features."""
    logging.info("Aplicando engenharia de features...")
    df_proc = df.copy()
    
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18']
    df_proc.drop(columns=[col for col in cols_to_drop if col in df_proc.columns], errors='ignore', inplace=True)
    
    if 'TotalWorkingYears' in df_proc.columns and 'NumCompaniesWorked' in df_proc.columns:
        df_proc['YearsPerCompany'] = df_proc.apply(
            lambda row: row['TotalWorkingYears'] / row['NumCompaniesWorked'] if row['NumCompaniesWorked'] > 0 else row['TotalWorkingYears'],
            axis=1
        ).round(4)
        
    df_proc = pd.get_dummies(df_proc, drop_first=True, dtype=float)
    X = df_proc.reindex(columns=features_list, fill_value=0)
    logging.info("✔ Engenharia de features aplicada.")
    return X

def make_predictions(df_features: pd.DataFrame, model_path: Path) -> np.ndarray:
    """Carrega o modelo e gera as predições de probabilidade."""
    logging.info(f"Carregando modelo de produção de '{model_path}'...")
    model = joblib.load(model_path)
    
    logging.info("Gerando predições de probabilidade de turnover...")
    probabilities = model.predict_proba(df_features)[:, 1]
    logging.info("✔ Predições geradas.")
    return probabilities

def save_predictions_to_postgres(df_original: pd.DataFrame, predictions: np.ndarray, db_url: str, table_name: str):
    """Monta o DataFrame de resultados e salva no banco de dados PostgreSQL."""
    logging.info("Montando e salvando tabela de resultados no PostgreSQL...")
    results_df = pd.DataFrame({
        'EmployeeNumber': df_original['EmployeeNumber'],
        'Department': df_original['Department'],
        'JobRole': df_original['JobRole'],
        'Attrition_Actual': df_original['Attrition'],
        'predicted_probability': predictions
    })
    results_df['predicted_probability'] = results_df['predicted_probability'].round(4)
    
    engine = create_engine(db_url)
    results_df.to_sql(table_name, engine, if_exists='replace', index=False)
    logging.info(f"✔ Predições salvas com sucesso na tabela '{table_name}'!")

def main():
    """Função principal que orquestra todo o processo de geração de predições."""
    logging.info("🚀 - Iniciando pipeline de geração de predições em lote para PostgreSQL...")
    try:
        df_employees = load_data_from_postgres(POSTGRES_URL, EMPLOYEES_TABLE)
        if df_employees.empty:
            logging.warning("A tabela de funcionários está vazia ou não pôde ser lida. Abortando.")
            return

        features_list = joblib.load(FEATURES_PATH)
        df_prepared = prepare_data_for_model(df_employees, features_list)
        
        probabilities = make_predictions(df_prepared, MODEL_PATH)
        
        save_predictions_to_postgres(df_employees, probabilities, POSTGRES_URL, PREDICTIONS_TABLE)

        logging.info("\n🏛️ - Marco Concluído: A tabela de predições em lote foi criada/atualizada no PostgreSQL.")

    except Exception as e:
        logging.error(f"❌ - Ocorreu um erro inesperado no pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()