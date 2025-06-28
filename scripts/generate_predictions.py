import sqlite3
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging

# --- Configuração do Logging  ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuração de Caminhos ---
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "hr_analytics.db"
MODEL_PATH = BASE_DIR / "models" / "production_model.pkl"
EMPLOYEES_TABLE = "employees"
PREDICTIONS_TABLE = "predictions"


def load_data_from_db(db_path: Path, table_name: str) -> pd.DataFrame:
    """Carrega dados de uma tabela específica do banco de dados SQLite."""
    logging.info(f"Conectando ao banco de dados em '{db_path}' e lendo a tabela '{table_name}'...")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    logging.info(f"✔ {len(df)} registros carregados com sucesso.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a engenharia de features nos dados brutos."""
    logging.info("Aplicando engenharia de features nos dados...")
    df_processed = df.copy()


    df_processed['YearsPerCompany'] = (df_processed['YearsAtCompany'] / (df_processed['NumCompaniesWorked'] + 1)).round(2)
    df_processed['MonthlyIncome_log'] = np.log(df_processed['MonthlyIncome'] + 1)
    df_processed['TotalWorkingYears_log'] = np.log(df_processed['TotalWorkingYears'] + 1)


    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.drop('Attrition', errors='ignore')
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    logging.info("✔ Engenharia de features aplicada.")
    return df_encoded


def make_predictions(df_features: pd.DataFrame, model_path: Path) -> np.ndarray:
    """Carrega o modelo, alinha as features e gera as predições de probabilidade."""
    logging.info("Carregando modelo de produção...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    model_feature_names = model.get_booster().feature_names
    logging.info(f"✔ Modelo carregado. O modelo espera {len(model_feature_names)} features.")

    logging.info("Alinhando colunas do DataFrame com as features do modelo...")
    X_final = df_features.reindex(columns=model_feature_names, fill_value=0)
    logging.info(f"✔ DataFrame final preparado com {X_final.shape[1]} colunas.")

    logging.info("Gerando predições de probabilidade de turnover...")
    probabilities = model.predict_proba(X_final)[:, 1]
    logging.info("✔ Predições geradas.")
    return probabilities


def save_predictions_to_db(df_original: pd.DataFrame, predictions: np.ndarray, db_path: Path, table_name: str):
    """Monta o DataFrame de resultados e salva no banco de dados."""
    logging.info("Montando e salvando tabela de resultados...")
    results_df = pd.DataFrame({
        'EmployeeNumber': df_original['EmployeeNumber'],
        'Department': df_original['Department'],
        'JobRole': df_original['JobRole'],
        'Attrition_Actual': df_original['Attrition'],
        'predicted_probability': predictions
    })
    results_df['predicted_probability'] = results_df['predicted_probability'].round(4)
    
    with sqlite3.connect(db_path) as conn:
        results_df.to_sql(table_name, conn, if_exists='replace', index=False)
    logging.info(f"✔ - Predições salvas com sucesso na tabela '{table_name}'!")


def main():
    """Função principal que orquestra todo o processo de geração de predições."""
    logging.info("🚀 - Iniciando pipeline de geração de predições em massa...")
    try:
        
        df_employees = load_data_from_db(DB_PATH, EMPLOYEES_TABLE)
        
       
        df_featured = engineer_features(df_employees)
        
        
        predictions = make_predictions(df_featured, MODEL_PATH)
        
        
        save_predictions_to_db(df_employees, predictions, DB_PATH, PREDICTIONS_TABLE)

        logging.info("\n🏛️ - Marco Concluído: A camada preditiva foi adicionada à base de dados.")

    except Exception as e:
        logging.error(f"❌ - Ocorreu um erro inesperado no pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()

