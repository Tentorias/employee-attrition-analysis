import os
from dotenv import load_dotenv 


load_dotenv()


import joblib
import pandas as pd
import numpy as np
import shap
import psycopg2 
import json 
from fastapi import FastAPI, HTTPException
from datetime import datetime 
from .schemas import EmployeeData, PredictionOut

app = FastAPI(
    title="API de Predição de Attrition de Funcionários",
    description="API para prever a probabilidade de um funcionário deixar a empresa e registrar os resultados.",
    version="1.1.0" 
)


DATABASE_URL = os.getenv("DATABASE_URL")


def create_prediction_log_table():
    """Cria a tabela de logs de predição no banco de dados, se não existir."""
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE,
                prediction VARCHAR(3),
                probability_yes REAL,
                employee_data JSONB,
                explanation JSONB
            );
        """)
        conn.commit()
        print("✅ Tabela 'prediction_logs' verificada/criada com sucesso.")
    except psycopg2.OperationalError as e:
        print(f"❌ Erro de conexão com o banco de dados: {e}")
    except Exception as e:
        print(f"❌ Erro ao criar a tabela no PostgreSQL: {e}")
    finally:
        if conn:
            conn.close()

# --- Carregamento dos Artefatos de Machine Learning ---
MODEL_PATH = os.path.join("models", "production_model.pkl")
FEATURES_PATH = os.path.join("artifacts", "features", "features.pkl")
EXPLAINER_PATH = os.path.join("models", "production_shap_explainer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    print(f"✅ Modelo, explicador SHAP e lista de features ({len(model_features)} colunas) carregados com sucesso.")
    create_prediction_log_table()
except FileNotFoundError as e:
    print(f"❌ Erro crítico ao carregar artefatos: {e}.")
    raise

@app.get("/health", summary="Verifica a saúde da API")
async def health_check():
    """Endpoint para verificar se a API está operacional."""
    return {"status": "ok", "database_connection": "configured" if DATABASE_URL else "not_configured"}

@app.post("/predict", response_model=PredictionOut, summary="Realiza a predição de attrition")
async def predict(employee_data: EmployeeData):
    """
    Prevê a probabilidade de attrition e salva a requisição e o resultado no banco de dados.
    """
    try:
        input_df = pd.DataFrame([employee_data.dict()])

        cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
        input_df.drop(columns=[col for col in cols_to_drop if col in input_df.columns], inplace=True)
        if 'NumCompaniesWorked' in input_df.columns and 'TotalWorkingYears' in input_df.columns:
            # Evita divisão por zero
            denominator = input_df['NumCompaniesWorked'].replace(0, 1)
            input_df['YearsPerCompany'] = input_df['TotalWorkingYears'] / denominator

        categorical_cols = input_df.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dtype=float)

        final_df = input_df.reindex(columns=model_features, fill_value=0)

        actual_model = model.named_steps['classifier'] if hasattr(model, 'steps') else model

        probability_yes = actual_model.predict_proba(final_df)[0, 1]
        prediction = "Yes" if probability_yes >= 0.5 else "No"

        shap_values = explainer.shap_values(final_df)

        explanation_raw = dict(zip(final_df.columns, shap_values[0]))
        explanation = {k: float(v) for k, v in explanation_raw.items()}
        explanation_sorted = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))

        # --- Lógica para salvar no Banco de Dados ---
        if DATABASE_URL:
            conn = None
            try:
                conn = psycopg2.connect(DATABASE_URL)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO prediction_logs (timestamp, prediction, probability_yes, employee_data, explanation)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.now(),
                        prediction,
                        float(probability_yes),
                        json.dumps(employee_data.dict()), 
                        json.dumps(explanation_sorted)   
                    )
                )
                conn.commit()
            except Exception as e:
                print(f"⚠️ Aviso: Falha ao salvar log no banco de dados: {e}")
            finally:
                if conn:
                    conn.close()
        # --- Fim da lógica do Banco de Dados ---

        return {
            "prediction": prediction,
            "probability_yes": float(probability_yes),
            "explanation": explanation_sorted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a predição: {str(e)}")