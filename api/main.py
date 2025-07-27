# api/main.py

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
from dotenv import load_dotenv

from .schemas import EmployeeData, PredictionOut

# --- Carregar variáveis de ambiente ---
load_dotenv()

app = FastAPI(
    title="API de Predição de Attrition de Funcionários",
    description="API para prever a probabilidade de um funcionário deixar a empresa e logar o resultado.",
    version="1.0.1" 
)

# --- Carregamento dos Artefactos ---
MODEL_PATH = os.path.join("models", "production_model.pkl")
FEATURES_PATH = os.path.join("artifacts", "features", "features.pkl")
EXPLAINER_PATH = os.path.join("models", "production_shap_explainer.pkl")
THRESHOLD_PATH = os.path.join("artifacts", "models", "optimal_threshold.pkl") # NOVO: Caminho para o threshold
DATABASE_URL = os.getenv("DATABASE_URL")

try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    optimal_threshold = joblib.load(THRESHOLD_PATH) # NOVO: Carregar o threshold
    engine = create_engine(DATABASE_URL) if DATABASE_URL else None
    print(f"✅ Modelo, explicador, lista de features e threshold carregados.")
    if engine:
        print("✅ Conexão com o banco de dados estabelecida.")
    else:
        print("⚠️ Aviso: DATABASE_URL não encontrada. O logging de predições está desativado.")

except FileNotFoundError as e:
    print(f"❌ Erro crítico ao carregar artefactos: {e}.")
    raise
except Exception as e:
    print(f"❌ Erro inesperado ao carregar artefactos: {e}.") # Capturar outros erros no carregamento
    raise


@app.get("/health", summary="Verifica a saúde da API")
async def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut, summary="Realiza a predição de attrition")
async def predict(employee_data: EmployeeData):
    try:
        input_df = pd.DataFrame([employee_data.dict()])

        # As transformações de pré-processamento aqui precisam ser *idênticas*
        # às de train.py/preprocess e data_processing.py.
        # É um ponto crítico de consistência!

        cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18']
        input_df.drop(columns=[col for col in cols_to_drop if col in input_df.columns], errors='ignore', inplace=True)
        
        # Consistentemente com train.py/data_processing.py
        if 'NumCompaniesWorked' in input_df.columns and 'TotalWorkingYears' in input_df.columns:
            # Replicar a lógica de YearsPerCompany
            input_df['YearsPerCompany'] = input_df['TotalWorkingYears'] / (input_df['NumCompaniesWorked'].replace(0, 1)) # Usar replace(0,1)

        if 'MonthlyIncome' in input_df.columns:
            input_df['MonthlyIncome_log'] = np.log1p(input_df['MonthlyIncome'])
        
        if 'TotalWorkingYears' in input_df.columns:
            input_df['TotalWorkingYears_log'] = np.log1p(input_df['TotalWorkingYears'])

        categorical_cols = input_df.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dtype=float)

        # Escalonamento se o modelo espera (seu train.py não tem scaler, mas data_processing.py sim)
        # Se seu modelo de produção foi treinado com dados escalados, você precisa de um scaler aqui
        # Caso contrário, remova os passos de escalamento em data_processing.py para o df_model.
        # Por enquanto, assumimos que o scaler NÃO está no modelo salvo.

        final_df = input_df.reindex(columns=model_features, fill_value=0)
        
        actual_model = model.named_steps['classifier'] if hasattr(model, 'steps') else model
        probability_yes = actual_model.predict_proba(final_df)[0, 1]
        
        # Usar o threshold carregado
        prediction = "Yes" if probability_yes >= optimal_threshold else "No" 

        shap_values = explainer.shap_values(final_df)
        explanation_raw = dict(zip(final_df.columns, shap_values[0]))
        explanation = {k: float(v) for k, v in explanation_raw.items()}
        explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))

        # --- LÓGICA DE LOGGING NO BANCO de DADOS ---
        if engine:
            try:
                log_df = pd.DataFrame({
                    'EmployeeNumber': [employee_data.EmployeeNumber],
                    'predicted_probability': [probability_yes],
                    'prediction_timestamp': [pd.Timestamp.now()],
                    'prediction_label': [prediction] # Adicionar a label da predição
                })
                log_df.to_sql('predictions', con=engine, if_exists='append', index=False)
            except Exception as db_error:
                print(f"⚠️ Erro ao salvar predição no banco de dados: {db_error}")

        return {
            "prediction": prediction,
            "probability_yes": float(probability_yes),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a predição: {str(e)}")