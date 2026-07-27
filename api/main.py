# api/main.py

import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine

from .schemas import EmployeeData, PredictionOut
from attrition.features import preprocess_employee_data

# --- Carregar variáveis de ambiente ---
load_dotenv()

app = FastAPI(
    title="API de Predição de Attrition de Funcionários",
    description="API para prever a probabilidade de um funcionário deixar a empresa e logar o resultado.",
    version="1.0.1",
)

# --- Carregamento dos Artefactos ---
MODEL_PATH = os.path.join("models", "production_model.pkl")
FEATURES_PATH = os.path.join("artifacts", "features", "features.pkl")
EXPLAINER_PATH = os.path.join("models", "production_shap_explainer.pkl")
THRESHOLD_PATH = os.path.join("artifacts", "models", "optimal_threshold.pkl")
DATABASE_URL = os.getenv("DATABASE_URL")

try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    optimal_threshold = joblib.load(THRESHOLD_PATH)
    engine = create_engine(DATABASE_URL) if DATABASE_URL else None
    print("[OK] Modelo, explicador, lista de features e threshold carregados.")
    if engine:
        print("[OK] Conexão com o banco de dados estabelecida.")
    else:
        print(
            "[AVISO] DATABASE_URL não encontrada. O logging de predições está desativado."
        )

except FileNotFoundError as e:
    print(f"[ERRO] Erro crítico ao carregar artefactos: {e}.")
    raise
except Exception as e:
    print(f"[ERRO] Erro inesperado ao carregar artefactos: {e}.")
    raise


@app.get("/health", summary="Verifica a saúde da API")
async def health_check():
    return {"status": "ok"}


@app.post(
    "/predict", response_model=PredictionOut, summary="Realiza a predição de attrition"
)
async def predict(employee_data: EmployeeData):
    try:
        input_df = pd.DataFrame([employee_data.dict()])

        # --- PRÉ-PROCESSAMENTO CENTRALIZADO ---
        final_df = preprocess_employee_data(input_df, model_features=model_features)

        actual_model = (
            model.named_steps["classifier"] if hasattr(model, "steps") else model
        )
        probability_yes = actual_model.predict_proba(final_df)[:, 1][0]

        prediction = "Yes" if probability_yes >= optimal_threshold else "No"

        shap_values = explainer.shap_values(final_df)
        explanation_raw = dict(zip(final_df.columns, shap_values[0]))
        explanation = {k: float(v) for k, v in explanation_raw.items()}
        explanation = dict(
            sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True)
        )

        # --- LÓGICA DE LOGGING NO BANCO DE DADOS ---
        if engine:
            try:
                log_df = pd.DataFrame(
                    {
                        "EmployeeNumber": [employee_data.EmployeeNumber],
                        "predicted_probability": [probability_yes],
                        "prediction_timestamp": [pd.Timestamp.now()],
                        "prediction_label": [prediction],
                    }
                )
                log_df.to_sql(
                    "predictions", con=engine, if_exists="append", index=False
                )
            except Exception as db_error:
                print(f"[ERRO] Erro ao salvar predição no banco de dados: {db_error}")

        return {
            "prediction": prediction,
            "probability_yes": float(probability_yes),
            "explanation": explanation,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro interno ao processar a predição: {str(e)}"
        )
