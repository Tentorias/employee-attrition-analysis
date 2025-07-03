# api/main.py
# -*- coding: utf-8 -*-

"""
Módulo principal da API de Predição de Attrition de Funcionários.
"""

import os
import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from .schemas import EmployeeData, PredictionOut

# ----------------- CONFIGURAÇÃO DA API ----------------- #

app = FastAPI(
    title="API de Predição de Attrition de Funcionários",
    description="""
    API para prever a probabilidade de um funcionário deixar a empresa.
    Recebe os dados brutos de um funcionário e retorna a predição (Sim/Não),
    a probabilidade de saída e uma explicação dos fatores mais influentes
    baseada em valores SHAP.
    """,
    version="1.0.0"
)

# ----------------- CARREGAMENTO DOS ARTEFACTOS DE PRODUÇÃO ----------------- #

# Caminhos corretos para os artefactos de produção
MODEL_PATH = os.path.join("models", "production_model.pkl")
# CORREÇÃO DEFINITIVA: Apontar para o local onde o pipeline GERA o ficheiro de features.
FEATURES_PATH = os.path.join("artifacts", "features", "features.pkl") 
EXPLAINER_PATH = os.path.join("models", "production_shap_explainer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    print(f"✅ Modelo, explicador SHAP e lista de features ({len(model_features)} colunas) carregados com sucesso.")
except FileNotFoundError as e:
    print(f"❌ Erro crítico ao carregar artefactos: {e}.")
    print("Certifique-se de que todos os artefactos de produção existem e estão nos caminhos corretos.")
    raise

# ----------------- ENDPOINTS DA API ----------------- #

@app.get("/health", summary="Verifica a saúde da API")
async def health_check():
    """
    Endpoint de health check.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut, summary="Realiza a predição de attrition")
async def predict(employee_data: EmployeeData):
    """
    Recebe os dados de um funcionário e realiza a predição de attrition.
    """
    try:
        input_df = pd.DataFrame([employee_data.dict()])

        # Replicar a engenharia de features do pipeline
        cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18']
        input_df.drop(columns=[col for col in cols_to_drop if col in input_df.columns], inplace=True)

        if 'NumCompaniesWorked' in input_df.columns and 'TotalWorkingYears' in input_df.columns:
            input_df['YearsPerCompany'] = input_df['TotalWorkingYears'] / (input_df['NumCompaniesWorked'] + 1)

        categorical_cols = input_df.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dtype=float)

        # Reindexar para garantir que as colunas correspondem ao modelo
        final_df = input_df.reindex(columns=model_features, fill_value=0)
        
        # Extrair o modelo real do pipeline
        if hasattr(model, 'steps'):
            actual_model = model.named_steps['classifier']
        else:
            actual_model = model
            
        probability_yes = actual_model.predict_proba(final_df)[0, 1]
        prediction = "Yes" if probability_yes >= 0.5 else "No"

        # Gerar a explicação com SHAP
        shap_values = explainer.shap_values(final_df)
        explanation = dict(zip(final_df.columns, shap_values[0]))
        explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))

        return {
            "prediction": prediction,
            "probability_yes": float(probability_yes),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a predição: {str(e)}")
