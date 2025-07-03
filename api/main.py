# api/main.py
# -*- coding: utf-8 -*-

"""
Módulo principal da API de Predição de Attrition de Funcionários.

Este módulo usa FastAPI para criar os endpoints, carrega o modelo de Machine Learning
e o explicador SHAP na inicialização, e define a lógica para
processar os dados, fazer predições e retornar explicações.
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

# ----------------- CARREGAMENTO DOS ARTEFATOS DE PRODUÇÃO ----------------- #

# Caminhos corretos para os artefatos de produção
# Caminhos relativos ao projeto, que funcionarão no Render
MODEL_PATH = os.path.join("models", "production_model.pkl")
FEATURES_PATH = os.path.join("models", "features.pkl") # Ou o caminho correto que decidimos
EXPLAINER_PATH = os.path.join("models", "production_shap_explainer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    print(f"✅ Modelo, explicador SHAP e lista de features ({len(model_features)} colunas) carregados com sucesso.")
except FileNotFoundError as e:
    print(f"❌ Erro crítico ao carregar artefatos: {e}.")
    print("Certifique-se de que todos os artefatos de produção existem.")
    raise

# ----------------- ENDPOINTS DA API ----------------- #

@app.get("/health", summary="Verifica a saúde da API")
async def health_check():
    """
    Endpoint de health check. Retorna um status 'ok' se a API estiver
    operacional. Útil para monitoramento.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut, summary="Realiza a predição de attrition")
async def predict(employee_data: EmployeeData):
    """
    Recebe os dados de um funcionário e realiza a predição de attrition,
    replicando exatamente o pipeline de engenharia de features.
    """
    try:
        input_df = pd.DataFrame([employee_data.dict()])

        # 1. Replicar a engenharia de features do script 'engineer.py'
        # Remover colunas constantes que não são usadas no treinamento
        cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18']
        input_df.drop(columns=[col for col in cols_to_drop if col in input_df.columns], inplace=True)

        input_df['YearsPerCompany'] = input_df['TotalWorkingYears'] / (input_df['NumCompaniesWorked'] + 1)

        # Identificar TODAS as colunas categóricas para aplicar One-Hot Encoding
        categorical_cols = input_df.select_dtypes(include=["object"]).columns.tolist()
        
        if categorical_cols:
            input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dtype=float)

        # 2. Reindexar para garantir que as colunas correspondem ao modelo
        final_df = input_df.reindex(columns=model_features, fill_value=0)
        
        # 3. Fazer a predição
        # O modelo real está dentro do pipeline do imblearn
        if hasattr(model, 'steps'):
            actual_model = model.named_steps['classifier']
        else:
            actual_model = model
            
        probability_yes = actual_model.predict_proba(final_df)[0, 1]
        prediction = "Yes" if probability_yes >= 0.5 else "No"

        # 4. Gerar a explicação com SHAP
        shap_values = explainer.shap_values(final_df)
        explanation = dict(zip(final_df.columns, shap_values[0]))
        explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))

        # 5. Retornar o resultado completo
        return {
            "prediction": prediction,
            "probability_yes": float(probability_yes),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a predição: {str(e)}")
