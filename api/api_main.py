import os
import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from .schemas import EmployeeData, PredictionOut

app = FastAPI(
    title="API de Predição de Attrition de Funcionários",
    description="API para prever a probabilidade de um funcionário deixar a empresa.",
    version="1.0.0"
)

# Carregamento dos Artefactos de Produção
MODEL_PATH = os.path.join("models", "production_model.pkl")
FEATURES_PATH = os.path.join("artifacts", "features", "features.pkl")
EXPLAINER_PATH = os.path.join("models", "production_shap_explainer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    print(f"✅ Modelo, explicador SHAP e lista de features ({len(model_features)} colunas) carregados com sucesso.")
except FileNotFoundError as e:
    print(f"❌ Erro crítico ao carregar artefactos: {e}.")
    raise

@app.get("/health", summary="Verifica a saúde da API")
async def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut, summary="Realiza a predição de attrition")
async def predict(employee_data: EmployeeData):
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

        final_df = input_df.reindex(columns=model_features, fill_value=0)
        
        if hasattr(model, 'steps'):
            actual_model = model.named_steps['classifier']
        else:
            actual_model = model
            
        probability_yes = actual_model.predict_proba(final_df)[0, 1]
        prediction = "Yes" if probability_yes >= 0.5 else "No"

        shap_values = explainer.shap_values(final_df)
        
        # CORREÇÃO: Converter os valores SHAP (numpy.float32) para floats padrão do Python
        explanation_raw = dict(zip(final_df.columns, shap_values[0]))
        explanation = {k: float(v) for k, v in explanation_raw.items()}
        explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))

        return {
            "prediction": prediction,
            "probability_yes": float(probability_yes),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a predição: {str(e)}")