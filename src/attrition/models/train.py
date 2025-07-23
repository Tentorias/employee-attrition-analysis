# src/attrition/models/train.py (Integrado com Optuna)
import argparse
import logging
import os
import joblib
import pandas as pd
import numpy as np
import json # <<< Adicionado
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
# <<< Adicionado para chamar o tunning.py >>>
from . import tunning

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ... (funções ensure_dir e preprocess permanecem as mesmas) ...
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = df.copy()
    df_proc = df_proc.drop(columns=["EmployeeCount", "Over18", "StandardHours"], errors='ignore')
    if 'Gender' in df_proc.columns:
        df_proc["Gender"] = df_proc["Gender"].map({"Male": 1, "Female": 0})
    if 'TotalWorkingYears' in df_proc.columns and 'NumCompaniesWorked' in df_proc.columns:
        df_proc["YearsPerCompany"] = df_proc["TotalWorkingYears"] / df_proc["NumCompaniesWorked"].replace(0, 1)
    if 'MonthlyIncome' in df_proc.columns:
        df_proc["MonthlyIncome_log"] = np.log1p(df_proc["MonthlyIncome"])
    if 'TotalWorkingYears' in df_proc.columns:
        df_proc["TotalWorkingYears_log"] = np.log1p(df_proc["TotalWorkingYears"])
    cat_cols = df_proc.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True, dtype=float)
    return df_proc

# <<< Função main totalmente refatorada >>>
def main(raw_data_path, model_path, features_path, params_path, x_test_out, y_test_out, retrain_full_data=False, run_optuna_tuning=False):
    """Pipeline completo: carrega, divide, processa, otimiza (opcional) e treina."""
    logging.info(f"Carregando dados brutos de {raw_data_path}")
    df_raw = pd.read_csv(raw_data_path)
    df_raw["Attrition"] = df_raw["Attrition"].map({"Yes": 1, "No": 0})
    
    X = df_raw.drop("Attrition", axis=1)
    y = df_raw["Attrition"]

    logging.info("Dividindo os dados em treino e teste (para pré-processamento)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logging.info("Aplicando pré-processamento...")
    X_train_processed = preprocess(X_train)
    
    # Otimização com Optuna (se ativada)
    if run_optuna_tuning:
        tunning.run_tuning(X_train=X_train_processed, y_train=y_train, n_trials=50, output_path=params_path) # 50 tentativas para ser mais rápido

    # Carrega os melhores parâmetros ou usa os padrões
    try:
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        logging.info(f"Usando melhores parâmetros de '{params_path}'")
    except FileNotFoundError:
        logging.warning("Arquivo de parâmetros não encontrado. Usando XGBoost padrão.")
        best_params = {}

    # Define o classificador com os melhores parâmetros
    classifier = XGBClassifier(random_state=42, n_jobs=-1, **best_params)
    
    # Define o pipeline final (sem SMOTE, pois o tunning já ajusta para o desbalanceamento)
    model = classifier

    # Decide se retreina com todos os dados para produção
    if retrain_full_data:
        logging.info("Modo de produção: processando e treinando com todos os dados...")
        X_processed = preprocess(X)
        model.fit(X_processed, y)
    else:
        logging.info("Modo de avaliação: treinando com dados de treino...")
        model.fit(X_train_processed, y_train)

    # Salva os artefatos
    train_cols = X_train_processed.columns.tolist()
    logging.info(f"Salvando lista de {len(train_cols)} features em {features_path}")
    ensure_dir(features_path)
    joblib.dump(train_cols, features_path)
    
    logging.info(f"Salvando modelo treinado em {model_path}")
    ensure_dir(model_path)
    joblib.dump(model, model_path)
    
    if not retrain_full_data:
        X_test_processed = preprocess(X_test).reindex(columns=train_cols, fill_value=0)
        logging.info(f"Salvando dados de teste em {x_test_out} e {y_test_out}")
        ensure_dir(x_test_out)
        X_test_processed.to_csv(x_test_out, index=False)
        y_test.to_csv(y_test_out, index=False)