# src/attrition/models/train.py (VERSÃO FINAL CORRIGIDA)
import argparse
import logging
import os
import joblib
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica as transformações de pré-processamento e engenharia de features.
    """
    df_proc = df.copy()
    # 1. Mapeamento e remoção de colunas
    df_proc = df_proc.drop(columns=["EmployeeCount", "Over18", "StandardHours"], errors='ignore')
    if 'Gender' in df_proc.columns:
        df_proc["Gender"] = df_proc["Gender"].map({"Male": 1, "Female": 0})
    
    # 2. Engenharia de Features
    if 'TotalWorkingYears' in df_proc.columns and 'NumCompaniesWorked' in df_proc.columns:
        df_proc["YearsPerCompany"] = df_proc["TotalWorkingYears"] / df_proc["NumCompaniesWorked"].replace(0, 1)
    if 'MonthlyIncome' in df_proc.columns:
        df_proc["MonthlyIncome_log"] = np.log1p(df_proc["MonthlyIncome"])
    if 'TotalWorkingYears' in df_proc.columns:
        df_proc["TotalWorkingYears_log"] = np.log1p(df_proc["TotalWorkingYears"])
    
    # 3. One-Hot Encoding
    cat_cols = df_proc.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True, dtype=float)
        
    return df_proc

# <<< MUDANÇA AQUI: Adicionado 'retrain_full_data' como argumento
def main(raw_data_path, model_path, features_path, x_test_out, y_test_out, retrain_full_data=False):
    """
    Pipeline completo e corrigido: carrega, divide, processa, treina e salva.
    """
    logging.info(f"Carregando dados brutos de {raw_data_path}")
    df_raw = pd.read_csv(raw_data_path)
    df_raw["Attrition"] = df_raw["Attrition"].map({"Yes": 1, "No": 0})
    
    X = df_raw.drop("Attrition", axis=1)
    y = df_raw["Attrition"]

    # <<< MUDANÇA AQUI: Lógica para decidir se divide os dados ou usa o dataset completo
    if retrain_full_data:
        logging.info("Modo de produção: usando todos os dados para treino.")
        X_train, y_train = X, y
        X_test, y_test = None, None # Não teremos dados de teste neste modo
    else:
        logging.info("Modo de avaliação: dividindo os dados em treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logging.info("Aplicando pré-processamento...")
    X_train_processed = preprocess(X_train)
    
    # Salva a lista de features baseada nos dados de treino
    train_cols = X_train_processed.columns.tolist()
    logging.info(f"Salvando lista de {len(train_cols)} features em {features_path}")
    ensure_dir(features_path)
    joblib.dump(train_cols, features_path)
    
    logging.info("Treinando o modelo com SMOTE e XGBoost...")
    model = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("classifier", XGBClassifier(random_state=42, n_jobs=-1))
    ])
    model.fit(X_train_processed, y_train)

    logging.info(f"Salvando modelo treinado em {model_path}")
    ensure_dir(model_path)
    joblib.dump(model, model_path)
    
    # Salva os dados de teste somente se estiver no modo de avaliação
    if not retrain_full_data and X_test is not None:
        X_test_processed = preprocess(X_test)
        X_test_processed = X_test_processed.reindex(columns=train_cols, fill_value=0)
        
        logging.info(f"Salvando dados de teste em {x_test_out} e {y_test_out}")
        ensure_dir(x_test_out)
        X_test_processed.to_csv(x_test_out, index=False)
        y_test.to_csv(y_test_out, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", default="data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    parser.add_argument("--model-path", default="artifacts/models/model.pkl")
    parser.add_argument("--features-path", default="artifacts/features/features.pkl")
    parser.add_argument("--x-test-out", default="artifacts/features/X_test.csv")
    parser.add_argument("--y-test-out", default="artifacts/features/y_test.csv")
    parser.add_argument("--retrain-full-data", action="store_true")
    args = parser.parse_args()
    main(**vars(args))