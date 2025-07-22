# src/attrition/data/process.py (CORRIGIDO)

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

<<<<<<< HEAD
=======

>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76
BASE_DIR = Path(__file__).resolve().parents[3]

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def transform_logs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[f"{c}_log"] = np.log1p(df[c])
    return df

def cap_outliers(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    q_low, q_high = series.quantile([lower, upper])
    return series.clip(q_low, q_high)

def drop_and_map(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours"])
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    return df

def encode_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, drop_first=True, dtype=float) # Alterado para dtype=float para consistência

def save_processed(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main(raw_path: str, out_path: str):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(f"🔄 Carregando dados de {raw_path}")
    df = load_raw(raw_path)

<<<<<<< HEAD
=======
    
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76
    numeric_cols_to_log = ["MonthlyIncome", "TotalWorkingYears"]
    
    # <<< LISTA CORRIGIDA AQUI >>>
    categorical_cols_to_encode = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "JobRole",
        "MaritalStatus",
        "OverTime",
    ]

    logging.info("🔧 Aplicando transformações de log")
    df = transform_logs(df, numeric_cols_to_log)

    logging.info("🔧 Limitando outliers")
    for col in numeric_cols_to_log:
<<<<<<< HEAD
=======
        
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76
        df[col] = cap_outliers(df[col])

    logging.info("🔧 Removendo colunas irrelevantes e mapeando binárias")
    df = drop_and_map(df)

    logging.info("🔧 Codificando variáveis categóricas")
<<<<<<< HEAD
=======
   
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76
    cols_to_encode_present = [col for col in categorical_cols_to_encode if col in df.columns]
    df = encode_categoricals(df, cols_to_encode_present)

    logging.info(f"💾 Salvando dados processados em {out_path}")
    save_processed(df, out_path)
    logging.info("✅ Concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-processa o CSV bruto de Employee Attrition.")
    parser.add_argument("--raw-path", type=str, default=str(BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"))
    parser.add_argument("--out-path", type=str, default=str(BASE_DIR / "data" / "processed" / "employee_attrition_processed.csv"))
    args = parser.parse_args()
    main(raw_path=args.raw_path, out_path=args.out_path)