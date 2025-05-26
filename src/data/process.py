# src/data/process.py

import os
import argparse
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def transform_logs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[f"{c}_log"] = np.log1p(df[c])
    return df

# … outras funções …

def save_processed(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Pré-processa o CSV bruto de Employee Attrition."
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
        help="Caminho para o CSV bruto (raw)."
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=os.path.join(BASE_DIR, "data", "processed", "employee_attrition_processed.csv"),
        help="Onde salvar o CSV processado."
    )
    args = parser.parse_args()

    logger.info(f"🔄 Carregando dados de {args.raw_path}")
    df = load_raw(args.raw_path)

    logger.info("🔧 Aplicando transformações de log")
    df = transform_logs(df, ["MonthlyIncome", "TotalWorkingYears"])

    # … cap_outliers, drop_and_map, encode_categoricals …

    logger.info(f"💾 Salvando dados processados em {args.out_path}")
    save_processed(df, args.out_path)
    logger.info("✅ Concluído.")

if __name__ == "__main__":
    main()
