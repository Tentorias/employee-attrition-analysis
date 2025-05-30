# src/features/engineer.py

import os
import argparse
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

def load_processed(path: str) -> pd.DataFrame:
    """
    Carrega o CSV já processado em `data/processed`.
    """
    return pd.read_csv(path)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica engenharia de features:
    - Cria variável derivada YearsPerCompany
    - Executa one-hot encoding em todas as colunas categóricas
    """
    # Variável derivada
    df["YearsPerCompany"] = (
        df["TotalWorkingYears"] / df["NumCompaniesWorked"].replace(0, 1)
    )
    # Identifica colunas categóricas (object)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        logger.info(f"🔧 Codificando colunas categóricas: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def save_features(df: pd.DataFrame, path: str):
    """
    Salva a matriz de features em CSV, criando diretório se necessário.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Gera matriz de features a partir do CSV processado."
    )
    parser.add_argument(
        "--in-path",
        type=str,
        default=os.path.join(
            BASE_DIR, "data", "processed", "employee_attrition_processed.csv"
        ),
        help="CSV processado de entrada."
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=os.path.join(
            BASE_DIR, "artifacts", "features_matrix.csv"
        ),
        help="Onde salvar a matriz de features."
    )
    args = parser.parse_args()

    logger.info(f"🔄 Carregando dados processados de {args.in_path}")
    df = load_processed(args.in_path)

    logger.info("🔧 Criando features derivadas e codificando categóricas")
    df_feat = engineer_features(df)

    logger.info(f"💾 Salvando matriz de features em {args.out_path}")
    save_features(df_feat, args.out_path)
    logger.info("✅ Concluído.")

if __name__ == "__main__":
    main()
