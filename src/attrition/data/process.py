import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Sobe até a raiz do projeto (project_root)
BASE_DIR = Path(__file__).resolve().parents[3]


def load_raw(path: str) -> pd.DataFrame:
    """Carrega o CSV bruto de rotatividade (raw)."""
    return pd.read_csv(path)


def transform_logs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Aplica log-transform nas colunas especificadas."""
    for c in cols:
        df[f"{c}_log"] = np.log1p(df[c])
    return df


def cap_outliers(
    series: pd.Series, lower: float = 0.01, upper: float = 0.99
) -> pd.Series:
    """Limita outliers baseados nos quantis lower e upper."""
    q_low, q_high = series.quantile([lower, upper])
    return series.clip(q_low, q_high)


def drop_and_map(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas irrelevantes e mapeia variáveis categóricas binárias."""
    df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours"])
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    return df


def encode_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Aplica one-hot encoding nas colunas categóricas fornecidas."""
    return pd.get_dummies(df, columns=cols, drop_first=True)


def save_processed(df: pd.DataFrame, path: str):
    """Salva o DataFrame processado em CSV, criando diretório se necessário."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def main(raw_path: str, out_path: str):
    """
    Script principal para pré-processamento de dados.
    Recebe:
      - raw_path: caminho para o CSV bruto
      - out_path: caminho para salvar o CSV processado
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(f"🔄 Carregando dados de {raw_path}")
    df = load_raw(raw_path)

    # Lista de colunas numéricas e categóricas a serem processadas
    numeric_cols_to_log = ["MonthlyIncome", "TotalWorkingYears"]
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
        # Assumindo que queremos limitar os outliers nas colunas originais
        df[col] = cap_outliers(df[col])

    logging.info("🔧 Removendo colunas irrelevantes e mapeando binárias")
    df = drop_and_map(df)

    logging.info("🔧 Codificando variáveis categóricas")
    # Filtra para garantir que apenas as colunas presentes no DataFrame sejam codificadas
    cols_to_encode_present = [col for col in categorical_cols_to_encode if col in df.columns]
    df = encode_categoricals(df, cols_to_encode_present)

    logging.info(f"💾 Salvando dados processados em {out_path}")
    save_processed(df, out_path)
    logging.info("✅ Concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pré-processa o CSV bruto de Employee Attrition."
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=str(
            BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
        ),
        help="Caminho para o CSV bruto (raw).",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=str(
            BASE_DIR / "data" / "processed" / "employee_attrition_processed.csv"
        ),
        help="Onde salvar o CSV processado.",
    )
    args = parser.parse_args()
    main(raw_path=args.raw_path, out_path=args.out_path)
