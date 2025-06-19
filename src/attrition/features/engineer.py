# src/attrition/features/engineer.py (VERSÃO FINAL E CORRIGIDA)

import argparse
import logging
import os
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]


def load_processed(path: str) -> pd.DataFrame:
    """Carrega o CSV processado."""
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas e aplica one-hot encoding."""
    # 1. Cria a variável derivada
    df["YearsPerCompany"] = df["TotalWorkingYears"] / df["NumCompaniesWorked"].replace(
        0, 1
    )

    # --- MUDANÇA PRINCIPAL AQUI ---
    # 2. Deixamos o get_dummies cuidar da coluna 'Attrition' também
    # Ele vai criar a coluna 'Attrition_Yes' automaticamente
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Adicionamos 'Attrition' à lista se ainda não estiver (ela pode ser object ou category)
    if "Attrition" not in cat_cols:
        cat_cols.append("Attrition")

    logger.info(f"🔧 Codificando colunas categóricas: {cat_cols}")
    # Usamos dtype=float para garantir que as novas colunas sejam numéricas
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    return df_encoded


def save_features(df: pd.DataFrame, matrix_path: str, features_list_path: str):
    """Salva o DataFrame com features e a lista de colunas."""
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    df.to_csv(matrix_path, index=False)

    # Salva a lista de colunas de FEATURES (sem a coluna alvo)
    features_to_save = df.drop(
        columns=["Attrition_Yes"], errors="ignore"
    ).columns.tolist()
    joblib.dump(features_to_save, features_list_path)
    logger.info(f"💾 Lista de features salva em {features_list_path}")


def main(input_path: str, output_path: str, features_out_path: str):
    """Script principal para gerar a matriz de features."""
    logger.info(f"🔄 Carregando dados processados de {input_path}")
    df = load_processed(input_path)

    logger.info("🔧 Criando features derivadas e codificando categóricas")
    df_feat = engineer_features(df)

    logger.info(f"💾 Salvando matriz de features em {output_path}")
    save_features(
        df_feat, matrix_path=output_path, features_list_path=features_out_path
    )
    logger.info("✅ Concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera matriz de features a partir do CSV processado."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=str(
            BASE_DIR / "data" / "processed" / "employee_attrition_processed.csv"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(BASE_DIR / "artifacts" / "features" / "features_matrix.csv"),
    )
    parser.add_argument(
        "--features-out-path",
        type=str,
        default=str(BASE_DIR / "artifacts" / "features" / "features.pkl"),
    )

    args = parser.parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        features_out_path=args.features_out_path,
    )
