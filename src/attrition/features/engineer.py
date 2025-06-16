import argparse
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Ajusta BASE_DIR para apontar para a raiz do projeto (project_root)
BASE_DIR = Path(__file__).resolve().parents[3]


def load_processed(path: str) -> pd.DataFrame:
    """Carrega o CSV processado em 'data/processed'."""
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Cria a variável derivada YearsPerCompany
    2) Aplica one-hot encoding em TODOS os campos categóricos (tipo object),
       para que nenhuma coluna do tipo 'object' permaneça.
    """
    # 1) variável derivada
    df["YearsPerCompany"] = df["TotalWorkingYears"] / df["NumCompaniesWorked"].replace(
        0, 1
    )

    # 2) identificar colunas do tipo object
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if cat_cols:
        logger.info(f"🔧 Codificando colunas categóricas: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def save_features(df: pd.DataFrame, path: str):
    """Salva o DataFrame com features em CSV e retorna o próprio DataFrame."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main(in_path: str, out_path: str):
    """
    Script principal para gerar a matriz de features.
    Recebe:
      - in_path: caminho para o CSV processado de entrada
      - out_path: caminho para salvar o CSV com features
    """
    logger.info(f"🔄 Carregando dados processados de {in_path}")
    df = load_processed(in_path)

    logger.info("🔧 Criando features derivadas e codificando categóricas")
    df_feat = engineer_features(df)

    logger.info(f"💾 Salvando matriz de features em {out_path}")
    save_features(df_feat, out_path)
    logger.info("✅ Concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera matriz de features a partir do CSV processado."
    )
    parser.add_argument(
        "--in-path",
        type=str,
        default=str(
            BASE_DIR / "data" / "processed" / "employee_attrition_processed.csv"
        ),
        help="CSV processado de entrada.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=str(BASE_DIR / "artifacts" / "features" / "features_matrix.csv"),
        help="Onde salvar a matriz de features.",
    )
    args = parser.parse_args()
    main(in_path=args.in_path, out_path=args.out_path)
