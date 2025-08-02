# src/attrition/models/train.py
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import tunning

logging.basicConfig(level=logging.INFO, format="%(message)s")


def ensure_dir(file_path):
    """Garante que o diretório para um arquivo exista."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = df.copy()
    df_proc = df_proc.drop(
        columns=["EmployeeCount", "Over18", "StandardHours"], errors="ignore"
    )
    if "Gender" in df_proc.columns:
        df_proc["Gender"] = df_proc["Gender"].map({"Male": 1, "Female": 0})
    if (
        "TotalWorkingYears" in df_proc.columns
        and "NumCompaniesWorked" in df_proc.columns
    ):
        df_proc["YearsPerCompany"] = df_proc["TotalWorkingYears"] / df_proc[
            "NumCompaniesWorked"
        ].replace(0, 1)
    if "MonthlyIncome" in df_proc.columns:
        df_proc["MonthlyIncome_log"] = np.log1p(df_proc["MonthlyIncome"])
    if "TotalWorkingYears" in df_proc.columns:
        df_proc["TotalWorkingYears_log"] = np.log1p(df_proc["TotalWorkingYears"])
    cat_cols = df_proc.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df_proc = pd.get_dummies(
            df_proc, columns=cat_cols, drop_first=True, dtype=float
        )
    return df_proc


def main(
    raw_data_path,
    model_path,
    features_path,
    params_path,
    x_test_out,
    y_test_out,
    retrain_full_data=False,
    run_optuna_tuning=False,
):
    """Pipeline completo: carrega, divide, processa, otimiza (opcional) e treina."""
    logging.info(f"Carregando dados brutos de {raw_data_path}")
    df_raw = pd.read_csv(raw_data_path)
    df_raw["Attrition"] = df_raw["Attrition"].map({"Yes": 1, "No": 0})

    X = df_raw.drop("Attrition", axis=1)
    y = df_raw["Attrition"]

    # CORREÇÃO: DIVIDIR OS DADOS PRIMEIRO, ANTES DE QUALQUER PRÉ-PROCESSAMENTO
    logging.info("Dividindo os dados em treino e teste...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pré-processar os conjuntos de treino e teste separadamente
    logging.info("Aplicando pré-processamento...")
    X_train = preprocess(X_train_raw)
    X_test = preprocess(X_test_raw)

    # Aplicar o balanceamento apenas no conjunto de TREINO
    logging.info("Aplicando SMOTEENN nos dados de treino...")
    smoteenn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)

    if run_optuna_tuning:
        tunning.run_tuning(
            X_train=X_train_resampled,
            y_train=y_train_resampled,
            n_trials=100,
            output_path=params_path,
        )

    try:
        with open(params_path, "r") as f:
            best_params = json.load(f)
        logging.info(f"Usando melhores parâmetros de '{params_path}'")
    except FileNotFoundError:
        logging.warning("Arquivo de parâmetros não encontrado. Usando XGBoost padrão.")
        best_params = {}

    count_neg, count_pos = y_train_resampled.value_counts()
    scale_pos_weight_value = count_neg / count_pos

    classifier = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight_value,
        **best_params,
    )

    model = classifier

    if retrain_full_data:
        logging.info("Modo de produção: treinando com todos os dados resampleados...")

        # CORREÇÃO: Treinar o modelo de produção em todos os dados resampleados
        X_all_processed = preprocess(X)
        X_all_resampled, y_all_resampled = smoteenn.fit_resample(X_all_processed, y)
        model.fit(X_all_resampled, y_all_resampled)
    else:
        logging.info("Modo de avaliação: treinando com dados de treino resampleados...")
        model.fit(X_train_resampled, y_train_resampled)

    train_cols = X_train.columns.tolist()
    logging.info(f"Salvando lista de {len(train_cols)} features em {features_path}")
    ensure_dir(features_path)
    joblib.dump(train_cols, features_path)

    logging.info(f"Salvando modelo treinado em {model_path}")
    ensure_dir(model_path)
    joblib.dump(model, model_path)

    if not retrain_full_data:
        # AVALIAR NO CONJUNTO DE TESTE ORIGINAL
        X_test_final = X_test.reindex(columns=train_cols, fill_value=0)
        logging.info(f"Salvando dados de teste em {x_test_out} e {y_test_out}")
        ensure_dir(x_test_out)
        X_test_final.to_csv(x_test_out, index=False)
        y_test.to_csv(y_test_out, index=False)


if __name__ == "__main__":
    main()


# poetry run python src/attrition/main.py run-pipeline --tune
# poetry run python src/attrition/main.py run-pipeline --tune --min-precision-target 0.60
