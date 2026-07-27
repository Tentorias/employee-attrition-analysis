# src/attrition/features.py
"""
Módulo centralizado para engenharia de features e pré-processamento de dados.
Funciona como 'Single Source of Truth' para treino, avaliação, Streamlit e API.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def preprocess_employee_data(
    df: pd.DataFrame, model_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aplica todas as transformações de limpeza, engenharia de features e encoding
    nos dados de funcionários.

    Args:
        df (pd.DataFrame): DataFrame bruto de funcionários.
        model_features (list, optional): Lista de colunas/features esperadas pelo
            modelo treinado. Se fornecido, reindexa o DataFrame resultante
            para corresponder exatamente a essas colunas.

    Returns:
        pd.DataFrame: DataFrame processado e pronto para modelagem/inferência.
    """
    if df.empty:
        if model_features is not None:
            return pd.DataFrame(columns=model_features)
        return pd.DataFrame()

    df_proc = df.copy()

    # 1. Remover colunas administrativas e não-preditivas (incluindo ID e Target se presentes)
    cols_to_drop = [
        "EmployeeCount",
        "Over18",
        "StandardHours",
        "EmployeeNumber",
        "Attrition",
    ]
    df_proc = df_proc.drop(
        columns=[col for col in cols_to_drop if col in df_proc.columns],
        errors="ignore",
    )

    # 2. Mapeamento numérico simples
    if "Gender" in df_proc.columns:
        df_proc["Gender"] = df_proc["Gender"].map({"Male": 1, "Female": 0})

    # 3. Transformações e Feature Engineering
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

    if "MonthlyIncome" in df_proc.columns and "YearsAtCompany" in df_proc.columns:
        df_proc["Income_Longevity_Interaction"] = (
            df_proc["MonthlyIncome"] * df_proc["YearsAtCompany"]
        )

    if (
        "YearsSinceLastPromotion" in df_proc.columns
        and "YearsWithCurrManager" in df_proc.columns
    ):
        df_proc["Stagnation_Index"] = df_proc["YearsSinceLastPromotion"] / df_proc[
            "YearsWithCurrManager"
        ].replace(0, 1)

    # 4. One-Hot Encoding de variáveis categóricas
    cat_cols = (
        df_proc.select_dtypes(include=["object", "string", "category"])
        .columns.tolist()
    )
    if cat_cols:
        df_proc = pd.get_dummies(
            df_proc, columns=cat_cols, drop_first=True, dtype=float
        )

    # 5. Converter booleanos em floats (compatibilidade universal com XGBoost)
    for col in df_proc.columns:
        if df_proc[col].dtype == "bool":
            df_proc[col] = df_proc[col].astype(float)

    # 6. Reindexação para alinhar exatamente às features do modelo (se fornecidas)
    if model_features is not None:
        df_proc = df_proc.reindex(columns=model_features, fill_value=0.0)

    return df_proc
