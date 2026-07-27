# --- tests/test_unit_functions.py ---

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from attrition import preprocess_employee_data

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))


class LocalDummyModel:
    """Um modelo falso para testar a função de avaliação sem treinar um modelo real."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]] * len(X))


# --------- Teste de Unidade para a Nova Função `preprocess_employee_data` ---------


def test_preprocess_function_logic():
    """
    Testa todos os aspectos da função centralizada 'preprocess_employee_data'.
    """
    raw_data = {
        "EmployeeNumber": [101, 102],
        "Attrition": ["Yes", "No"],
        "EmployeeCount": [1, 1],
        "Over18": ["Y", "Y"],
        "StandardHours": [80, 80],
        "Gender": ["Male", "Female"],
        "TotalWorkingYears": [10, 5],
        "NumCompaniesWorked": [2, 0],
        "MonthlyIncome": [5000, 1000],
        "Department": ["Sales", "Research & Development"],
        "YearsAtCompany": [5, 2],
        "YearsSinceLastPromotion": [2, 0],
        "YearsWithCurrManager": [2, 1],
    }
    df_raw = pd.DataFrame(raw_data)

    df_processed = preprocess_employee_data(df_raw)

    # Verifica remoção de colunas administrativas, ID e Target
    assert "EmployeeCount" not in df_processed.columns
    assert "Over18" not in df_processed.columns
    assert "StandardHours" not in df_processed.columns
    assert "EmployeeNumber" not in df_processed.columns
    assert "Attrition" not in df_processed.columns

    # Verifica mapeamento numérico e tipo
    assert pd.api.types.is_numeric_dtype(df_processed["Gender"])
    assert df_processed["Gender"].iloc[0] == 1
    assert df_processed["Gender"].iloc[1] == 0

    # Verifica features de engenharia criadas corretamente
    assert "YearsPerCompany" in df_processed.columns
    assert "MonthlyIncome_log" in df_processed.columns
    assert "TotalWorkingYears_log" in df_processed.columns
    assert df_processed["YearsPerCompany"].iloc[0] == 5.0
    assert df_processed["YearsPerCompany"].iloc[1] == 5.0

    # Verifica features de interação (que antes faltavam no Streamlit/API)
    assert "Income_Longevity_Interaction" in df_processed.columns
    assert "Stagnation_Index" in df_processed.columns
    assert df_processed["Income_Longevity_Interaction"].iloc[0] == 25000.0  # 5000 * 5
    assert df_processed["Stagnation_Index"].iloc[0] == 1.0  # 2 / 2

    # Verifica One-Hot Encoding
    assert "Department_Sales" in df_processed.columns
    assert df_processed["Department_Sales"].iloc[0] == 1.0
    assert df_processed["Department_Sales"].iloc[1] == 0.0


# --------- Teste de Unidade para a função de avaliação (continua válido) ---------


def test_evaluate_model_basic():
    """Testa a função 'evaluate_model' de forma isolada."""
    model = LocalDummyModel()
    X_test = pd.DataFrame({"a": [1, 2]})
    y_test = pd.Series([0, 1])

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    report = classification_report(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    assert isinstance(report, str)
    assert cm.shape == (2, 2)
