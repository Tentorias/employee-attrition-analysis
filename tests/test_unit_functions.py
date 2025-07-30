# --- tests/test_unit_functions.py ---

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from attrition.models.train import preprocess

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# --- tests/test_unit_functions.py ---


class LocalDummyModel:
    """Um modelo falso para testar a função de avaliação sem treinar um modelo real."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]] * len(X))


# --------- Teste de Unidade para a Nova Função `preprocess` ---------


def test_preprocess_function_logic():
    """
    Testa todos os aspectos da nova função centralizada 'preprocess'.
    """
    raw_data = {
        "EmployeeCount": [1, 1],
        "Over18": ["Y", "Y"],
        "StandardHours": [80, 80],
        "Gender": ["Male", "Female"],
        "TotalWorkingYears": [10, 5],
        "NumCompaniesWorked": [2, 0],
        "MonthlyIncome": [5000, 1000],
        "Department": ["Sales", "Research & Development"],
    }
    df_raw = pd.DataFrame(raw_data)

    df_processed = preprocess(df_raw)

    assert "EmployeeCount" not in df_processed.columns
    assert "Over18" not in df_processed.columns
    assert "StandardHours" not in df_processed.columns

    assert pd.api.types.is_integer_dtype(df_processed["Gender"])
    assert df_processed["Gender"].iloc[0] == 1
    assert df_processed["Gender"].iloc[1] == 0

    assert "YearsPerCompany" in df_processed.columns
    assert "MonthlyIncome_log" in df_processed.columns
    assert "TotalWorkingYears_log" in df_processed.columns
    assert df_processed["YearsPerCompany"].iloc[0] == 5.0
    assert df_processed["YearsPerCompany"].iloc[1] == 5.0

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
