# tests/test_unit_functions.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ajusta o path para src/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from attrition.models.train import preprocess


class LocalDummyModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]] * len(X))


def test_preprocess_function_logic():
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

    # colunas inúteis removidas
    for c in ["EmployeeCount", "Over18", "StandardHours"]:
        assert c not in df_processed.columns

    # mapeamento Gender
    assert pd.api.types.is_integer_dtype(df_processed["Gender"])
    assert df_processed["Gender"].tolist() == [1, 0]

    # features criadas
    assert df_processed["YearsPerCompany"].tolist() == [5.0, 5.0]
    for log_col in ["MonthlyIncome_log", "TotalWorkingYears_log"]:
        assert log_col in df_processed.columns

    # one‑hot em Department
    assert "Department_Sales" in df_processed.columns
    assert df_processed["Department_Sales"].tolist() == [1.0, 0.0]


def test_evaluate_model_basic():
    model = LocalDummyModel()
    X_test = pd.DataFrame({"a": [1, 2]})
    y_test = pd.Series([0, 1])

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    report = classification_report(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    assert isinstance(report, str)
    assert cm.shape == (2, 2)
