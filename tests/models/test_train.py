import pandas as pd
import numpy as np
import pytest
from attrition.models.train import train_model, optimize_threshold

@pytest.fixture
def balanced_data():
    # gera dataset balanceado simples
    X = pd.DataFrame({'f1': range(10), 'f2': range(10,20)})
    y = pd.Series([0,1]*5)
    return X, y

def test_train_model_returns_classifier(balanced_data):
    X, y = balanced_data
    model = train_model(X, y, random_state=0)
    from xgboost import XGBClassifier
    assert isinstance(model, XGBClassifier)

def test_optimize_threshold_extremes(balanced_data):
    X, y = balanced_data
    model = train_model(X, y, random_state=0)
    thr, f1 = optimize_threshold(model, X, y)
    assert 0.3 <= thr <= 0.7
    assert 0 <= f1 <= 1
