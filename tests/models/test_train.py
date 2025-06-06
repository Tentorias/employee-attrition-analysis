# tests/models/test_train.py

import pandas as pd
import pytest

from attrition.models.train import optimize_threshold, train_model


@pytest.fixture
def dummy_data():
    # Gera dataset binário
    X = pd.DataFrame({"feat": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    return X, y


def test_train_model_returns_classifier(dummy_data):
    X, y = dummy_data
    model = train_model(X, y, random_state=0)
    # model tem método predict_proba
    assert hasattr(model, "predict_proba")


def test_optimize_threshold_extremes(dummy_data):
    X, y = dummy_data
    model = train_model(X, y, random_state=0)
    thr, f1 = optimize_threshold(model, X, y)
    # threshold entre 0 e 1, F1 é float
    assert 0.0 <= thr <= 1.0
    assert isinstance(f1, float)
