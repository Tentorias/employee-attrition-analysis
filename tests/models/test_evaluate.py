import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from attrition.models import evaluate


class LocalDummyModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]] * len(X))


def test_evaluate_model_basic():
    model = LocalDummyModel()
    X_test = pd.DataFrame({"a": [1, 2]})
    y_test = pd.Series([0, 1])
    f1, report, cm = evaluate.evaluate_model(model, X_test, y_test, threshold=0.5)
    assert isinstance(report, str)
    assert cm.shape == (2, 2)
    assert isinstance(f1, float)


def test_evaluate_model_with_threshold():
    model = LocalDummyModel()
    X_test = pd.DataFrame({"a": [1, 2]})
    y_test = pd.Series([0, 1])
    f1, report, cm = evaluate.evaluate_model(model, X_test, y_test, threshold=0.5)
    assert isinstance(report, str)
    assert cm.shape == (2, 2)
    assert isinstance(f1, float)


def test_main_cli(tmp_path):
    model_path = tmp_path / "model.pkl"
    threshold_path = tmp_path / "threshold.pkl" 
    x_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"

    model = DummyClassifier(strategy="constant", constant=1)
    model.fit([[0], [1]], [0, 1])

    joblib.dump(model, model_path)
    joblib.dump(0.5, threshold_path)
    pd.DataFrame({"a": [1, 2]}).to_csv(x_test_path, index=False)
    pd.Series([0, 1], name="target").to_csv(y_test_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")

    result = subprocess.run(
        [
            sys.executable,
            "src/attrition/models/evaluate.py",
            "--model-path",
            str(model_path),
            "--threshold-path",
            str(threshold_path),
            "--x-test-path",
            str(x_test_path),
            "--y-test-path",
            str(y_test_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"O script falhou com o erro: {result.stderr}"
    assert "Relatório de Avaliação" in result.stderr


def test_main_cli_help():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    result = subprocess.run(
        [sys.executable, "src/attrition/models/evaluate.py", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
