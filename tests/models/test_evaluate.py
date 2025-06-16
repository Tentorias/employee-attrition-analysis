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
        return np.tile([0.2, 0.8], (len(X), 1))


def test_evaluate_model_basic():
    model = LocalDummyModel()
    X_test = pd.DataFrame({"a": [1, 2]})
    y_test = pd.Series([0, 1])
    report, cm, f1 = evaluate.evaluate_model(model, X_test, y_test)
    assert isinstance(report, str)
    assert cm.shape == (2, 2)
    assert isinstance(f1, float)


def test_evaluate_model_with_threshold():
    model = LocalDummyModel()
    X_test = pd.DataFrame({"a": [1, 2]})
    y_test = pd.Series([0, 1])
    report, cm, f1 = evaluate.evaluate_model(model, X_test, y_test, threshold=0.5)
    assert isinstance(report, str)
    assert cm.shape == (2, 2)
    assert isinstance(f1, float)


def test_load_model(tmp_path):
    model = {"foo": "bar"}
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)
    loaded = evaluate.load_model(str(model_path))
    assert loaded == model


def test_load_features(tmp_path):
    features = ["a", "b"]
    features_path = tmp_path / "features.pkl"
    joblib.dump(features, features_path)
    loaded = evaluate.load_features(str(features_path))
    assert loaded == features


def test_main_cli(tmp_path):
    model_path = tmp_path / "model.pkl"
    features_path = tmp_path / "features.pkl"
    X_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"
    # Use DummyClassifier do sklearn para evitar problemas de pickle
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0], [1]], [0, 1])
    joblib.dump(model, model_path)
    joblib.dump(["a"], features_path)
    pd.DataFrame({"a": [1, 2]}).to_csv(X_test_path, index=False)
    pd.Series([0, 1], name="target").to_csv(y_test_path, index=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    result = subprocess.run(
        [
            sys.executable,
            "src/attrition/models/evaluate.py",
            "--model-path",
            str(model_path),
            "--features-path",
            str(features_path),
            "--X-test-path",
            str(X_test_path),
            "--y-test-path",
            str(y_test_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert (
        "F1-score" in result.stdout
        or "F1-score" in result.stderr
        or result.returncode == 0
    )


def test_main_legacy_cli(tmp_path):
    model_path = tmp_path / "model.pkl"
    test_data_path = tmp_path / "test.csv"
    # Use DummyClassifier do sklearn para evitar problemas de pickle
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0], [1]], [0, 1])
    joblib.dump(model, model_path)
    pd.DataFrame({"Attrition": [0, 1], "a": [1, 2]}).to_csv(test_data_path, index=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    result = subprocess.run(
        [
            sys.executable,
            "src/attrition/models/evaluate.py",
            "--model-path",
            str(model_path),
            "--test-data-path",
            str(test_data_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert (
        "F1-score" in result.stdout
        or "F1-score" in result.stderr
        or result.returncode == 0
    )


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
