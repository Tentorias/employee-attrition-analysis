import os
import subprocess
import sys

import joblib
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from attrition.models import explain


def test_load_features(tmp_path):
    features = ["a", "b"]
    features_path = tmp_path / "features.pkl"
    joblib.dump(features, features_path)
    loaded = explain.load_features(str(features_path))
    assert loaded == features


def test_main_runs(tmp_path):
    # Use pelo menos duas features para evitar erro do SHAP
    model = DecisionTreeClassifier()
    X = [[0, 1], [1, 0]]
    y = [0, 1]
    model.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    features = ["a", "b"]
    features_path = tmp_path / "features.pkl"
    joblib.dump(features, features_path)
    df = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    output_path = tmp_path / "shap_plot.png"
    explain.main(
        model_path=str(model_path),
        data_path=str(data_path),
        features_path=str(features_path),
        output_path=str(output_path),
    )
    assert output_path.exists()


def test_main_missing_column(tmp_path):
    model = DecisionTreeClassifier()
    X = [[0, 1], [1, 0]]
    y = [0, 1]
    model.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    features = ["a", "b"]
    features_path = tmp_path / "features.pkl"
    joblib.dump(features, features_path)
    # Só a coluna "a" presente, "b" está faltando
    df = pd.DataFrame({"a": [0, 1]})
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    with pytest.raises(ValueError):
        explain.main(
            model_path=str(model_path),
            data_path=str(data_path),
            features_path=str(features_path),
        )


def test_cli_help():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    result = subprocess.run(
        [sys.executable, "src/attrition/models/explain.py", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
