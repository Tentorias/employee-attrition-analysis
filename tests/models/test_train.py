import os
import subprocess
import sys

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from attrition.models.train import (main, optimize_threshold, parse_args,
                                    train_model)


def test_train_model_returns_classifier():
    X = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])
    model = train_model(X, y)
    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert len(preds) == len(y)


# 2. Test fallback SMOTE in train_model


def test_train_model_smote_fallback():
    X = pd.DataFrame({"a": [1], "b": [2]})
    y = pd.Series([0])
    model = train_model(X, y)
    assert hasattr(model, "predict")
    assert model.predict(X).tolist() == [0]


# 3. Test optimize_threshold branches


def test_optimize_threshold_extremes():
    X = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])
    model = DecisionTreeClassifier()
    model.fit(X, y)
    thr, f1 = optimize_threshold(model, X, y)
    assert 0 <= thr <= 1
    assert 0 <= f1 <= 1


def test_optimize_threshold_no_proba():
    class Dummy:
        def predict(self, X):
            return [0] * len(X)

    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 0])
    thr, f1 = optimize_threshold(Dummy(), X, y)
    assert thr == 0.5 and isinstance(f1, float)


# 4. Test parse_args correctness


def test_parse_args_defaults_and_types():
    args = parse_args(
        [
            "--in-path",
            "in.csv",
            "--features-path",
            "feat.pkl",
            "--model-path",
            "mod.pkl",
        ]
    )
    assert args.in_path == "in.csv"
    assert args.features_path == "feat.pkl"
    assert args.model_path == "mod.pkl"
    assert args.threshold_path is None
    assert args.target_col == "Attrition"
    assert isinstance(args.test_size, float)
    assert isinstance(args.random_state, int)


# 5. Test main() in-process


def test_main_function(tmp_path):
    df = pd.DataFrame({"a": [0, 1], "b": [1, 0], "t": [0, 1]})
    in_csv = tmp_path / "in.csv"
    df.to_csv(in_csv, index=False)
    feat_pkl = tmp_path / "feat.pkl"
    joblib.dump(["a", "b"], feat_pkl)
    model_pkl = tmp_path / "model.pkl"
    thr_pkl = tmp_path / "thr.pkl"

    main(
        str(in_csv),
        str(feat_pkl),
        str(model_pkl),
        str(thr_pkl),
        target_col="t",
        test_size=0.5,
        random_state=0,
    )
    assert model_pkl.exists()
    assert thr_pkl.exists()
    # loading should not error
    _ = joblib.load(model_pkl)


# 6. Test CLI via subprocess


def test_cli_train_subprocess(tmp_path):
    df = pd.DataFrame({"a": [0, 1], "b": [1, 0], "t": [0, 1]})
    in_csv = tmp_path / "in.csv"
    df.to_csv(in_csv, index=False)
    feat_pkl = tmp_path / "feat.pkl"
    joblib.dump(["a", "b"], feat_pkl)
    model_pkl = tmp_path / "model.pkl"
    thr_pkl = tmp_path / "thr.pkl"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    result = subprocess.run(
        [
            sys.executable,
            "src/attrition/models/train.py",
            "--in-path",
            str(in_csv),
            "--features-path",
            str(feat_pkl),
            "--model-path",
            str(model_pkl),
            "--threshold-path",
            str(thr_pkl),
            "--target-col",
            "t",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "Treino concluído" in result.stdout
    assert model_pkl.exists()
    assert thr_pkl.exists()


# 7. Test help


def test_cli_help():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    result = subprocess.run(
        [sys.executable, "src/attrition/models/train.py", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
