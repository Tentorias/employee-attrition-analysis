# tests/models/test_train.py

import os
import subprocess
import sys

import joblib
import pandas as pd
from attrition.models.train import main


def test_main_function(tmp_path):
    df = pd.DataFrame({
        "a": [0, 1, 0, 1, 1, 0],
        "b": [1, 0, 1, 0, 1, 1],
        "t": [0, 1, 0, 1, 0, 1]
    })
    in_csv = tmp_path / "in.csv"
    df.to_csv(in_csv, index=False)
    feat_pkl = tmp_path / "feat.pkl"
    joblib.dump(["a", "b"], feat_pkl)
    model_pkl = tmp_path / "model.pkl"
    thr_pkl = tmp_path / "thr.pkl"

    main(
        in_path=str(in_csv),
        features_path=str(feat_pkl),
        model_path=str(model_pkl),
        threshold_path=str(thr_pkl),
        target_col="t",
        test_size=0.5,
        random_state=0,
    )
    assert model_pkl.exists()
    assert thr_pkl.exists()
    _ = joblib.load(model_pkl)

def test_cli_train_subprocess(tmp_path):
    df = pd.DataFrame({
        "a": [0, 1, 0, 1, 1, 0],
        "b": [1, 0, 1, 0, 1, 1],
        "t": [0, 1, 0, 1, 0, 1]
    })
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
            "--in-path", str(in_csv),
            "--features-path", str(feat_pkl),
            "--model-path", str(model_pkl),
            "--threshold-path", str(thr_pkl),
            "--target-col", "t",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"O script falhou com o erro: {result.stderr}"
    assert model_pkl.exists()
    assert thr_pkl.exists()


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