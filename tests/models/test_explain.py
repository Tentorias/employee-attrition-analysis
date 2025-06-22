import os
import subprocess
import sys

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def run_cli(args):
    """Função helper para rodar o script explain.py via CLI."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    return subprocess.run(
        [sys.executable, "src/attrition/models/explain.py"] + args,
        capture_output=True,
        text=True,
        env=env,
    )


def test_main_runs_successfully(tmp_path):
    model = DecisionTreeClassifier()
    X_train = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    y_train = pd.Series([0, 1])
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    X_test = pd.DataFrame({"a": [1, 0], "b": [0, 1]})
    x_test_path = tmp_path / "x_test.csv"
    X_test.to_csv(x_test_path, index=False)

    output_path = tmp_path / "shap_plot.png"

    result = run_cli([
        "--model-path", str(model_path),
        "--x-test-path", str(x_test_path),
        "--output-path", str(output_path)
    ])

    assert result.returncode == 0, f"O script falhou com o erro: {result.stderr}"
    assert "Gráfico SHAP salvo com sucesso" in result.stderr
    assert output_path.exists()


def test_main_fails_with_missing_file(tmp_path):
    result = run_cli([
        "--model-path", str(tmp_path / "missing_model.pkl"),
        "--x-test-path", str(tmp_path / "missing_data.csv")
    ])
    assert result.returncode != 0
    assert "Arquivo não encontrado" in result.stderr


def test_cli_help():
    result = run_cli(["--help"])
    assert result.returncode == 0
    assert "usage: explain.py" in result.stdout
