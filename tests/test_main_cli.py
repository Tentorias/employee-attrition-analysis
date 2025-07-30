# tests/test_main_cli.py
import os
import subprocess
import sys

import pandas as pd


def run_cli(args):
    """Função helper para executar o main.py como um processo externo."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    return subprocess.run(
        [sys.executable, "src/attrition/main.py"] + args,
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_help():
    """Testa se o comando --help funciona."""
    result = run_cli(["--help"])
    assert result.returncode == 0
    assert "usage" in result.stdout


def test_cli_invalid_command():
    """Testa se um comando inválido retorna um erro."""
    result = run_cli(["comando_invalido"])
    assert result.returncode != 0


def test_cli_run_pipeline_success(tmp_path):
    """
    Testa o comando 'run-pipeline' de ponta a ponta, verificando se
    todos os artefatos são criados em um diretório temporário.
    """

    raw_data_path = tmp_path / "raw_data.csv"
    sample_data = {
        "Attrition": ["No", "Yes"] * 20,
        "Gender": ["Male", "Female"] * 20,
        "TotalWorkingYears": range(40),
        "NumCompaniesWorked": range(40),
        "MonthlyIncome": range(1000, 1040),
        "Department": ["Sales", "R&D"] * 20,
    }
    pd.DataFrame(sample_data).to_csv(raw_data_path, index=False)

    model_path = tmp_path / "model.pkl"
    features_path = tmp_path / "features.pkl"
    params_path = tmp_path / "params.json"
    x_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"
    prod_model_path = tmp_path / "prod_model.pkl"

    result = run_cli(
        [
            "run-pipeline",
            "--raw-data-path",
            str(raw_data_path),
            "--model-path",
            str(model_path),
            "--features-path",
            str(features_path),
            "--params-path",
            str(params_path),
            "--x-test-path",
            str(x_test_path),
            "--y-test-path",
            str(y_test_path),
            "--prod-model-path",
            str(prod_model_path),
        ]
    )

    assert result.returncode == 0, f"O pipeline falhou com o erro: {result.stderr}"
    assert model_path.exists()
    assert features_path.exists()
    assert x_test_path.exists()
    assert y_test_path.exists()
    assert prod_model_path.exists()
