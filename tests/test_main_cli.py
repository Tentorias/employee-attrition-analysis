import os
import subprocess
import sys

import pandas as pd


def run_cli(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    return subprocess.run(
        [sys.executable, "src/attrition/main.py"] + args,
        capture_output=True,
        text=True,
        env=env,
    )


def test_main_help():
    result = run_cli(["--help"])
    assert "usage" in result.stdout or "usage" in result.stderr


def test_main_process(tmp_path):
    in_file = tmp_path / "in.csv"
    out_file = tmp_path / "out.csv"
    pd.DataFrame({"MonthlyIncome": [1000], "TotalWorkingYears": [1]}).to_csv(
        in_file, index=False
    )
    result = run_cli(
        ["process", "--raw-path", str(in_file), "--out-path", str(out_file)]
    )
    assert out_file.exists()
    assert result.returncode == 0


def test_main_process(tmp_path):
    in_file = tmp_path / "in.csv"
    out_file = tmp_path / "out.csv"

    # DataFrame de teste completo, simulando o dado bruto real
    sample_data = {
        "MonthlyIncome": [1000],
        "TotalWorkingYears": [1],
        "EmployeeCount": [1],
        "Over18": ["Y"],
        "StandardHours": [80],
        "Attrition": ["No"],
        "Gender": ["Male"],
        "BusinessTravel": ["Travel_Rarely"],
        "Department": ["Sales"],
        "EducationField": ["Marketing"],
        "JobRole": ["Sales Executive"],
        "MaritalStatus": ["Single"],
        "OverTime": ["No"],
    }
    pd.DataFrame(sample_data).to_csv(in_file, index=False)

    result = run_cli(
        ["process", "--raw-path", str(in_file), "--out-path", str(out_file)]
    )

    assert result.returncode == 0, f"O script falhou com o erro: {result.stderr}"
    assert out_file.exists()


def test_main_train_shows_error(tmp_path):
    result = run_cli(["train"])
    assert result.returncode != 0
    assert "usage" in result.stderr


def test_main_evaluate_shows_error(tmp_path):
    result = run_cli(["evaluate"])
    assert result.returncode != 0
    assert "usage" in result.stderr


def test_main_explain_shows_error(tmp_path):
    result = run_cli(["explain"])
    assert result.returncode != 0
    assert "usage" in result.stderr


def test_main_no_command_shows_help():
    result = run_cli([])
    assert result.returncode == 0 or result.returncode == 1
    assert "usage" in result.stdout or "usage" in result.stderr


def test_main_invalid_command():
    result = run_cli(["invalidcmd"])
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "usage" in result.stderr
