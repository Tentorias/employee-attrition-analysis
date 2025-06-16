import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from attrition.data.process import (cap_outliers, drop_and_map,
                                    encode_categoricals, load_raw,
                                    save_processed, transform_logs)


@pytest.fixture
def sample_raw(tmp_path):
    df = pd.DataFrame(
        {
            "MonthlyIncome": [1000, 2000, 0],
            "TotalWorkingYears": [1, 4, 0],
        }
    )
    file = tmp_path / "raw.csv"
    df.to_csv(file, index=False)
    return str(file)


def test_load_raw(sample_raw):
    df = load_raw(sample_raw)
    assert "MonthlyIncome" in df.columns


def test_transform_logs():
    df = pd.DataFrame({"MonthlyIncome": [0, 10], "TotalWorkingYears": [0, 4]})
    df2 = transform_logs(df.copy(), ["MonthlyIncome", "TotalWorkingYears"])
    assert "MonthlyIncome_log" in df2.columns
    assert pytest.approx(df2["MonthlyIncome_log"].iloc[1]) == np.log1p(10)


def test_save_and_reload(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3]})
    out = tmp_path / "out.csv"
    save_processed(df, str(out))
    df2 = pd.read_csv(out)
    pd.testing.assert_frame_equal(df, df2)


def test_cap_outliers_basic():
    s = pd.Series([1, 2, 3, 100])
    capped = cap_outliers(s, lower=0.25, upper=0.75)
    assert capped.max() <= s.quantile(0.75)
    assert capped.min() >= s.quantile(0.25)


def test_drop_and_map_basic():
    df = pd.DataFrame(
        {
            "EmployeeCount": [1, 1],
            "Over18": ["Y", "Y"],
            "StandardHours": [40, 40],
            "Attrition": ["Yes", "No"],
            "Gender": ["Male", "Female"],
        }
    )
    df2 = drop_and_map(df)
    assert "EmployeeCount" not in df2.columns
    assert "Over18" not in df2.columns
    assert "StandardHours" not in df2.columns
    assert set(df2["Attrition"].unique()).issubset({0, 1})
    assert set(df2["Gender"].unique()).issubset({0, 1})


def test_encode_categoricals_basic():
    df = pd.DataFrame({"Department": ["HR", "IT", "HR"], "Value": [1, 2, 3]})
    df2 = encode_categoricals(df, ["Department"])
    assert any("Department_" in col for col in df2.columns)


def test_process_cli(tmp_path):
    in_file = tmp_path / "in.csv"
    out_file = tmp_path / "out.csv"
    pd.DataFrame({"MonthlyIncome": [1000], "TotalWorkingYears": [1]}).to_csv(
        in_file, index=False
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    result = subprocess.run(
        [
            sys.executable,
            "src/attrition/data/process.py",
            "--raw-path",
            str(in_file),
            "--out-path",
            str(out_file),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out_file.exists()
    assert result.returncode == 0


def test_main_with_all_steps(tmp_path):
    df = pd.DataFrame(
        {
            "MonthlyIncome": [1000, 2000, 0],
            "TotalWorkingYears": [1, 4, 0],
            "EmployeeCount": [1, 1, 1],
            "Over18": ["Y", "Y", "Y"],
            "StandardHours": [40, 40, 40],
            "Attrition": ["Yes", "No", "No"],
            "Gender": ["Male", "Female", "Male"],
            "Department": ["HR", "IT", "HR"],
            "EducationField": ["Field1", "Field2", "Field1"],
        }
    )
    out_file = tmp_path / "out.csv"
    df = transform_logs(df, ["MonthlyIncome", "TotalWorkingYears"])
    df["MonthlyIncome"] = cap_outliers(df["MonthlyIncome"])
    df["TotalWorkingYears"] = cap_outliers(df["TotalWorkingYears"])
    df = drop_and_map(df)
    df = encode_categoricals(df, ["Department", "EducationField"])
    save_processed(df, str(out_file))
    assert out_file.exists()
