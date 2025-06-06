# tests/data/test_process.py

import numpy as np
import pandas as pd
import pytest

from attrition.data.process import load_raw, save_processed, transform_logs


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
