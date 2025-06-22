import os
import subprocess
import sys

import pandas as pd

from attrition.features.engineer import (engineer_features, load_processed,
                                         save_features)


def test_engineer_features_all_types():
    df = pd.DataFrame(
        {
            "TotalWorkingYears": [1, 2, 0],
            "NumCompaniesWorked": [1, 2, 0],
            "Department": ["HR", "IT", "Sales"],
            "JobRole": ["Manager", "Staff", "Staff"],
            "Attrition": [1, 0, 1],  # Coluna adicionada
        }
    )
    df_feat = engineer_features(df)
    assert "YearsPerCompany" in df_feat.columns
    assert any("Department_" in col for col in df_feat.columns)
    assert any("JobRole_" in col for col in df_feat.columns)


def test_engineer_features_no_object():
    df = pd.DataFrame(
        {
            "TotalWorkingYears": [1, 2],
            "NumCompaniesWorked": [1, 2],
            "Attrition": [0, 1],  # Coluna adicionada
        }
    )
    df_feat = engineer_features(df)
    assert "YearsPerCompany" in df_feat.columns 


def test_load_and_save_features(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    out_file = tmp_path / "features.csv"
    features_list_file = tmp_path / "features.pkl" 
    save_features(df, str(out_file), str(features_list_file)) 
    df2 = load_processed(str(out_file))
    pd.testing.assert_frame_equal(df, df2)
    assert features_list_file.exists() 


def test_engineer_cli(tmp_path):
    in_file = tmp_path / "in.csv"
    out_file = tmp_path / "out.csv"
    features_out_file = tmp_path / "features.pkl"
    df = pd.DataFrame(
        {
            "TotalWorkingYears": [1, 2],
            "NumCompaniesWorked": [1, 2],
            "Department": ["HR", "IT"],
            "Attrition": [0, 1], # Coluna adicionada
        }
    )
    df.to_csv(in_file, index=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    result = subprocess.run(
        [
            sys.executable,
            "src/attrition/features/engineer.py",
            "--input-path",
            str(in_file),
            "--output-path",
            str(out_file),
            "--features-out-path", # Argumento adicionado
            str(features_out_file),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"O script falhou com o erro: {result.stderr}"
    assert out_file.exists()
    assert features_out_file.exists()
