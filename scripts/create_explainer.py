# scripts/create_explainer.py

import logging
from pathlib import Path

import joblib
import pandas as pd
import shap

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_and_save_explainer():
    """
    Carrega o modelo de produção e os dados de teste para criar e salvar
    o objeto SHAP Explainer necessário para a aplicação Streamlit.
    """
    project_root = Path(__file__).resolve().parent.parent

    model_path = project_root / "models" / "production_model.pkl"
    x_test_path = project_root / "artifacts" / "features" / "X_test.csv"
    explainer_output_path = project_root / "models" / "production_shap_explainer.pkl"

    try:
        logging.info(f"Carregando modelo de produção de '{model_path}'...")
        model = joblib.load(model_path)

        logging.info(f"Carregando dados de teste de '{x_test_path}'...")
        X_test = pd.read_csv(x_test_path)

        actual_model = (
            model.named_steps["classifier"] if hasattr(model, "steps") else model
        )

        logging.info("Criando o objeto SHAP Explainer...")
        explainer = shap.Explainer(actual_model, X_test)

        logging.info(f"Salvando o explainer em '{explainer_output_path}'...")
        joblib.dump(explainer, explainer_output_path)

        logging.info("✅ Objeto SHAP Explainer criado e salvo com sucesso!")

    except FileNotFoundError as e:
        logging.error(
            f"Erro: Arquivo não encontrado. Certifique-se de que o pipeline de treino foi executado primeiro. Detalhes: {e}"
        )
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado: {e}")


if __name__ == "__main__":
    generate_and_save_explainer()
