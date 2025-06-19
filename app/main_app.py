# app/main_app.py

import os
import sys

import joblib
import pandas as pd
import streamlit as st

# Adiciona a pasta raiz ao path do Python para encontrar os módulos locais.
# É necessário que esta linha venha antes das importações de 'src' e 'ui_config'.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importa os dicionários de configuração e a função de predição
from ui_config import HELP_TEXTS, LABEL_MAPPING, VALUE_MAPPING  # noqa: E402

from src.attrition.models.predict import \
    main as predict_attrition  # noqa: E402

# --- Configuração da Página ---
st.set_page_config(
    page_title="Previsão de Attrition", page_icon="🤖", layout="centered"
)


# --- Funções de Carregamento e Predição ---
@st.cache_resource
def load_artifacts():
    """Carrega todos os artefatos necessários uma única vez."""
    try:
        model = joblib.load("models/production_model.pkl")
        threshold = joblib.load("artifacts/models/threshold_optimizado.pkl")
        processed_df = pd.read_csv("data/processed/employee_attrition_processed.csv")
        return model, threshold, processed_df
    except FileNotFoundError as e:
        st.error(
            f"Erro ao carregar artefatos: {e}. Certifique-se de que os caminhos "
            "estão corretos e o pipeline de treino foi executado."
        )
        st.stop()


def run_prediction(input_data_dict):
    """Chama o backend de predição com os dados brutos do formulário."""
    prediction, probability = predict_attrition(
        model_path="models/production_model.pkl",
        threshold_path="artifacts/models/threshold_optimizado.pkl",
        features_path="artifacts/features/features.pkl",
        input_data=input_data_dict,
    )
    return prediction, probability


# --- Lógica Principal do App ---
model, threshold, processed_df = load_artifacts()

st.title("Sistema de Análise de Risco de Attrition")
st.markdown(
    "Use as opções na barra lateral para inserir os dados de um funcionário e "
    "prever a probabilidade de ele deixar a empresa."
)

st.sidebar.header("Dados do Funcionário")
prediction_mode = st.sidebar.radio(
    "Escolha o modo de predição:",
    ("Rápida", "Avançada"),
    help="O modo 'Rápida' usa as features mais importantes. O modo 'Avançada' permite ajustar todos os fatores.",
)
input_data = {}

# --- Lógica dos Formulários Condicionais ---
if prediction_mode == "Rápida":
    st.sidebar.subheader("Predição Rápida")
    overtime_options = VALUE_MAPPING.get("OverTime", {})
    overtime_friendly = st.sidebar.selectbox(
        LABEL_MAPPING.get("OverTime"),
        list(overtime_options.values()),
        help=HELP_TEXTS.get("OverTime"),
    )
    input_data["OverTime"] = [
        k for k, v in overtime_options.items() if v == overtime_friendly
    ][0]

    input_data["MonthlyIncome"] = st.sidebar.slider(
        LABEL_MAPPING.get("MonthlyIncome"),
        1000,
        20000,
        5000,
        100,
        help=HELP_TEXTS.get("MonthlyIncome"),
    )
    input_data["JobLevel"] = st.sidebar.slider(
        LABEL_MAPPING.get("JobLevel"), 1, 5, 2, help=HELP_TEXTS.get("JobLevel")
    )
    input_data["TotalWorkingYears"] = st.sidebar.slider(
        LABEL_MAPPING.get("TotalWorkingYears"),
        0,
        40,
        10,
        help=HELP_TEXTS.get("TotalWorkingYears"),
    )
    input_data["JobSatisfaction"] = st.sidebar.slider(
        LABEL_MAPPING.get("JobSatisfaction"),
        1,
        4,
        3,
        help=HELP_TEXTS.get("JobSatisfaction"),
    )
    input_data["YearsAtCompany"] = st.sidebar.slider(
        LABEL_MAPPING.get("YearsAtCompany"),
        0,
        40,
        5,
        help=HELP_TEXTS.get("YearsAtCompany"),
    )

elif prediction_mode == "Avançada":
    st.sidebar.subheader("Predição Avançada")
    cols_to_drop = ["Attrition", "EmployeeCount", "StandardHours", "Over18"]
    for col in processed_df.drop(columns=cols_to_drop).columns:
        friendly_label = LABEL_MAPPING.get(col, col)
        help_text = HELP_TEXTS.get(col)

        if col in VALUE_MAPPING:
            options_map = VALUE_MAPPING.get(col, {})
            friendly_options = list(options_map.values())
            selected_friendly = st.sidebar.selectbox(
                friendly_label, friendly_options, help=help_text
            )
            input_data[col] = [
                k for k, v in options_map.items() if v == selected_friendly
            ][0]
        elif pd.api.types.is_numeric_dtype(processed_df[col]):
            min_val, max_val = int(processed_df[col].min()), int(
                processed_df[col].max()
            )
            default_val = int(processed_df[col].median())
            input_data[col] = st.sidebar.slider(
                friendly_label, min_val, max_val, default_val, help=help_text
            )

# --- Botão e Lógica de Predição ---
if st.sidebar.button("Fazer Predição", type="primary"):
    with st.spinner("Avaliando o perfil do funcionário..."):
        prediction, probability = run_prediction(input_data)

    if prediction is not None:
        st.header("Resultado da Análise de Risco")
        col1, col2 = st.columns([2, 3])
        with col1:
            if prediction == 1:
                st.error("**Alto Risco de Saída!**", icon="🚨")
            else:
                st.success("**Baixo Risco de Saída**", icon="✅")
            st.metric(label="Probabilidade de Attrition", value=f"{probability:.2%}")
            st.progress(float(probability))
        with col2:
            st.info(
                f"**Sobre a Predição:**\n\nO modelo indicou que este funcionário tem "
                f"**{probability:.0%}** de chance de deixar a empresa. A decisão "
                f"final ('Sai' ou 'Fica') é baseada em um threshold otimizado "
                f"de **{threshold:.2f}**.",
                icon="�",
            )
else:
    st.info(
        "Preencha os dados do funcionário na barra lateral e clique em "
        "'Fazer Predição'."
    )
