# app/main_app.py

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# --- CORREÇÃO PARA O ModuleNotFoundError ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui_config import HELP_TEXTS, LABEL_MAPPING, VALUE_MAPPING
from src.attrition.models.predict import main as predict_attrition

# --- 1. Constantes e Configurações ---
MODEL_PATH = "models/production_model.pkl"
THRESHOLD_PATH = "artifacts/models/threshold_optimizado.pkl"
FEATURES_PATH = "artifacts/features/features.pkl"
PROCESSED_DATA_PATH = "data/processed/employee_attrition_processed.csv"

REVERSED_VALUE_MAPPING = {
    feature: {v: k for k, v in options.items()}
    for feature, options in VALUE_MAPPING.items()
}

# --- 2. Organização das Features para a Nova UI ---
# Agrupamos as features em categorias lógicas para as abas.
FEATURE_GROUPS = {
    "Informações Pessoais": [
        "Age", "Gender", "MaritalStatus", "DistanceFromHome"
    ],
    "Carreira e Cargo": [
        "Department", "JobRole", "JobLevel", "YearsAtCompany",
        "YearsInCurrentRole", "YearsWithCurrManager", "TotalWorkingYears",
        "NumCompaniesWorked", "YearsSinceLastPromotion", "TrainingTimesLastYear"
    ],
    "Remuneração": [
        "MonthlyIncome", "PercentSalaryHike", "StockOptionLevel"
    ],
    "Satisfação e Engajamento": [
        "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance", "OverTime",
        "PerformanceRating"
    ]
}


# --- Configuração da Página ---
st.set_page_config(
    page_title="Previsão de Attrition", page_icon="📈", layout="wide"
)

# --- Funções de Carregamento e Predição (sem alterações) ---
@st.cache_resource
def load_artifacts():
    """Carrega todos os artefatos necessários usando as constantes definidas."""
    try:
        model = joblib.load(MODEL_PATH)
        threshold = joblib.load(THRESHOLD_PATH)
        processed_df = pd.read_csv(PROCESSED_DATA_PATH)
        return model, threshold, processed_df
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar artefatos: {e}. Certifique-se de que os caminhos estão corretos.")
        st.stop()

def run_prediction(input_data_dict):
    """Chama o backend de predição com os dados brutos do formulário."""
    return predict_attrition(
        model_path=MODEL_PATH,
        threshold_path=THRESHOLD_PATH,
        features_path=FEATURES_PATH,
        input_data=input_data_dict,
    )

def generate_form_widgets(container, features_to_display: list, df_reference: pd.DataFrame):
    """Gera dinamicamente os widgets dentro de um container (como uma aba)."""
    input_data = {}
    for col in features_to_display:
        if col not in df_reference.columns:
            continue
        
        friendly_label = LABEL_MAPPING.get(col, col)
        help_text = HELP_TEXTS.get(col)

        if col in VALUE_MAPPING:
            options_map = VALUE_MAPPING.get(col, {})
            friendly_options = list(options_map.values())
            selected_friendly = container.selectbox(
                friendly_label, friendly_options, help=help_text, key=f"sb_{col}"
            )
            input_data[col] = REVERSED_VALUE_MAPPING.get(col, {}).get(selected_friendly)

        elif pd.api.types.is_numeric_dtype(df_reference[col]):
            min_val, max_val = int(df_reference[col].min()), int(df_reference[col].max())
            default_val = int(df_reference[col].median())
            step = 100 if "Income" in col else 1
            input_data[col] = container.slider(
                friendly_label, min_val, max_val, default_val, step, help=help_text, key=f"sl_{col}"
            )
    return input_data

# --- Lógica Principal da UI ---
model, threshold, processed_df = load_artifacts()

st.title("📈 Sistema de Análise de Risco de Attrition")
st.markdown(
    "Preencha as informações do funcionário nas abas abaixo para obter uma predição sobre o risco de saída."
)

input_data = {}
tabs = st.tabs(list(FEATURE_GROUPS.keys()))

for i, group_name in enumerate(FEATURE_GROUPS.keys()):
    with tabs[i]:
        # Usamos update para combinar os dicionários de cada aba
        input_data.update(
            generate_form_widgets(st.container(), FEATURE_GROUPS[group_name], processed_df)
        )

# --- Botão e Lógica de Predição ---
st.write("") # Adiciona um espaço antes do botão
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    if st.button("Fazer Predição", type="primary", use_container_width=True):
        with st.spinner("Avaliando o perfil do funcionário..."):
            prediction, probability = run_prediction(input_data)

        if prediction is not None:
            st.header("Resultado da Análise")
            
            # --- MUDANÇA: Separando a exibição do resultado e da probabilidade ---
            if prediction == 1:
                st.error("**Alto Risco de Saída!**", icon="🚨")
            else:
                st.success("**Baixo Risco de Saída**", icon="✅")

            st.metric(
                label="Probabilidade de Saída",
                value=f"{probability:.2%}"
            )
            st.progress(float(probability))
            # --- FIM DA MUDANÇA ---

            help_message = (
                "Este é o valor de corte usado para classificar a probabilidade. "
                f"Se a probabilidade for maior que {threshold:.2f}, o funcionário é "
                "considerado de 'Alto Risco'. Este valor foi otimizado para "
                "identificar o máximo de talentos em risco."
            )
            st.metric(
                label="Threshold de Decisão do Modelo",
                value=f"{threshold:.2f}",
                help=help_message,
            )
            
        else:
            st.error("Não foi possível realizar a predição.")
    else:
        st.info("Após preencher os dados, clique em 'Fazer Predição'.")
