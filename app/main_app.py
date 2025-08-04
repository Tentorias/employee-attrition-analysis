# app/main_app.py

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import streamlit as st
from dotenv import load_dotenv

from src.attrition.data_processing import load_and_preprocess_data

try:
    from app.ui_config import LABEL_MAPPING, UNACTIONABLE_FEATURES, VALUE_MAPPING
except ImportError:
    LABEL_MAPPING, VALUE_MAPPING, UNACTIONABLE_FEATURES = {}, {}, []
    st.warning("Arquivo ui_config.py não encontrado. Usando configurações padrão.")


st.set_page_config(
    page_title="Diagnóstico de Turnover",
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_dotenv()

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# --- CONSTANTES ---

MODEL_PATH = project_root / "models" / "production_model.pkl"
FEATURES_PATH = project_root / "artifacts" / "features" / "features.pkl"
SHAP_EXPLAINER_PATH = project_root / "models" / "production_shap_explainer.pkl"

# --- FUNÇÕES DE APOIO ---


@st.cache_resource
def load_model_artifacts():
    """Carrega os artefatos de ML (modelo, features, explicador SHAP)."""
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        explainer = joblib.load(SHAP_EXPLAINER_PATH)
        return model, features, explainer
    except FileNotFoundError:
        st.error(
            "Erro: Arquivos de modelo (.pkl) não encontrados. "
            "Execute o pipeline de treinamento."  # E501
        )
        return None, None, None
    except Exception as e:
        st.error(f"Erro ao carregar artefatos do modelo: {e}")
        return None, None, None


def get_top_factors(shap_values, features, top_n=5):
    """Extrai os principais fatores de risco a partir dos valores SHAP."""
    shap_map = dict(zip(features, shap_values))
    risk_factors = {k: v for k, v in shap_map.items() if v > 0}
    return sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)[:top_n]


def translate_feature_name(feature_name):
    """Traduz nomes de features de forma robusta para exibição."""
    if feature_name in LABEL_MAPPING:
        return LABEL_MAPPING[feature_name]

    for base_feature, mappings in VALUE_MAPPING.items():
        prefix = f"{base_feature}_"
        if feature_name.startswith(prefix):
            value = feature_name[len(prefix) :]
            base_label = LABEL_MAPPING.get(base_feature, base_feature)
            value_label = next(
                (v_label for v_key, v_label in mappings.items() if str(v_key) == value),
                value,
            )
            return f"{base_label}: {value_label}"

    return feature_name.replace("_", " ")


# --- LÓGICA PRINCIPAL DO APP ---

st.title("💡 Ferramenta Tática de Análise de Turnover")


model, model_features, explainer = load_model_artifacts()

if not all([model, model_features, explainer]):
    st.error("Aplicação não pode continuar. Verifique os artefatos do modelo.")
    st.stop()


# --- DADOS PRÉ-PROCESSADOS, RETORNANDO DOIS DFS ---
@st.cache_data(ttl=3600)
def get_preprocessed_employee_data_and_ui(features_list_from_model):
    """Carrega e pré-processa os dados, retornando DF para modelo e DF para UI."""
    return load_and_preprocess_data(model_features_list=features_list_from_model)


df_model_ready, df_for_ui = get_preprocessed_employee_data_and_ui(model_features)

if df_model_ready.empty or df_for_ui.empty:
    st.error(
        "Erro ao carregar ou pré-processar os dados. "
        "Verifique a conexão com o banco ou o caminho do CSV."
    )
    st.stop()

# --- CONVERTER EMPLOYEE_NUMBER PARA INT EM AMBOS OS DFS E VERIFICAR COLUNA ---
if (
    "EmployeeNumber" not in df_model_ready.columns
    or "EmployeeNumber" not in df_for_ui.columns
):
    st.error(
        "Coluna 'EmployeeNumber' não encontrada em um dos DataFrames "
        "após pré-processamento. Verifique data_processing.py."
    )
    st.stop()

df_model_ready["EmployeeNumber"] = df_model_ready["EmployeeNumber"].astype(int)
df_for_ui["EmployeeNumber"] = df_for_ui["EmployeeNumber"].astype(int)


# --- ADICIONA A PROBABILIDADE PREDITA COM TRATAMENTO DE ERRO ---
try:
    df_for_ui["predicted_probability"] = model.predict_proba(df_model_ready)[:, 1]
    # --- VERIFICAÇÃO DE PROBABILIDADES EM TEMPO REAL NO STREAMLIT ---
    st.sidebar.subheader("Verificação de Probabilidades")
    st.sidebar.write("Probabilidades calculadas (primeiras 5):")
    st.sidebar.write(df_for_ui["predicted_probability"].head())
    st.sidebar.write("Estatísticas das probabilidades calculadas:")
    st.sidebar.write(df_for_ui["predicted_probability"].describe())

except Exception as e:
    st.error(
        f"Erro ao calcular probabilidades de predição: {e}. "
        "Verifique a compatibilidade do modelo com o DataFrame processado."
    )
    st.stop()

# --- VERIFICAÇÃO ADICIONAL APÓS CÁLCULO ---
if "predicted_probability" not in df_for_ui.columns:
    st.error(
        "Coluna 'predicted_probability' não foi adicionada a df_for_ui. "
        "Verifique o cálculo de predição."
    )
    st.stop()


# --- AJUSTE NA INICIALIZAÇÃO DE selected_employee_id ---
if not df_for_ui["EmployeeNumber"].empty:
    if "selected_employee_id" not in st.session_state:
        st.session_state.selected_employee_id = int(df_for_ui["EmployeeNumber"].iloc[0])
else:
    st.error(
        "Coluna 'EmployeeNumber' vazia em df_for_ui. "
        "Não é possível selecionar funcionário."
    )
    st.stop()

tab_team, tab_individual = st.tabs(
    [
        "👥 Análise de Risco da Equipe",
        "👤 Diagnóstico Individual",
    ]
)

# --- ABA 1: Risco da Equipe ---
with tab_team:
    st.header("Visão Preditiva de Risco por Departamento")
    departments = sorted(df_for_ui["Department"].unique())
    selected_dept = st.selectbox("Selecione um Departamento:", departments)

    team_df = df_for_ui[df_for_ui["Department"] == selected_dept]

    if team_df.empty:
        st.info(f"Nenhum funcionário encontrado no departamento '{selected_dept}'.")
        emp_options = {}
    else:
        team_sorted = team_df.sort_values(by="predicted_probability", ascending=False)
        emp_options = {
            f"{row['JobRole']} (ID: {row['EmployeeNumber']})": row["EmployeeNumber"]
            for _, row in team_sorted.iterrows()
        }

    if emp_options:
        selected_emp_key = st.selectbox(
            "Selecione um funcionário para análise:", options=emp_options.keys()
        )
        if st.button("Analisar Funcionário", use_container_width=True, type="primary"):
            st.session_state.selected_employee_id = int(emp_options[selected_emp_key])
            st.success(
                f"Funcionário {st.session_state.selected_employee_id} carregado! "
                "Verifique a aba de Diagnóstico."
            )

    if not team_df.empty:
        team_display = team_sorted.copy()
        team_display["risk_percent"] = team_display["predicted_probability"] * 100
        st.dataframe(
            team_display[["EmployeeNumber", "JobRole", "risk_percent"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "EmployeeNumber": "ID",
                "JobRole": "Cargo",
                "risk_percent": st.column_config.ProgressColumn(
                    "Risco de Saída", format="%.1f%%", min_value=0, max_value=100
                ),
            },
        )
    else:
        st.info("Nenhum dado para exibir para o departamento selecionado.")

# --- DEBUGGING PRINTS ---
st.sidebar.subheader("Debug Info")
st.sidebar.write(f"selected_employee_id: {st.session_state.selected_employee_id}")
st.sidebar.write(
    f"type of selected_employee_id: {type(st.session_state.selected_employee_id)}"
)
st.sidebar.write(
    f"df_model_ready['EmployeeNumber'] dtype: {df_model_ready['EmployeeNumber'].dtype}"
)

id_exists_in_model_df = (
    st.session_state.selected_employee_id in df_model_ready["EmployeeNumber"].values
)
st.sidebar.write(
    f"Is selected_employee_id in df_model_ready['EmployeeNumber'] values? "
    f"{id_exists_in_model_df}"
)

# --- FIM DOS DEBUGGING PRINTS ---


filtered_df_model = df_model_ready[
    df_model_ready["EmployeeNumber"] == st.session_state.selected_employee_id
]

if not filtered_df_model.empty:
    employee_data_model = filtered_df_model.iloc[0]
else:
    st.error(
        f"Erro: Funcionário com ID {st.session_state.selected_employee_id} não encontrado "
        "em df_model_ready para cálculo SHAP. Isso não deveria acontecer. "
        "Verifique a integridade dos dados e o pipeline de pré-processamento para EmployeeNumber."
    )
    st.stop()


filtered_df_ui = df_for_ui[
    df_for_ui["EmployeeNumber"] == st.session_state.selected_employee_id
]
if not filtered_df_ui.empty:
    employee_data_ui = filtered_df_ui.iloc[0]
else:
    st.error(
        f"Erro: Funcionário com ID {st.session_state.selected_employee_id} não encontrado "
        "em df_for_ui para exibição. Verifique a integridade dos dados."
    )
    st.stop()


# --- ABA 2: Diagnóstico Individual ---
with tab_individual:
    st.header(f"Diagnóstico para o Funcionário: {employee_data_ui['EmployeeNumber']}")
    st.metric(
        "Risco Atual de Saída", f"{employee_data_ui['predicted_probability']:.1%}"
    )

    X_emp = employee_data_model.to_frame().T
    shap_values = explainer.shap_values(X_emp)[0]
    top_factors = get_top_factors(shap_values, X_emp.columns)

    actionable_factors = [
        (feat, val)
        for feat, val in top_factors
        if feat.split("_")[0] not in UNACTIONABLE_FEATURES
    ]

    st.subheader("Principais Fatores de Risco (Acionáveis)")
    if actionable_factors:
        for feat, val in actionable_factors:
            st.markdown(f"- **{translate_feature_name(feat)}**")
    else:
        st.info("Nenhum fator de risco acionável proeminente foi identificado.")

    st.markdown("---")
    st.subheader("💡 Recomendações de Ação de RH")
    recommendations = []

    causal_effect_overtime = 0.29
    causal_effect_satisfaction = -0.02

    if "OverTime_Yes" in employee_data_ui and employee_data_ui["OverTime_Yes"] == 1:
        recommendations.append(
            f"**Horas Extras (Impacto Causal: +{causal_effect_overtime:.1%}):** "
            "Este funcionário realiza horas extras. Considere avaliar e reduzir a carga de trabalho, "
            "ou oferecer folgas compensatórias para mitigar o risco de atrito."
        )

    if (
        "high_job_satisfaction" in employee_data_ui
        and employee_data_ui["high_job_satisfaction"] == 0
    ):
        recommendations.append(
            f"**Baixa Satisfação no Trabalho (Impacto Causal: {causal_effect_satisfaction:.1%}):** "
            "A satisfação no trabalho deste funcionário é baixa. Dialogue para identificar e abordar "
            "pontos de insatisfação (ex: ambiente, reconhecimento, desenvolvimento de carreira)."
        )

    st.subheader("Análise Aprofundada do Modelo (Insights Adicionais)")
    with st.expander("Clique para ver um insight sobre 'Horas Extras'"):
        st.info(
            """
            #### Insight Contraintuitivo sobre Horas Extras

            Nossa análise de sanidade mostrou que, para este modelo, a **falta de horas extras**
            pode estar associada a um risco maior quando combinada com outros fatores, como baixo salário.

            **Hipótese:** Isso pode indicar um funcionário **desengajado**, que não busca compensação
            adicional e pode já estar procurando outras oportunidades. É um padrão mais complexo que
            o modelo foi capaz de aprender, indo além da simples ideia de que "hora extra = burnout".
            """
        )

    if recommendations:
        for i, rec in enumerate(recommendations):
            st.markdown(f"{i+1}. {rec}")
    else:
        st.info(
            "Nenhuma recomendação específica gerada para este funcionário com base nos fatores causais atuais."
        )
