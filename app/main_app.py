import sys
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

# --- CONFIGURAÇÕES INICIAIS ---
st.set_page_config(
    page_title="Diagnóstico de Turnover",
    layout="wide",
    initial_sidebar_state="collapsed"
)
load_dotenv()

# Adiciona o caminho raiz do projeto para permitir importações locais
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Tenta importar configurações de UI, com fallback gracioso
try:
    from app.ui_config import LABEL_MAPPING, VALUE_MAPPING, UNACTIONABLE_FEATURES
except ImportError:
    LABEL_MAPPING, VALUE_MAPPING, UNACTIONABLE_FEATURES = {}, {}, []
    st.warning("Arquivo ui_config.py não encontrado. Usando configurações padrão.")

# --- CONSTANTES ---
MODEL_PATH = project_root / "models" / "production_model.pkl"
FEATURES_PATH = project_root / "artifacts" / "features" / "features.pkl"
SHAP_EXPLAINER_PATH = project_root / "models" / "production_shap_explainer.pkl"
DATABASE_URL = os.getenv("DATABASE_URL")

# --- FUNÇÕES DE APOIO ---

@st.cache_data(ttl=3600)  # Cache de 1 hora
def load_employee_data():
    """Carrega os dados brutos dos funcionários do banco de dados PostgreSQL."""
    if not DATABASE_URL:
        st.error("A URL do banco de dados (DATABASE_URL) não foi configurada.")
        return pd.DataFrame()
    try:
        engine = create_engine(DATABASE_URL)
        df_emp = pd.read_sql_query("SELECT * FROM employees", engine)
        return df_emp
    except Exception as e:
        st.error(f"Erro ao carregar dados dos funcionários: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model_artifacts():
    """Carrega os artefatos de ML (modelo, features, explicador SHAP)."""
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        explainer = joblib.load(SHAP_EXPLAINER_PATH)
        return model, features, explainer
    except FileNotFoundError:
        st.error("Erro: Arquivos de modelo (.pkl) não encontrados. Execute o pipeline de treinamento.")
        return None, None, None
    except Exception as e:
        st.error(f"Erro ao carregar artefatos do modelo: {e}")
        return None, None, None

def prepare_data_for_model(df, features):
    """Prepara o DataFrame para ser compatível com o modelo."""
    df_proc = df.copy()
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18']
    df_proc.drop(columns=[col for col in cols_to_drop if col in df_proc.columns], errors='ignore', inplace=True)
    if 'TotalWorkingYears' in df_proc.columns and 'NumCompaniesWorked' in df_proc.columns:
        df_proc['YearsPerCompany'] = df_proc.apply(
            lambda row: row['TotalWorkingYears'] / row['NumCompaniesWorked'] if row['NumCompaniesWorked'] > 0 else row['TotalWorkingYears'], 
            axis=1
        ).round(4)
    df_proc = pd.get_dummies(df_proc, drop_first=True, dtype=float)
    X = df_proc.reindex(columns=features, fill_value=0)
    return X

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
            value = feature_name[len(prefix):]
            base_label = LABEL_MAPPING.get(base_feature, base_feature)
            value_label = next((v_label for v_key, v_label in mappings.items() if str(v_key) == value), value)
            return f"{base_label}: {value_label}"
            
    return feature_name.replace('_', ' ')

# --- LÓGICA PRINCIPAL DO APP ---

st.title("💡 Ferramenta Tática de Análise de Turnover")

# Carrega dados e modelos
df_employees = load_employee_data()
model, model_features, explainer = load_model_artifacts()

if df_employees.empty or not all([model, model_features, explainer]):
    st.error("Aplicação não pode continuar. Verifique os dados e os artefatos do modelo.")
else:
    # Calcula as predições para todos os funcionários de uma vez
    X_all = prepare_data_for_model(df_employees.copy(), model_features)
    df_employees['predicted_probability'] = model.predict_proba(X_all)[:, 1]

    # Inicializa o estado da sessão
    if 'selected_employee_id' not in st.session_state:
        st.session_state.selected_employee_id = df_employees.iloc[0]['EmployeeNumber']

    # Define as abas da interface
    tab_team, tab_individual, tab_simulator = st.tabs([
        "👥 Risco da Equipe", 
        "👤 Diagnóstico Individual", 
        "🧪 Simulador What-If"
    ])

    # --- ABA 1: Risco da Equipe ---
    with tab_team:
        st.header("Visão Preditiva de Risco por Departamento")
        departments = sorted(df_employees['Department'].unique())
        selected_dept = st.selectbox("Selecione um Departamento:", departments)

        team_df = df_employees[df_employees['Department'] == selected_dept]
        team_sorted = team_df.sort_values(by='predicted_probability', ascending=False)
        
        emp_options = {f"{row['JobRole']} (ID: {row['EmployeeNumber']})": row['EmployeeNumber'] for _, row in team_sorted.iterrows()}

        if emp_options:
            selected_emp_key = st.selectbox("Selecione um funcionário para análise:", options=emp_options.keys())
            if st.button("Analisar Funcionário", use_container_width=True, type="primary"):
                st.session_state.selected_employee_id = emp_options[selected_emp_key]
                st.success(f"Funcionário {st.session_state.selected_employee_id} carregado! Veja as outras abas.")
        
        team_display = team_sorted.copy()
        team_display['risk_percent'] = team_display['predicted_probability'] * 100
        st.dataframe(
            team_display[['EmployeeNumber', 'JobRole', 'risk_percent']], 
            use_container_width=True, 
            hide_index=True, 
            column_config={
                "EmployeeNumber": "ID", 
                "JobRole": "Cargo", 
                "risk_percent": st.column_config.ProgressColumn(
                    "Risco de Saída", format="%.1f%%", min_value=0, max_value=100
                )
            }
        )

    # Puxa os dados do funcionário selecionado para as outras abas
    employee_data = df_employees[df_employees['EmployeeNumber'] == st.session_state.selected_employee_id].iloc[0]

    # --- ABA 2: Diagnóstico Individual ---
    with tab_individual:
        st.header(f"Diagnóstico para o Funcionário: {employee_data['EmployeeNumber']}")
        st.metric("Risco Atual de Saída", f"{employee_data['predicted_probability']:.1%}")

        # Calcula e exibe os fatores de risco
        X_emp = X_all[X_all.index == employee_data.name]
        shap_values = explainer.shap_values(X_emp)[0]
        top_factors = get_top_factors(shap_values, X_emp.columns)

        actionable_factors = [
            (feat, val) for feat, val in top_factors 
            if feat.split('_')[0] not in UNACTIONABLE_FEATURES
        ]
        
        st.subheader("Principais Fatores de Risco (Acionáveis)")
        if actionable_factors:
            for feat, val in actionable_factors:
                st.markdown(f"- **{translate_feature_name(feat)}**")
        else:
            st.info("Nenhum fator de risco acionável proeminente foi identificado.")

    # --- ABA 3: Simulador What-If ---
    with tab_simulator:
        st.header("Simulador de Cenários")
        st.info(f"Simulando para o funcionário **{employee_data['EmployeeNumber']}** com risco atual de **{employee_data['predicted_probability']:.1%}**")

        with st.form(key="simulation_form"):
            st.subheader("Altere os fatores abaixo para simular intervenções:")
            
            sim_input_data = {}
            # Exemplo de como adicionar campos para simulação (adicione mais conforme necessário)
            sim_input_data['MonthlyIncome'] = st.slider(
                "Renda Mensal (R$)", 
                min_value=int(df_employees['MonthlyIncome'].min()),
                max_value=int(df_employees['MonthlyIncome'].max()),
                value=int(employee_data['MonthlyIncome']),
                step=100
            )
            sim_input_data['JobSatisfaction'] = st.select_slider(
                "Satisfação no Trabalho",
                options=[1, 2, 3, 4],
                value=int(employee_data['JobSatisfaction'])
            )

            submit_button = st.form_submit_button(label="Simular Novo Risco")

        if submit_button:
            # Cria um DataFrame com os dados da simulação
            sim_df = employee_data.to_frame().T.copy()
            for key, value in sim_input_data.items():
                sim_df[key] = value

            # Prepara os dados e calcula o novo risco
            X_sim = prepare_data_for_model(sim_df, model_features)
            new_probability = model.predict_proba(X_sim)[0, 1]
            
            st.subheader("Resultado da Simulação")
            col1, col2 = st.columns(2)
            col1.metric("Risco Anterior", f"{employee_data['predicted_probability']:.1%}")
            col2.metric("Novo Risco Simulado", f"{new_probability:.1%}", delta=f"{(new_probability - employee_data['predicted_probability']):.1%}")