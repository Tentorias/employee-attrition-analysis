import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import numpy as np

# Carrega as variáveis de ambiente
load_dotenv()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Turnover",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CONFIG IMPORT ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from app.ui_config import LABEL_MAPPING, VALUE_MAPPING, UNACTIONABLE_FEATURES
except ImportError:
    LABEL_MAPPING, VALUE_MAPPING, UNACTIONABLE_FEATURES = {}, {}, []
    st.warning("Arquivo ui_config.py não encontrado ou incompleto. Usando labels padrão.")

# --- CONSTANTES ---
MODEL_PATH = project_root / "models" / "production_model.pkl"
FEATURES_PATH = project_root / "artifacts" / "features" / "features.pkl"
SHAP_EXPLAINER_PATH = project_root / "models" / "production_shap_explainer.pkl"
DATABASE_URL = os.getenv("DATABASE_URL")

# --- FUNÇÕES ---

@st.cache_data
def load_employee_data():
    if not DATABASE_URL:
        st.error("A URL do banco de dados (DATABASE_URL) não foi configurada.")
        return pd.DataFrame()
    try:
        engine = create_engine(DATABASE_URL)
        df_emp = pd.read_sql_query("SELECT * FROM employees", engine)
        return df_emp
    except Exception as e:
        st.error(f"Erro ao carregar dados dos funcionários do PostgreSQL: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        explainer = joblib.load(SHAP_EXPLAINER_PATH)
        return model, features, explainer
    except Exception as e:
        st.error(f"Erro ao carregar artefatos do modelo: {e}")
        return None, None, None

def prepare_data_for_model(df, features):
    df_proc = df.copy()
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18']
    df_proc.drop(columns=[col for col in cols_to_drop if col in df_proc.columns], errors='ignore', inplace=True)
    if 'TotalWorkingYears' in df_proc.columns and 'NumCompaniesWorked' in df_proc.columns:
        df_proc['YearsPerCompany'] = df_proc.apply(lambda row: row['TotalWorkingYears'] / row['NumCompaniesWorked'] if row['NumCompaniesWorked'] > 0 else row['TotalWorkingYears'], axis=1).round(4)
    df_proc = pd.get_dummies(df_proc, drop_first=True, dtype=float)
    X = df_proc.reindex(columns=features, fill_value=0)
    return X

def get_top_factors(shap_values, features, top_n=5):
    shap_map = dict(zip(features, shap_values))
    risk_factors = {k: v for k, v in shap_map.items() if v > 0}
    sorted_factors = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
    return sorted_factors[:top_n]

# --- FUNÇÃO DE TRADUÇÃO MELHORADA ---
def translate_feature_name(feature_name, label_map, value_map):
    """
    Traduz nomes de features de forma robusta, lidando com valores que contêm '_'.
    """
    # 1. Tenta a correspondência direta (para features numéricas)
    if feature_name in label_map:
        return label_map[feature_name]
    
    # 2. Tenta a correspondência com features categóricas (dummy)
    for base_feature in value_map.keys():
        prefix = f"{base_feature}_"
        if feature_name.startswith(prefix):
            # Extrai o valor que vem depois do prefixo
            value = feature_name[len(prefix):]
            if base_feature in label_map and value in value_map[base_feature]:
                base_label = label_map[base_feature]
                value_label = value_map[base_feature][value]
                return f"{base_label}: {value_label}"
            
    # 3. Se nada funcionar, retorna o nome original
    return feature_name

# --- LÓGICA PRINCIPAL DO APP ---
st.title("\U0001F4C8 Diagnóstico Tático de Turnover")

df_employees = load_employee_data()
model, model_features, explainer = load_model_artifacts()

if df_employees.empty or model is None:
    st.warning("Dados dos funcionários ou modelo indisponíveis.")
else:
    X_all = prepare_data_for_model(df_employees, model_features)
    probabilities = model.predict_proba(X_all)[:, 1]
    df_employees['predicted_probability'] = probabilities

    tab_team, tab_individual = st.tabs(["\U0001F465 Risco da Equipe", "\U0001F464 Diagnóstico Individual"])

    with tab_team:
        st.header("Risco de Saída por Departamento")
        departments = sorted(df_employees['Department'].unique())
        dept = st.selectbox("Departamento:", departments)
        team = df_employees[df_employees['Department'] == dept]
        team_sorted = team.sort_values(by='predicted_probability', ascending=False)
        emp_options = {f"{row['JobRole']} (ID: {row['EmployeeNumber']})": row['EmployeeNumber'] for _, row in team_sorted.iterrows()}

        if emp_options:
            selected_emp_key = st.selectbox("Selecione um funcionário:", options=emp_options.keys())
            emp_id = emp_options[selected_emp_key]
            if st.button("Analisar Funcionário", use_container_width=True):
                st.session_state.selected_employee = emp_id
                st.toast("Funcionário carregado! Acesse a aba 'Diagnóstico Individual'.", icon="\U0001F464")
        else:
            st.warning("Nenhum funcionário encontrado.")
        
        team_display = team_sorted.copy()
        team_display['risk_percent'] = team_display['predicted_probability'] * 100
        st.dataframe(team_display[['EmployeeNumber', 'JobRole', 'risk_percent']], use_container_width=True, hide_index=True, column_config={ "EmployeeNumber": "ID", "JobRole": "Cargo", "risk_percent": st.column_config.ProgressColumn("Risco de Saída", format="%.1f%%", min_value=0, max_value=100) })

    with tab_individual:
        st.header("Diagnóstico Individual")
        if 'selected_employee' in st.session_state:
            emp_row = df_employees[df_employees['EmployeeNumber'] == st.session_state.selected_employee]
            if not emp_row.empty:
                emp = emp_row.iloc[0]
                st.subheader(f"Funcionário ID: {emp['EmployeeNumber']}")
                st.write(f"Cargo: **{emp['JobRole']}**")
                st.metric("Risco Atual de Saída", f"{emp['predicted_probability']:.1%}")

                X_emp = X_all[X_all.index == emp.name]
                shap_vals = explainer.shap_values(X_emp)
                top_factors = get_top_factors(shap_vals[0], X_emp.columns)

                actionable_factors = []
                for feat, val in top_factors:
                    base_feat = feat.split('_')[0]
                    if base_feat not in UNACTIONABLE_FEATURES:
                        actionable_factors.append((feat, val))
                
                st.subheader("Principais Fatores (Acionáveis) que Contribuem para o Risco")
                if actionable_factors:
                    for feat, val in actionable_factors[:3]:
                        feat_label = translate_feature_name(feat, LABEL_MAPPING, VALUE_MAPPING)
                        st.markdown(f"- **{feat_label}**")
                else:
                    st.info("Nenhum fator de risco acionável foi identificado para este funcionário.")
            else:
                st.warning("Funcionário não encontrado.")
        else:
            st.info("Selecione um funcionário na aba 'Risco da Equipe' para visualizar o diagnóstico.")