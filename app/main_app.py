import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import sqlite3
import joblib

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
    from app.ui_config import LABEL_MAPPING
except ImportError:
    LABEL_MAPPING = {}
    st.warning("Arquivo ui_config.py não encontrado. Usando labels padrão.")

# --- CONSTANTES ---
MODEL_PATH = project_root / "models" / "production_model.pkl"
FEATURES_PATH = project_root / "artifacts" / "features" / "features.pkl"
SHAP_EXPLAINER_PATH = project_root / "models" / "production_shap_explainer.pkl"
DB_PATH = project_root / "database" / "hr_analytics.db"

# --- FUNÇÕES ---
@st.cache_data
def load_data():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df_emp = pd.read_sql_query("SELECT * FROM employees", conn)
            df_pred = pd.read_sql_query("SELECT * FROM predictions", conn)
        df = pd.merge(df_emp, df_pred[['EmployeeNumber', 'predicted_probability']], on='EmployeeNumber', how='left')
        df['predicted_probability'].fillna(0, inplace=True)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        explainer = joblib.load(SHAP_EXPLAINER_PATH)
        return model, features, explainer
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None, None

def prepare_data(df, features):
    df_proc = df.copy()
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18']
    df_proc.drop(columns=[col for col in cols_to_drop if col in df_proc.columns], errors='ignore', inplace=True)
    if 'TotalWorkingYears' in df_proc.columns and 'NumCompaniesWorked' in df_proc.columns:
        df_proc['YearsPerCompany'] = (df_proc['TotalWorkingYears'] / df_proc['NumCompaniesWorked'].replace(0, 1)).round(4)
    df_proc = pd.get_dummies(df_proc, drop_first=True, dtype=float)
    X = df_proc.reindex(columns=features, fill_value=0)
    return X

def get_top_factors(shap_values, features, top_n=3):
    shap_map = dict(zip(features, shap_values))
    risk_factors = {k: v for k, v in shap_map.items() if v > 0}
    sorted_factors = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
    return sorted_factors[:top_n]

# --- APP ---
df_data = load_data()
model, model_features, explainer = load_model_artifacts()

st.title("\U0001F4C8 Diagnóstico Tático de Turnover")

if df_data.empty or model is None:
    st.warning("Dados ou modelo indisponíveis.")
else:
    tab_team, tab_individual = st.tabs(["\U0001F465 Risco da Equipe", "\U0001F464 Diagnóstico Individual"])

    with tab_team:
        st.header("Risco de Saída por Departamento")
        departments = sorted(df_data['Department'].unique())
        dept = st.selectbox("Departamento:", departments)

        team = df_data[df_data['Department'] == dept]
        team_sorted = team.sort_values(by='predicted_probability', ascending=False)
        emp_options = {f"{row['JobRole']} (ID: {row['EmployeeNumber']})": row['EmployeeNumber'] for _, row in team_sorted.iterrows()}

        selected_emp = st.selectbox("Selecione um funcionário:", options=emp_options.keys())
        emp_id = emp_options[selected_emp]

        if st.button("Analisar Funcionário", use_container_width=True):
            st.session_state.selected_employee = emp_id
            st.toast("Funcionário carregado! Acesse a aba 'Diagnóstico Individual'.", icon="\U0001F464")

        team_display = team_sorted.copy()
        team_display['risk_percent'] = team_display['predicted_probability'] * 100
        st.dataframe(
            team_display[['EmployeeNumber', 'JobRole', 'risk_percent']],
            use_container_width=True, hide_index=True,
            column_config={
                "EmployeeNumber": "ID",
                "JobRole": "Cargo",
                "risk_percent": st.column_config.ProgressColumn("Risco de Saída", format="%.1f%%", min_value=0, max_value=100),
            }
        )

    with tab_individual:
        st.header("Diagnóstico Individual")

        if 'selected_employee' in st.session_state:
            emp_row = df_data[df_data['EmployeeNumber'] == st.session_state.selected_employee]
            if not emp_row.empty:
                emp = emp_row.iloc[0]
                st.subheader(f"Funcionário ID: {emp['EmployeeNumber']}")
                st.write(f"Cargo: **{emp['JobRole']}**")
                st.metric("Risco Atual de Saída", f"{emp['predicted_probability']:.1%}")

                emp_df = pd.DataFrame([emp_row.iloc[0]])
                X_emp = prepare_data(emp_df, model_features)
                shap_vals = explainer.shap_values(X_emp)
                top_factors = get_top_factors(shap_vals[0], X_emp.columns)

                st.subheader("Principais Fatores que Contribuem para o Risco")
                for feat, val in top_factors:
                    feat_label = LABEL_MAPPING.get(feat, feat)
                    st.markdown(f"- **{feat_label}**")
            else:
                st.warning("Funcionário não encontrado.")
        else:
            st.info("Selecione um funcionário na aba 'Risco da Equipe' para visualizar o diagnóstico.")