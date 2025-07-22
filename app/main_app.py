import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import joblib

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Simulador de Attrition", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- CORREÇÃO PARA O ModuleNotFoundError ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Tenta importar a UI config, mas não quebra se não encontrar
try:
    from ui_config import HELP_TEXTS, LABEL_MAPPING, VALUE_MAPPING
except ImportError:
    HELP_TEXTS, LABEL_MAPPING, VALUE_MAPPING = {}, {}, {}
    st.warning("Arquivo ui_config.py não encontrado. Usando labels padrão.")


# --- 1. Constantes e Configurações Corrigidas ---
MODEL_PATH = project_root / "models" / "production_model.pkl"
FEATURES_PATH = project_root / "artifacts" / "features" / "features.pkl"
SHAP_EXPLAINER_PATH = project_root / "models" / "production_shap_explainer.pkl"
DB_PATH = project_root / "database" / "hr_analytics.db"

NON_EDITABLE_FIELDS = ["Age", "Gender", "MaritalStatus", "DistanceFromHome", "Department", "JobRole"]

REVERSED_VALUE_MAPPING = {
    feature: {v: k for k, v in options.items()}
    for feature, options in VALUE_MAPPING.items()
}
FEATURE_GROUPS = {
    "Informações Pessoais 🔒": ["Age", "Gender", "MaritalStatus", "DistanceFromHome"],
    "Carreira e Cargo 🎯": ["Department", "JobRole", "JobLevel", "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager", "TotalWorkingYears", "NumCompaniesWorked", "YearsSinceLastPromotion", "TrainingTimesLastYear"],
    "Remuneração 💰": ["MonthlyIncome", "PercentSalaryHike", "StockOptionLevel"],
    "Satisfação e Engajamento ❤️": ["EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance", "OverTime", "PerformanceRating"]
}

# --- 2. Funções de Carregamento e Análise Corrigidas ---

@st.cache_data(ttl=3600)
def load_data():
    """
    Carrega os dados da tabela 'employees' e 'predictions' do banco de dados PostgreSQL.
    """
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        st.error("A variável de ambiente DATABASE_URL não foi encontrada. Configure-a no arquivo .env.")
        return pd.DataFrame()

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            df_emp = pd.read_sql_query("SELECT * FROM employees", conn)
            try:
                df_pred = pd.read_sql_query("SELECT * FROM predictions", conn)
                df = pd.merge(df_emp, df_pred[['EmployeeNumber', 'predicted_probability']], on='EmployeeNumber', how='left')
                df['predicted_probability'].fillna(0, inplace=True)
            except Exception:
                st.warning("Tabela 'predictions' não encontrada no banco. O risco será exibido como 0%.")
                df = df_emp
                df['predicted_probability'] = 0.0
        return df
    except Exception as e:
        st.error(f"Erro ao conectar ou carregar dados do PostgreSQL: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model_artifacts():
    """Carrega todos os artefatos de ML necessários."""
    paths = [MODEL_PATH, FEATURES_PATH, SHAP_EXPLAINER_PATH]
    if not all(p.exists() for p in paths):
        st.error("Um ou mais artefatos de modelo (.pkl) não foram encontrados. Execute o pipeline de treinamento.")
        return None, None, None
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        model_features = joblib.load(FEATURES_PATH)
        explainer = joblib.load(SHAP_EXPLAINER_PATH)
        return model_pipeline, model_features, explainer
    except Exception as e:
        st.error(f"Erro ao carregar os arquivos .pkl: {e}")
        return None, None, None

def prepare_data_for_model(input_df: pd.DataFrame, model_features: list):
    """
    Aplica a engenharia de features e alinha com o modelo,
    replicando a lógica da API e do pipeline de treino.
    """
    df_processed = input_df.copy()
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18']
    df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns], inplace=True)
    df_processed['YearsPerCompany'] = (df_processed['TotalWorkingYears'] / (df_processed['NumCompaniesWorked'] + 1)).round(4)
    categorical_cols = df_processed.select_dtypes(include=["object"]).columns.tolist()
    if 'Attrition' in categorical_cols:
        categorical_cols.remove('Attrition')
    if categorical_cols:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True, dtype=float)
    X_final = df_processed.reindex(columns=model_features, fill_value=0)
    return X_final

def get_top_shap_contributors(shap_values, feature_names):
    """Extrai os 3 principais fatores que aumentam o risco."""
    feature_shap_map = dict(zip(feature_names, shap_values))
    risk_factors = {k: v for k, v in feature_shap_map.items() if v > 0}
    sorted_risk_factors = sorted(risk_factors.items(), key=lambda item: item[1], reverse=True)
    return sorted_risk_factors[:3]

def generate_form_widgets(container, features_to_display: list, df_reference: pd.DataFrame, default_values: dict, employee_id: int):
    """Gera os widgets do formulário com chaves dinâmicas."""
    input_data = {}
    for col in features_to_display:
        if col not in df_reference.columns: continue
        
        friendly_label = LABEL_MAPPING.get(col, col)
        help_text = HELP_TEXTS.get(col)
        default_val = default_values.get(col)
        is_disabled = col in NON_EDITABLE_FIELDS
        widget_key = f"{col}_{employee_id}"

        if col in VALUE_MAPPING:
            options_map = VALUE_MAPPING.get(col, {})
            friendly_options = list(options_map.values())
            try:
                default_option = VALUE_MAPPING[col].get(default_val, friendly_options[0])
                default_index = friendly_options.index(default_option)
            except (ValueError, KeyError): 
                default_index = 0
            
            selected_friendly = container.selectbox(friendly_label, friendly_options, index=default_index, help=help_text, key=f"sb_{widget_key}", disabled=is_disabled)
            input_data[col] = REVERSED_VALUE_MAPPING.get(col, {}).get(selected_friendly)
        elif pd.api.types.is_numeric_dtype(df_reference[col]):
            min_val, max_val = int(df_reference[col].min()), int(df_reference[col].max())
            step = 100 if "Income" in col else 1
            val = int(default_val) if default_val is not None else min_val
            if val < min_val: val = min_val
            if val > max_val: val = max_val
            input_data[col] = container.slider(friendly_label, min_val, max_val, val, step, help=help_text, key=f"sl_{widget_key}", disabled=is_disabled)
    return input_data


# --- 3. Lógica Principal da UI ---
df_full = load_data() # <<< CORREÇÃO APLICADA AQUI
model_pipeline, model_features, explainer = load_model_artifacts()

def update_employee_state(employee_id):
    """Carrega os dados do funcionário e calcula sua análise de risco inicial."""
    employee_data = df_full[df_full['EmployeeNumber'] == employee_id].iloc[0].to_dict()
    
    if 'simulation_result' in st.session_state:
        del st.session_state.simulation_result

    if model_pipeline and explainer and model_features:
        employee_df = pd.DataFrame([employee_data])
        X_final = prepare_data_for_model(employee_df, model_features)
        
        shap_values = explainer.shap_values(X_final)
        top_contributors = get_top_shap_contributors(shap_values[0], X_final.columns)
        st.session_state.initial_analysis = {"top_contributors": top_contributors}
    
    st.session_state.selected_employee = employee_data
    st.toast(f"Funcionário {employee_id} carregado para análise!", icon="👤")

if 'selected_employee' not in st.session_state:
    st.session_state.selected_employee = df_full.iloc[0].to_dict() if not df_full.empty else {}

st.title("💡 Ferramenta Tática de Análise de Turnover")

if df_full.empty or model_pipeline is None:
    st.warning("Não foi possível carregar os dados ou o modelo. A aplicação não pode continuar.")
else:
    actual_model = model_pipeline.named_steps['classifier'] if hasattr(model_pipeline, 'steps') else model_pipeline

    tab_analise_equipe, tab_simulador = st.tabs(["👥 Análise de Risco da Equipe", "👤 Simulação Individual"])

    with tab_analise_equipe:
        st.header("Visão Preditiva de Risco por Departamento")
        departments = sorted(df_full['Department'].unique())
        selected_department = st.selectbox("Selecione um Departamento:", departments, key="dept_selector")
        if selected_department:
            team_df = df_full[df_full['Department'] == selected_department]
            team_df_sorted = team_df.sort_values(by="predicted_probability", ascending=False)
            employee_options = {f"{row['JobRole']} (ID: {row['EmployeeNumber']})": row['EmployeeNumber'] for _, row in team_df_sorted.iterrows()}
            
            col1, col2 = st.columns([3, 1.5])
            with col1:
                selected_employee_display = st.selectbox("Selecione um funcionário:", options=employee_options.keys())
            with col2:
                st.write("")
                if selected_employee_display:
                    selected_employee_id = employee_options[selected_employee_display]
                    st.button("Analisar Funcionário", type="primary", use_container_width=True, on_click=update_employee_state, args=(selected_employee_id,))
            
            df_display = team_df_sorted.copy()
            df_display['risk_percent'] = df_display['predicted_probability'] * 100
            st.dataframe(
                df_display[['EmployeeNumber', 'JobRole', 'risk_percent']],
                use_container_width=True, hide_index=True,
                column_config={
                    "EmployeeNumber": "ID do Funcionário",
                    "JobRole": "Cargo",
                    "risk_percent": st.column_config.ProgressColumn(
                        "Risco de Saída", format="%.1f%%", min_value=0, max_value=100
                    ),
                }
            )

    with tab_simulador:
        st.header("Simulador 'What-If' para Análise Individual")
        emp_data = st.session_state.selected_employee
        if emp_data:
            employee_id = emp_data.get('EmployeeNumber', 0)
            st.info(f"Analisando o Funcionário: **{employee_id}** | Cargo: **{emp_data.get('JobRole', 'N/A')}** | Risco Atual: **{emp_data.get('predicted_probability', 0):.1%}**")

            if 'initial_analysis' in st.session_state:
                st.subheader("Diagnóstico Inicial (Fatores de Risco)")
                st.warning("Estes são os principais fatores que contribuem para o risco de saída ATUAL deste funcionário.", icon="🔥")
                for feature, _ in st.session_state.initial_analysis['top_contributors']:
                    st.markdown(f"- **{LABEL_MAPPING.get(feature, feature)}**")
            
            st.markdown("---")
            st.subheader("Formulário de Simulação")
            
            input_data = {}
            inner_tabs = st.tabs(list(FEATURE_GROUPS.keys()))
            for i, group_name in enumerate(inner_tabs):
                with group_name:
                    input_data.update(generate_form_widgets(st.container(), FEATURE_GROUPS[list(FEATURE_GROUPS.keys())[i]], df_full, emp_data, employee_id))
            
            col1, col2, col3 = st.columns([2, 1.5, 2])
            with col2:
                if st.button("Simular Mudanças", type="primary", use_container_width=True):
                    with st.spinner("Avaliando novo cenário..."):
                        sim_df = pd.DataFrame([input_data])
                        X_final_sim = prepare_data_for_model(sim_df, model_features)
                        
                        probability = actual_model.predict_proba(X_final_sim)[:, 1][0]
                        
                        threshold = 0.5
                        try:
                            threshold = joblib.load(project_root / "artifacts" / "models" / "threshold_optimizado.pkl")
                        except:
                            pass

                        prediction = 1 if probability >= threshold else 0
                        st.session_state.simulation_result = {"prediction": prediction, "probability": probability}
            
            if 'simulation_result' in st.session_state:
                res = st.session_state.simulation_result
                st.header("Resultado da Simulação")
                st.metric("Novo Risco Simulado", f"{res['probability']:.1%}")
                if res['prediction'] == 1: st.error("**Ainda em Alto Risco!**", icon="🚨")
                else: st.success("**Risco Reduzido com Sucesso!**", icon="✅")
        else:
            st.warning("Nenhum funcionário selecionado.")