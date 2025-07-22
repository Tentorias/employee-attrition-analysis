import sys
from pathlib import Path
import streamlit as st
import pandas as pd
<<<<<<< HEAD
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
=======
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76
import joblib
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv


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
<<<<<<< HEAD
DB_PATH = project_root / "database" / "hr_analytics.db"
=======
DATABASE_URL = os.getenv("DATABASE_URL")
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76

# --- FUNÇÕES ---

<<<<<<< HEAD
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
=======
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
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76
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

<<<<<<< HEAD
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
=======
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
   
    if feature_name in label_map:
        return label_map[feature_name]
    
 
    for base_feature in value_map.keys():
        prefix = f"{base_feature}_"
        if feature_name.startswith(prefix):
            value = feature_name[len(prefix):]
            if base_feature in label_map and value in value_map[base_feature]:
                base_label = label_map[base_feature]
                value_label = value_map[base_feature][value]
                return f"{base_label}: {value_label}"
            
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76
    
    return feature_name

<<<<<<< HEAD
    if model_pipeline and explainer and model_features:
        employee_df = pd.DataFrame([employee_data])
        X_final = prepare_data_for_model(employee_df, model_features)
        
        shap_values = explainer.shap_values(X_final)
        top_contributors = get_top_shap_contributors(shap_values[0], X_final.columns)
        st.session_state.initial_analysis = {"top_contributors": top_contributors}
    
    st.session_state.selected_employee = employee_data
    st.toast(f"Funcionário {employee_id} carregado para análise!", icon="👤")
=======
# --- LÓGICA PRINCIPAL DO APP ---
st.title("\U0001F4C8 Diagnóstico Tático de Turnover")
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76

df_employees = load_employee_data()
model, model_features, explainer = load_model_artifacts()

if df_employees.empty or model is None:
    st.warning("Dados dos funcionários ou modelo indisponíveis.")
else:
<<<<<<< HEAD
    actual_model = model_pipeline.named_steps['classifier'] if hasattr(model_pipeline, 'steps') else model_pipeline
=======
    X_all = prepare_data_for_model(df_employees, model_features)
    probabilities = model.predict_proba(X_all)[:, 1]
    df_employees['predicted_probability'] = probabilities
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76

    tab_team, tab_individual = st.tabs(["\U0001F465 Risco da Equipe", "\U0001F464 Diagnóstico Individual"])

    with tab_team:
        st.header("Risco de Saída por Departamento")
        departments = sorted(df_employees['Department'].unique())
        dept = st.selectbox("Departamento:", departments)
        team = df_employees[df_employees['Department'] == dept]
        team_sorted = team.sort_values(by='predicted_probability', ascending=False)
        emp_options = {f"{row['JobRole']} (ID: {row['EmployeeNumber']})": row['EmployeeNumber'] for _, row in team_sorted.iterrows()}

<<<<<<< HEAD
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
=======
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
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76
