import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np # Adicionado para transformações logarítmicas
import sqlite3
import joblib

# --- CORREÇÃO PARA O ModuleNotFoundError ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui_config import HELP_TEXTS, LABEL_MAPPING, VALUE_MAPPING
# REMOVIDO: Não vamos mais usar a função de predição externa
# from src.attrition.models.predict import main as predict_attrition

# --- 1. Constantes e Configurações ---
MODEL_PATH = project_root / "models" / "production_model.pkl"
THRESHOLD_PATH = project_root / "artifacts" / "models" / "threshold_optimizado.pkl"
DB_PATH = project_root / "database" / "hr_analytics.db"

REVERSED_VALUE_MAPPING = {
    feature: {v: k for k, v in options.items()}
    for feature, options in VALUE_MAPPING.items()
}

FEATURE_GROUPS = {
    "Informações Pessoais": ["Age", "Gender", "MaritalStatus", "DistanceFromHome"],
    "Carreira e Cargo": ["Department", "JobRole", "JobLevel", "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager", "TotalWorkingYears", "NumCompaniesWorked", "YearsSinceLastPromotion", "TrainingTimesLastYear"],
    "Remuneração": ["MonthlyIncome", "PercentSalaryHike", "StockOptionLevel"],
    "Satisfação e Engajamento": ["EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance", "OverTime", "PerformanceRating"]
}

# --- 2. Funções de Carregamento ---

@st.cache_data
def load_data_from_db():
    """Carrega e junta as tabelas 'employees' e 'predictions' do banco de dados."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df_employees = pd.read_sql_query("SELECT * FROM employees", conn)
            df_predictions = pd.read_sql_query("SELECT * FROM predictions", conn)
        df_full = pd.merge(df_employees, df_predictions[['EmployeeNumber', 'predicted_probability']], on='EmployeeNumber', how='left')
        df_full['predicted_probability'].fillna(0, inplace=True)
        return df_full
    except Exception as e:
        st.error(f"Erro ao carregar dados do banco de dados: {e}. Execute 'scripts/generate_predictions.py' primeiro.")
        return pd.DataFrame()

@st.cache_resource
def load_model_artifacts():
    """Carrega o modelo e o threshold."""
    try:
        model = joblib.load(MODEL_PATH)
        threshold = joblib.load(THRESHOLD_PATH)
        return model, threshold
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar modelo/threshold: {e}. Certifique-se de que os caminhos estão corretos.")
        return None, None

def generate_form_widgets(container, features_to_display: list, df_reference: pd.DataFrame, default_values: dict):
    """Gera widgets usando valores padrão do funcionário selecionado."""
    input_data = {}
    for col in features_to_display:
        if col not in df_reference.columns:
            continue
        
        friendly_label = LABEL_MAPPING.get(col, col)
        help_text = HELP_TEXTS.get(col)
        default_val = default_values.get(col)

        if col in VALUE_MAPPING:
            options_map = VALUE_MAPPING.get(col, {})
            friendly_options = list(options_map.values())
            try:
                default_index = friendly_options.index(VALUE_MAPPING[col].get(default_val, friendly_options[0]))
            except ValueError:
                default_index = 0
            selected_friendly = container.selectbox(friendly_label, friendly_options, index=default_index, help=help_text, key=f"sb_{col}")
            input_data[col] = REVERSED_VALUE_MAPPING.get(col, {}).get(selected_friendly)
        elif pd.api.types.is_numeric_dtype(df_reference[col]):
            min_val, max_val = int(df_reference[col].min()), int(df_reference[col].max())
            step = 100 if "Income" in col else 1
            # Garante que o valor padrão esteja dentro do range do slider
            val = int(default_val)
            if val < min_val: val = min_val
            if val > max_val: val = max_val
            input_data[col] = container.slider(friendly_label, min_val, max_val, val, step, help=help_text, key=f"sl_{col}")
    return input_data

# --- 3. Lógica Principal da UI ---
df_full = load_data_from_db()
model, threshold = load_model_artifacts()

if 'selected_employee' not in st.session_state:
    st.session_state.selected_employee = df_full.iloc[0].to_dict() if not df_full.empty else {}

def update_employee_state(employee_id):
    employee_data = df_full[df_full['EmployeeNumber'] == employee_id].iloc[0].to_dict()
    st.session_state.selected_employee = employee_data
    st.toast(f"Funcionário {employee_id} carregado para simulação!", icon="👤")

st.title("💡 Ferramenta Tática de Análise de Turnover")

if df_full.empty or model is None:
    st.warning("Não foi possível carregar os dados ou o modelo. A aplicação não pode continuar.")
else:
    tab_analise_equipe, tab_simulador = st.tabs(["👥 Análise de Risco da Equipe", "👤 Simulação Individual"])

    with tab_analise_equipe:
        st.header("Visão Preditiva de Risco por Departamento")
        st.markdown("Selecione um departamento para ver a lista de funcionários ordenada por risco de saída. Clique em 'Simular' para analisar um caso individual.")
        departments = sorted(df_full['Department'].unique())
        selected_department = st.selectbox("Selecione um Departamento", departments, key="dept_selector")
        if selected_department:
            team_df_sorted = df_full[df_full['Department'] == selected_department].sort_values(by="predicted_probability", ascending=False)
            st.subheader(f"Funcionários em Risco em: {selected_department}")
            for _, row in team_df_sorted.iterrows():
                prob = row['predicted_probability']
                color = "red" if prob > 0.6 else "orange" if prob > 0.3 else "green"
                col_info, col_prob, col_action = st.columns([3, 1.5, 1.5])
                col_info.markdown(f"**Cargo:** {row['JobRole']} (ID: {row['EmployeeNumber']})")
                col_prob.metric("Risco de Saída", f"{prob:.1%}")
                col_action.button("Simular", key=f"btn_{row['EmployeeNumber']}", on_click=update_employee_state, args=(row['EmployeeNumber'],))
                st.divider()

    with tab_simulador:
        st.header("Simulador 'What-If' para Análise Individual")
        emp_data = st.session_state.selected_employee
        if emp_data:
            st.info(f"Simulando para o Funcionário: **{emp_data['EmployeeNumber']}** | Cargo: **{emp_data['JobRole']}**")
            
            input_data = {}
            inner_tabs = st.tabs(list(FEATURE_GROUPS.keys()))
            for i, group_name in enumerate(inner_tabs):
                with group_name:
                    input_data.update(
                        generate_form_widgets(st.container(), FEATURE_GROUPS[list(FEATURE_GROUPS.keys())[i]], df_full, emp_data)
                    )

            st.write("")
            col1, col2, col3 = st.columns([2, 1.5, 2])
            with col2:
                if st.button("Fazer Predição", type="primary", use_container_width=True):
                    with st.spinner("Avaliando o perfil do funcionário..."):
                        # --- INÍCIO DA LÓGICA DE PREDIÇÃO INTEGRADA ---
                        # 1. Cria um DataFrame com os dados atuais do formulário
                        sim_df = pd.DataFrame([input_data])
                        
                        # 2. Recria a engenharia de features, igual ao script de predição em massa
                        sim_df['YearsPerCompany'] = (sim_df['YearsAtCompany'] / (sim_df['NumCompaniesWorked'] + 1)).round(2)
                        sim_df['MonthlyIncome_log'] = np.log(sim_df['MonthlyIncome'] + 1)
                        sim_df['TotalWorkingYears_log'] = np.log(sim_df['TotalWorkingYears'] + 1)
                        
                        categorical_cols = sim_df.select_dtypes(include=['object', 'category']).columns
                        df_encoded = pd.get_dummies(sim_df, columns=categorical_cols, drop_first=True)
                        
                        # 3. Alinha com as features do modelo
                        model_feature_names = model.get_booster().feature_names
                        X_final = df_encoded.reindex(columns=model_feature_names, fill_value=0)
                        
                        # 4. Faz a predição
                        probability = model.predict_proba(X_final)[:, 1][0]
                        prediction = 1 if probability >= threshold else 0
                        # --- FIM DA LÓGICA DE PREDIÇÃO ---

                    if prediction is not None:
                        st.header("Resultado da Análise")
                        if prediction == 1:
                            st.error("**Alto Risco de Saída!**", icon="🚨")
                        else:
                            st.success("**Baixo Risco de Saída**", icon="✅")
                        st.metric("Nova Probabilidade de Saída", f"{probability:.2%}")
                        st.progress(float(probability))
                    else:
                        st.error("Não foi possível realizar a predição.")
        else:
            st.warning("Nenhum funcionário selecionado. Por favor, selecione um na aba 'Análise de Risco da Equipe'.")
