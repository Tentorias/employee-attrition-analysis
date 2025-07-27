# src/attrition/causal_analysis.py

from dowhy import CausalModel
import pandas as pd

def run_causal_analysis(df: pd.DataFrame, treatment_var: str, outcome_var: str, common_causes: list):
    """
    Executa a análise causal usando DoWhy para uma dada variável de tratamento.

    Args:
        df (pd.DataFrame): DataFrame pré-processado.
        treatment_var (str): Nome da coluna da variável de tratamento (causa).
        outcome_var (str): Nome da coluna da variável de resultado (efeito).
        common_causes (list): Lista de nomes das colunas de variáveis de confusão.

    Returns:
        tuple: (CausalEstimate, RefutationResults_RandomCause, RefutationResults_UnobservedCause)
    """
    if df.empty:
        print("DataFrame vazio. Não é possível rodar a análise causal.")
        return None, None, None

    
    valid_common_causes = [col for col in common_causes if col in df.columns]
    
    if len(valid_common_causes) != len(common_causes):
        print(f"Atenção: Algumas common_causes não foram encontradas no DataFrame e foram removidas: {list(set(common_causes) - set(valid_common_causes))}")
    
    
    required_cols = [treatment_var, outcome_var] + valid_common_causes
    df_for_causal = df[required_cols].copy() 

    
    model = CausalModel(
        data=df_for_causal,
        treatment=treatment_var,
        outcome=outcome_var,
        common_causes=valid_common_causes
    )

   
    identified_estimand = model.identify_effect(
        proceed_when_unidentifiable=True
    )
    print(f"\nIdentified Estimand for {treatment_var}:\n", identified_estimand)

    
    causal_estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_matching",
        control_value=0,
        treatment_value=1,
    )
    print(f"\nCausal Estimate of {treatment_var} on {outcome_var}: {causal_estimate.value}")

    
    refutation_random = model.refute_estimate(
        identified_estimand, causal_estimate,
        method_name="random_common_cause"
    )
    print(f"\nRefutation Results (Random Common Cause) for {treatment_var}:\n", refutation_random)

    
    refutation_unobserved = model.refute_estimate(
        identified_estimand, causal_estimate,
        method_name="add_unobserved_common_cause",
    )
    print(f"\nRefutation Results (Unobserved Common Cause) for {treatment_var}:\n", refutation_unobserved)

    return causal_estimate, refutation_random, refutation_unobserved

# Exemplo de uso
if __name__ == "__main__":
    from data_processing import load_and_preprocess_data

    preprocessed_df = load_and_preprocess_data()

    if not preprocessed_df.empty:
        all_possible_common_causes = [
            'age', 'dailyrate', 'distancefromhome', 'education', 'employeecount',
            'employeenumber', 'environmentsatisfaction', 'hourlyrate', 'jobinvolvement',
            'joblevel', 'jobsatisfaction', 'monthlyincome', 'monthlyrate',
            'numcompaniesworked', 'percentsalaryhike', 'performancerating',
            'relationshipsatisfaction', 'standardhours', 'stockoptionlevel',
            'totalworkingyears', 'trainingtimeslastyear', 'worklifebalance',
            'yearsatcompany', 'yearsincurrentrole', 'yearssincelastpromotion',
            'yearswithcurrmanager',
            'gender_Male',
            'department_Research & Development', 
            'department_Sales',
            'businesstravel_Travel_Frequently',
            'businesstravel_Travel_Rarely',
            'educationfield_Life Sciences', 
            'educationfield_Marketing',
            'educationfield_Medical',
            'educationfield_Other',
            'educationfield_Technical Degree',
            'jobrole_Healthcare Representative',
            'jobrole_Human Resources',
            'jobrole_Laboratory Technician',
            'jobrole_Manager',
            'jobrole_Manufacturing Director',
            'jobrole_Research Director',
            'jobrole_Research Scientist',
            'jobrole_Sales Executive',
            'jobrole_Sales Representative',
            'maritalstatus_Married',
            'maritalstatus_Single',
            'over18_Y', 
        ]

     
        print("\n--- Análise Causal: Horas Extras ---")
        overtime_estimate, _, _ = run_causal_analysis(
            preprocessed_df,
            treatment_var='overtime_Yes',
            outcome_var='attrition',
            common_causes=all_possible_common_causes
        )


        print("\n--- Análise Causal: Alta Satisfação no Trabalho ---")
        satisfaction_estimate, _, _ = run_causal_analysis(
            preprocessed_df,
            treatment_var='high_job_satisfaction',
            outcome_var='attrition',
            common_causes=all_possible_common_causes
        )