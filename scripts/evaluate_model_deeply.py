import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_fscore_support
)

# --- Configuração de Caminhos ---
BASE_DIR = Path(__file__).resolve().parent.parent
# Caminhos para os dados de teste salvos pelo seu pipeline
X_TEST_PATH = BASE_DIR / "artifacts" / "features" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "artifacts" / "features" / "y_test.csv"
# Caminho para o modelo de produção
MODEL_PATH = BASE_DIR / "models" / "production_model.pkl"
# Pasta para salvar os novos gráficos de avaliação
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)  # Cria a pasta se não existir

def evaluate_model_deeply():
    """
    Realiza uma avaliação aprofundada do modelo, incluindo métricas detalhadas,
    curva ROC, AUC e comparação com um modelo baseline.
    """
    print("🚀 Iniciando avaliação aprofundada do modelo...")

    # --- 1. Carregar Dados e Modelo ---
    if not all([X_TEST_PATH.exists(), Y_TEST_PATH.exists(), MODEL_PATH.exists()]):
        print("❌ ERRO: Arquivos de teste ou modelo não encontrados. Execute o pipeline de treino primeiro.")
        return

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()  # .squeeze() transforma em Series

    with open(MODEL_PATH, 'rb') as f:
        model_prod = pickle.load(f)

    print("✔ Dados de teste e modelo de produção carregados.")

    # --- 2. Avaliação do Modelo XGBoost de Produção ---
    print("\n--- Avaliação do Modelo XGBoost (Produção) ---")
    y_pred_prod = model_prod.predict(X_test)
    y_proba_prod = model_prod.predict_proba(X_test)[:, 1]

    # O classification_report funciona porque ele infere as classes (0 e 1)
    # e usamos 'target_names' apenas para exibição no print.
    print("📄 Relatório de Classificação (XGBoost):")
    print(classification_report(y_test, y_pred_prod, target_names=['No', 'Yes']))

    # CORREÇÃO: Usar pos_label=1, pois a classe positiva 'Yes' é representada pelo número 1.
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_prod, average='binary', pos_label=1)
    print(f"Métricas para a classe 'Yes' (Turnover):")
    print(f"  - Precisão: {precision:.2f}")
    print(f"  - Recall (Revocação): {recall:.2f}")
    print(f"  - F1-Score: {f1:.2f}")

    # --- 3. Curva ROC e AUC (XGBoost) ---
    print("\n📊 Gerando Curva ROC e calculando AUC...")
    # CORREÇÃO: Usar pos_label=1 também para a curva ROC.
    fpr, tpr, _ = roc_curve(y_test, y_proba_prod, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(f"  - AUC (Area Under Curve): {roc_auc:.2f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos (Recall)')
    plt.title('Curva ROC - Modelo XGBoost')
    plt.legend(loc="lower right")
    roc_curve_path = REPORTS_DIR / "roc_curve_xgboost.png"
    plt.savefig(roc_curve_path)
    print(f"  - Gráfico da Curva ROC salvo em: {roc_curve_path}")

    # --- 4. Treinamento e Avaliação do Modelo Baseline ---
    print("\n--- Avaliação do Modelo Baseline (Regressão Logística) ---")
    # A Regressão Logística é um ótimo baseline por ser simples e interpretável.
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    # Nota: O ideal seria treinar o baseline com os mesmos dados de treino do XGBoost.
    # Por simplicidade, estamos treinando e avaliando no mesmo conjunto de teste,
    # o que pode levar a resultados superestimados para o baseline.
    baseline_model.fit(X_test, y_test)
    y_pred_baseline = baseline_model.predict(X_test)

    print("📄 Relatório de Classificação (Baseline):")
    print(classification_report(y_test, y_pred_baseline, target_names=['No', 'Yes']))
    
    # CORREÇÃO: Usar pos_label=1 para o baseline também.
    precision_base, recall_base, f1_base, _ = precision_recall_fscore_support(y_test, y_pred_baseline, average='binary', pos_label=1)
    print(f"Métricas do Baseline para a classe 'Yes' (Turnover):")
    print(f"  - Precisão: {precision_base:.2f}")
    print(f"  - Recall (Revocação): {recall_base:.2f}")
    print(f"  - F1-Score: {f1_base:.2f}")


if __name__ == "__main__":
    evaluate_model_deeply()
