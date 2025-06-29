import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_fscore_support
)

# --- Configuração de Caminhos ---
BASE_DIR = Path(__file__).resolve().parent.parent
X_TEST_PATH = BASE_DIR / "artifacts" / "features" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "artifacts" / "features" / "y_test.csv"
MODEL_PATH = BASE_DIR / "models" / "production_model.pkl"
REPORTS_DIR = BASE_DIR / "reports"
# NOVO: Caminho para salvar o novo artefato
SHAP_EXPLAINER_PATH = BASE_DIR / "artifacts" / "models" / "shap_explainer.pkl" 
REPORTS_DIR.mkdir(exist_ok=True)

def create_artifacts():
    """
    MODIFICADO: Realiza a avaliação e, mais importante,
    cria e salva o artefato do explicador SHAP.
    """
    print("🚀 Iniciando avaliação e criação de artefatos...")

    # --- 1. Carregar Dados e Modelo ---
    if not all([X_TEST_PATH.exists(), Y_TEST_PATH.exists(), MODEL_PATH.exists()]):
        print("❌ ERRO: Arquivos de teste ou modelo não encontrados. Execute o pipeline de treino primeiro.")
        return

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    model_prod = joblib.load(MODEL_PATH)
    print("✔ Dados de teste e modelo carregados.")

    # --- 2. NOVO: Criar e Salvar o Explicador SHAP ---
    print("\n🔥 Criando e salvando o explicador SHAP...")
    # O explicador SHAP é criado com base no modelo.
    explainer = shap.Explainer(model_prod)
    with open(SHAP_EXPLAINER_PATH, 'wb') as f:
        joblib.dump(explainer, f)
    print(f"✔ Explicador SHAP salvo com sucesso em: {SHAP_EXPLAINER_PATH}")

    # --- 3. Avaliação do Modelo (código anterior mantido) ---
    print("\n--- Avaliação do Modelo XGBoost (Produção) ---")
    y_pred_prod = model_prod.predict(X_test)
    y_proba_prod = model_prod.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_prod, target_names=['No', 'Yes']))
    
    # ... (o restante do código de avaliação e baseline permanece o mesmo) ...
    print("\n📊 Gerando Curva ROC e calculando AUC...")
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
    roc_curve_path = REPORTS_DIR / "roc_curve_xgboost.png"
    plt.savefig(roc_curve_path)
    print(f"  - Gráfico da Curva ROC salvo em: {roc_curve_path}")


if __name__ == "__main__":
    create_artifacts()
