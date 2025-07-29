from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# --- CONFIGURAÇÕES ---
# Define os caminhos para os artefatos, subindo para a raiz do projeto
project_root = Path(__file__).resolve().parent.parent
MODEL_PATH = project_root / "models" / "production_model.pkl"
X_TEST_PATH = project_root / "artifacts" / "features" / "X_test.csv"
Y_TEST_PATH = project_root / "artifacts" / "features" / "y_test.csv"


def check_performance():
    """
    Carrega o modelo de produção e os dados de teste para realizar uma
    análise de performance completa, gerando métricas e visualizações.
    """
    print("--- INICIANDO ANÁLISE DE PERFORMANCE DO MODELO ---")

    # --- 1. Carregar Artefatos ---
    try:
        model = joblib.load(MODEL_PATH)
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).squeeze()
        print("✅ Modelo e dados de teste carregados com sucesso.")
    except FileNotFoundError as e:
        print(
            f"❌ ERRO: Arquivo não encontrado. Execute o pipeline de treino primeiro. Detalhes: {e}"
        )
        return

    # --- 2. Fazer Predições ---
    # Usa um threshold padrão de 0.5 para avaliação objetiva
    threshold = 0.5
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    print(f"\nPredições calculadas com threshold de {threshold:.2f}")

    # --- 3. Relatório de Classificação ---
    print("\n--- 1. Relatório de Classificação ---")
    print(classification_report(y_test, y_pred, target_names=["Fica (0)", "Sai (1)"]))

    # --- 4. Matriz de Confusão ---
    print("\n--- 2. Matriz de Confusão ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Previsto: Fica", "Previsto: Sai"],
        yticklabels=["Real: Fica", "Real: Sai"],
    )
    plt.title("Matriz de Confusão")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.show()
    print("-> A janela com o gráfico da Matriz de Confusão foi exibida.")

    # --- 5. Curva ROC e AUC ---
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\n--- 3. Curva ROC (AUC = {auc_score:.4f}) ---")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"Curva ROC (AUC = {auc_score:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curva ROC (Receiver Operating Characteristic)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    print("-> A janela com o gráfico da Curva ROC foi exibida.")
    print(
        "   (Quanto mais a curva se aproxima do canto superior esquerdo, melhor o modelo)"
    )

    # --- 6. Curva de Precisão-Recall ---
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"\n--- 4. Curva de Precisão-Recall (AUC = {pr_auc:.4f}) ---")
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, color="blue", lw=2, label=f"Curva P-R (AUC = {pr_auc:.2f})"
    )
    plt.xlabel("Recall (Sensibilidade)")
    plt.ylabel("Precision (Precisão)")
    plt.title("Curva de Precisão-Recall")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()
    print("-> A janela com o gráfico da Curva de Precisão-Recall foi exibida.")
    print(
        "   (Ideal para datasets desbalanceados. Quanto mais próxima do canto superior direito, melhor)"
    )

    print("\n--- ANÁLISE DE PERFORMANCE CONCLUÍDA ---")


if __name__ == "__main__":
    check_performance()
