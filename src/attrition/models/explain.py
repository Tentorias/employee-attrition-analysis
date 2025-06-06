import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


def main(model_path, data_path):
    model = joblib.load(model_path)
    X = pd.read_csv(data_path)
    feature_cols = joblib.load(
        "models/features_columns.pkl"
    )  # ajuste o caminho se necessário
    X = X[feature_cols]  # garante ordem e colunas idênticas

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary_plot.png")
