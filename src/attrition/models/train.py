# src/attrition/models/train.py (VERSÃO LIMPA)
import argparse
import logging
import os
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def main(in_path, features_path, model_path, target_col="Attrition", x_test_out=None, y_test_out=None, test_size=0.2, random_state=42, retrain_full_data=False):
    df = pd.read_csv(in_path)
    features = joblib.load(features_path)
    X = df[features]
    y = df[target_col]

    model = Pipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("classifier", XGBClassifier(random_state=random_state, n_jobs=-1))
    ])

    if retrain_full_data:
        logging.info("Retreinando o modelo com todos os dados...")
        model.fit(X, y)
        ensure_dir(model_path)
        joblib.dump(model, model_path)
        logging.info(f"Modelo final salvo em: {model_path}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        model.fit(X_train, y_train)
        ensure_dir(model_path)
        joblib.dump(model, model_path)
        logging.info(f"Modelo treinado salvo em: {model_path}")

        if x_test_out and y_test_out:
            ensure_dir(x_test_out)
            X_test.to_csv(x_test_out, index=False)
            ensure_dir(y_test_out)
            y_test.to_csv(y_test_out, index=False)
            logging.info(f"Dados de teste salvos em: {x_test_out} e {y_test_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Adicione os argumentos do parser aqui se precisar executar este arquivo diretamente
    args = parser.parse_args()
    main(**vars(args))