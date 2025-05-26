# src/models/train.py

import os
import argparse
import joblib
import pandas as pd
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

def load_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# … train_model, optimize_threshold …

def save_model(model, thr, model_path: str, thr_path: str):
    joblib.dump(model, model_path)
    joblib.dump(thr, thr_path)

def main():
    parser = argparse.ArgumentParser(
        description="Treina modelo XGBoost e otimiza threshold."
    )
    parser.add_argument(
        "--in-path",
        type=str,
        default=os.path.join(BASE_DIR, "artifacts", "features_matrix.csv"),
        help="CSV com matriz de features."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(BASE_DIR, "artifacts", "models", "xgb_attrition_final.pkl"),
        help="Onde salvar o modelo treinado."
    )
    parser.add_argument(
        "--thr-path",
        type=str,
        default=os.path.join(BASE_DIR, "artifacts", "models", "threshold_optimizado.pkl"),
        help="Onde salvar o threshold otimizado."
    )
    args = parser.parse_args()

    logger.info(f"🔄 Carregando features de {args.in_path}")
    df = load_features(args.in_path)
    X = df.drop("Attrition", axis=1);  y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("🔧 Treinando XGBoost com SMOTE")
    model = train_model(X_train, y_train)

    logger.info("🔍 Otimizando threshold")
    thr, f1 = optimize_threshold(model, X_test, y_test)

    logger.info(f"💾 Salvando modelo em {args.model_path} e threshold em {args.thr_path}")
    save_model(model, thr, args.model_path, args.thr_path)
    logger.info(f"✅ Modelo salvo com F1={f1:.3f} e threshold={thr:.2f}")

if __name__ == "__main__":
    main()
