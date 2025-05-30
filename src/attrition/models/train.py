# src/attrition/models/train.py

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Configura logs
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Raiz do projeto (project_root)
BASE_DIR = Path(__file__).resolve().parents[3]


def load_features(path: Path) -> pd.DataFrame:
    """
    Carrega a matriz de features (CSV) para treinamento.
    """
    return pd.read_csv(path)


def train_model(X, y, random_state: int = 42) -> XGBClassifier:
    """
    Balanceia classes com SMOTE e treina um XGBClassifier.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=random_state
    )
    model.fit(X_res, y_res)
    return model


def optimize_threshold(model: XGBClassifier, X_test, y_test) -> tuple[float, float]:
    """
    Busca threshold ótimo para maximizar F1-score.
    Retorna (melhor_threshold, melhor_f1).
    """
    probs = model.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = 0.5, 0.0
    for thr in (i / 100 for i in range(30, 71, 5)):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr, best_f1


def save_artifacts(model: XGBClassifier, threshold: float, model_path: Path, thr_path: Path) -> None:
    """
    Salva o modelo e o threshold em disco.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    thr_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(threshold, thr_path)


def main():
    parser = argparse.ArgumentParser(
        description="Treina o modelo XGBoost e otimiza o threshold de decisão."
    )
    parser.add_argument(
        "--in-path",
        type=Path,
        default=BASE_DIR / "artifacts" / "features_matrix.csv",
        help="Caminho para o CSV de features."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=BASE_DIR / "artifacts" / "models" / "xgb_attrition_final.pkl",
        help="Onde salvar o modelo treinado."
    )
    parser.add_argument(
        "--thr-path",
        type=Path,
        default=BASE_DIR / "artifacts" / "models" / "threshold_optimizado.pkl",
        help="Onde salvar o threshold otimizado."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporção do conjunto de teste."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed para reproducibilidade."
    )
    args = parser.parse_args()

    logger.info(f"🔄 Carregando features de {args.in_path}")
    df = load_features(args.in_path)

    # Inspeção de tipos
    logger.info(
        "Tipos de colunas após load_features: %s",
        df.dtypes.to_dict()
    )

    # Define X e y
    if "Attrition_Yes" in df.columns:
        y = df["Attrition_Yes"].astype(int)
        X = df.drop("Attrition_Yes", axis=1)
    else:
        y = df["Attrition"].astype(int)
        X = df.drop("Attrition", axis=1)

    logger.info("🔄 Split e balanceamento com SMOTE")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    logger.info("🔧 Treinando XGBClassifier")
    model = train_model(X_train, y_train, random_state=args.random_state)

    logger.info("🎯 Otimizando threshold")
    best_thr, best_f1 = optimize_threshold(model, X_test, y_test)

    logger.info(
        f"💾 Salvando modelo em {args.model_path} e threshold em {args.thr_path}"
    )
    save_artifacts(model, best_thr, args.model_path, args.thr_path)

    logger.info(f"✅ Treino concluído — F1({best_thr:.2f}) = {best_f1:.3f}")


if __name__ == "__main__":
    main()