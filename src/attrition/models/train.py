# src/attrition/models/train.py (VERSÃO FINAL DE PRODUÇÃO)

import argparse
import os

import joblib
import pandas as pd
from imblearn.combine import SMOTEENN  # <-- MUDANÇA 1: Importa o SMOTEENN
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_model(X, y, random_state: int = 42) -> XGBClassifier:
    """
    Balanceia classes com SMOTEENN e treina um XGBClassifier com os melhores
    parâmetros de regularização encontrados via Optuna.
    """
    print("Balanceando e limpando dados de treino com SMOTEENN...")
    try:
        # <-- MUDANÇA 2: Usa o SMOTEENN em vez do SMOTE
        balancer = SMOTEENN(random_state=random_state)
        X_res, y_res = balancer.fit_resample(X, y)
    except Exception as e:
        print(f"Não foi possível aplicar SMOTEENN. Usando dados originais. Erro: {e}")
        X_res, y_res = X, y

    # <-- MUDANÇA 3: Usa os parâmetros focados em reduzir o overfitting
    best_params = {
        "n_estimators": 690,
        "max_depth": 4,
        "learning_rate": 0.08062385807713071,
        "subsample": 0.6700765286740634,
        "colsample_bytree": 0.64133652875517,
        "gamma": 3.16552819501044,
        "reg_alpha": 5.287390539786665,
        "reg_lambda": 4.83443268564372,
    }
    print("Treinando modelo XGBoost com parâmetros finais e regularizados...")

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        **best_params,
    )

    model.fit(X_res, y_res)
    return model


def optimize_threshold(model, X_test, y_test):
    """Busca threshold ótimo para maximizar F1-score."""
    print("Otimizando threshold de classificação...")
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        preds = model.predict(X_test)
        return 0.5, f1_score(y_test, preds)

    best_thr, best_f1 = 0.5, 0.0
    for thr in [i / 100 for i in range(10, 81, 2)]:
        preds = (probs >= thr).astype(int)
        score = f1_score(y_test, preds)
        if score > best_f1:
            best_thr, best_f1 = thr, score
    return best_thr, best_f1


def main(
    in_path: str,
    features_path: str,
    model_path: str,
    threshold_path: str = None,
    target_col: str = "Attrition_Yes",
    test_size: float = 0.2,
    random_state: int = 42,
    x_test_out: str = None,
    y_test_out: str = None,
    retrain_full_data: bool = False,
):
    """Fluxo principal: opcionalmente retreina com todos os dados."""
    print("--- Iniciando Pipeline de Treinamento ---")
    df = pd.read_csv(in_path)
    features = joblib.load(features_path)
    X = df[features]
    y = df[target_col]

    if retrain_full_data:
        print("MODO RETREINO: Usando 100% dos dados para treinar o modelo final.")
        model = train_model(X, y, random_state)
        print("Salvando modelo final de produção...")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(model, model_path)
        print(f"\n✅ Modelo de produção salvo em: {model_path}")

    else:
        print(
            f"MODO PADRÃO: Dividindo dados ({1-test_size:.0%} treino / {test_size:.0%} teste)."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        model = train_model(X_train, y_train, random_state)
        best_thr, best_f1 = optimize_threshold(model, X_test, y_test)

        print("Salvando artefatos de treino...")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(model, model_path)

        if threshold_path:
            os.makedirs(os.path.dirname(threshold_path) or ".", exist_ok=True)
            joblib.dump(best_thr, threshold_path)

        if x_test_out:
            X_test.to_csv(x_test_out, index=False)
        if y_test_out:
            y_test.to_csv(y_test_out, index=False)

        print("\n--- Pipeline de Treinamento Concluído ---")
        print(f"✅ F1-Score (Teste com threshold {best_thr:.2f}) = {best_f1:.4f}")


# As funções parse_args e cli_main permanecem as mesmas
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Treina XGBClassifier e otimiza threshold."
    )
    parser.add_argument("--in-path", required=True, help="CSV com features_matrix")
    parser.add_argument(
        "--features-path", required=True, help="Pickle com lista de features X"
    )
    parser.add_argument(
        "--model-path", required=True, help="Onde salvar o modelo (.pkl)"
    )
    parser.add_argument(
        "--threshold-path", help="Onde salvar threshold otimizado (.pkl)"
    )
    parser.add_argument(
        "--target-col", default="Attrition_Yes", help="Nome da coluna alvo"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--x-test-out", type=str, help="Onde salvar X_test.csv")
    parser.add_argument("--y-test-out", type=str, help="Onde salvar y_test.csv")
    parser.add_argument(
        "--retrain-full-data",
        action="store_true",
        help="Pula o split e treina com todos os dados.",
    )
    return parser.parse_args(args)


def cli_main():
    args = parse_args()
    main(
        in_path=args.data_path,
        features_path=args.features_path,
        model_path=args.model_path,
        threshold_path=args.thr_path,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
        x_test_out=args.x_test_out,
        y_test_out=args.y_test_out,
        retrain_full_data=args.retrain_full_data,
    )


if __name__ == "__main__":
    cli_main()
