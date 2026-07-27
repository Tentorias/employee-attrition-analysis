# 📋 TODO List & Roteiro de Refatoração — `employee-attrition-analysis`

Este documento serve como um roteiro prático e acompanhamento de tarefas para refatorarmos o projeto e torná-lo **100% funcional, limpo e à prova de Data Leakage**.

---

## 🗺️ Status Geral das Tarefas

- [x] **Etapa 1:** Centralizar o Pré-Processamento (Single Source of Truth)
- [x] **Etapa 2:** Remover Data Leakage de IDs no Modelo (`EmployeeNumber`)
- [x] **Etapa 3:** Corrigir Data Leakage na Otimização (Optuna + SMOTEENN)
- [x] **Etapa 4:** Corrigir Erros Críticos de Script (`run_batch_predictions.py`)
- [x] **Etapa 5:** Limpar Duplicações e Ajustar Automação (`Makefile` e `seed_database.py`)
- [x] **Etapa 6:** Validar Todos os Testes Automatizados e Integridade

---

## 🛠️ Detalhamento por Etapa

### [x] Etapa 1: Centralizar o Pré-Processamento (Single Source of Truth)
- [x] Criar o módulo `src/attrition/features.py` com a função `preprocess_employee_data(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame`.
- [x] Incluir em `preprocess_employee_data`:
  - Remoção consistente de colunas não preditivas (`EmployeeCount`, `Over18`, `StandardHours` e `EmployeeNumber` na modelagem).
  - Mapeamento numérico (`Gender`).
  - Criação de `YearsPerCompany`, `MonthlyIncome_log` e `TotalWorkingYears_log`.
  - Criação das features interativas (`Income_Longevity_Interaction` e `Stagnation_Index`).
  - One-Hot Encoding (`pd.get_dummies`) consistente.
- [x] Refatorar `src/attrition/models/train.py` para usar `preprocess_employee_data`.
- [x] Refatorar `src/attrition/data_processing.py` para usar `preprocess_employee_data` ao gerar `df_model`.
- [x] Refatorar `api/main.py` (endpoint `/predict`) para usar `preprocess_employee_data`.
- [x] Refatorar `src/attrition/models/predict.py` (CLI) para usar `preprocess_employee_data`.

---

### [x] Etapa 2: Remover Data Leakage de IDs no Modelo (`EmployeeNumber`)
- [x] Em `train.py`, garantir que a coluna `EmployeeNumber` seja excluída explicitamente de `X` antes do fit do XGBoost e da validação cruzada.
- [x] Verificar que o `features.pkl` salvo não contenha `EmployeeNumber`.

---

### [x] Etapa 3: Corrigir Data Leakage na Otimização (Optuna + SMOTEENN)
- [x] Em `src/attrition/models/tunning.py`, encapsular o resampling e o modelo num pipeline:
  ```python
  from imblearn.pipeline import Pipeline
  pipeline = Pipeline([
      ('smoteenn', SMOTEENN(random_state=42)),
      ('xgb', xgb.XGBClassifier(**params))
  ])
  ```
- [x] Rodar o `cross_val_score(pipeline, X_train, y_train, ...)` passando os dados de treino **antes** de qualquer resampling global.
- [x] Reduzir o espaço de busca de hiperparâmetros (`max_depth` até 6, `n_estimators` até 500) para evitar overfitting em ~1.400 linhas.

---

### [x] Etapa 4: Corrigir Erros Críticos de Script (`run_batch_predictions.py`)
- [x] Em `scripts/run_batch_predictions.py` (Linha 37):
  - Substituir `requests.text(...)` por `sqlalchemy.text("TRUNCATE TABLE predictions RESTART IDENTITY;")`.
  - Adicionar `from sqlalchemy import text` nos imports.

---

### [x] Etapa 5: Limpar Duplicações e Ajustar Automação
- [x] Deletar o script redundante `scripts/seed_database.py` que causava divergência em minúsculas/maiúsculas.
- [x] Atualizar o `makefile`:
  - Na receita `create-explainer`, apontar para `scripts/create_shap_explainer.py` (que suporta argumentos via CLI).
- [x] Atualizar testes unitários em `tests/test_unit_functions.py` para testar `preprocess_employee_data`.

---

### [x] Etapa 6: Validação Final e Testes
- [x] Rodar `py -3.12 -m pytest tests/ -v` para certificar que os testes unitários passam.
- [x] Rodar `py -3.12 scripts/test_architectural_integrity.py` e verificar aprovação 100%.
- [x] Testar `py -3.12 src/attrition/main.py run-pipeline` ponta a ponta.
- [x] Testar `py -3.12 scripts/run_batch_predictions.py` via banco e API.
