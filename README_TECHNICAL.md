# 🛠️ Documentação Técnica & Arquitetura MLOps (`README_TECHNICAL.md`)

Este documento apresenta a especificação técnica detalhada, a arquitetura de engenharia de *features*, as garantias anti-vazamento de dados (*data leakage*) e os comandos de automação da solução **Employee Attrition Analysis**.

---

## 1. Arquitetura Geral & Single Source of Truth

Para evitar divergências de processamento (*training-serving skew*) entre o pipeline de treino offline, a API operacional em tempo real e a interface analítica, o projeto implementa o padrão **Single Source of Truth** na engenharia de atributos:

```
  [CSV / PostgreSQL Raw] ----> [src/attrition/features.py] (preprocess_employee_data)
                                           |
         +---------------------------------+---------------------------------+
         |                                 |                                 |
         v                                 v                                 v
  [Pipeline de Treino]           [API REST FastAPI]               [Streamlit Tático]
  (Optuna + SMOTEENN)            (/predict endpoint)              (SHAP TreeExplainer)
         |                                 |                                 |
         v                                 v                                 v
  [production_model.pkl]         [JSON Prediction / DB]            [Gráficos & Causalidade]
```

- **Módulo Central (`src/attrition/features.py`)**:
  - A função `preprocess_employee_data(df, is_training=False, model_features=None)` é chamada obrigatoriamente por todos os consumidores: script de treino, CLI de batch prediction, API FastAPI e Streamlit.
  - Responsável pela remoção de IDs (`EmployeeNumber`) e target (`Attrition`), conversão booleana, criação de features derivadas (`Income_Longevity_Interaction`, `Stagnation_Index`) e *One-Hot Encoding* compatível com `pandas 2/3`.
  - Garante que todo DataFrame processado resulte em exatamente **49 features originais e derivadas**, alinhadas ao catálogo `artifacts/features/features.pkl`.

---

## 2. Blindagem contra Data Leakage (3 Níveis de Defesa)

Vazamentos de dados são a causa número 1 de modelos que apresentam alta performance em validação mas falham em produção. Este projeto implementa três camadas de blindagem:

### **Nível 1: Exclusão de Identificadores (`EmployeeNumber`)**
- **O Risco:** Identificadores numéricos crescentes correlacionam-se acidentalmente com a antiguidade ou com a ordem de registro na base de dados.
- **A Solução:** `EmployeeNumber` é explicitamente expurgado em `preprocess_employee_data` antes da modelagem. O Streamlit foi refatorado para manter o alinhamento da interface visual através do índice posicional (`selected_indices[0]`), sem injetar IDs no vetor do modelo.

### **Nível 2: Pipeline de Validação Cruzada com SMOTEENN (`tunning.py`)**
- **O Risco:** Aplicar reamostragem sintética (`SMOTEENN`) em todo o dataset antes da validação cruzada contamina os folds de validação com dados sintéticos gerados a partir de amostras de teste.
- **A Solução:** O reamostrador foi encapsulado dentro do pipeline de estimadores utilizando `imblearn.pipeline.Pipeline`:
  ```python
  from imblearn.pipeline import Pipeline
  from imblearn.combine import SMOTEENN

  model_pipeline = Pipeline([
      ("smoteenn", SMOTEENN(random_state=42)),
      ("xgb", XGBClassifier(**params))
  ])
  ```
  Isso garante que a sintese de dados minoritários ocorra **estritamente no interior dos folds de treino** de cada iteração do `StratifiedKFold`.

### **Nível 3: Isolamento de Artefatos na Suíte de Testes CLI**
- **O Risco:** Testes de integração de linha de comando (`pytest`) costumam gerar modelos temporários (*toy models*) que acidentalmente sobrescrevem arquivos reais em `models/` ou `artifacts/`.
- **A Solução:** Em `tests/test_main_cli.py`, os caminhos `--explainer-path` e `--threshold-output-path` foram isolados no diretório temporário do `pytest` (`tmp_path`). A execução de testes nunca polui ou altera os artefatos oficiais de produção.

---

## 3. Diagnóstico e Resolução do SHAP (`50 vs. 9 features`)

Durante a integração, foi documentado e resolvido o erro clássico de dimensionalidade:
```text
xgboost._c_api.XGBoostError: Check failed: static_cast<bst_ulong>(1) == chunksize * rows (50 vs. 9)
```
- **Causa Raiz:** O modelo em produção possui 49 features (+ 1 bias = 50 saídas SHAP). No entanto, testes automatizados mal isolados haviam sobrescrito o arquivo `models/production_shap_explainer.pkl` com um explicador treinado em um modelo de teste de 8 features (+ 1 bias = 9).
- **Resolução:**
  1. Parametrizou-se a gravação de artefatos em `evaluate.py`.
  2. Regenerou-se o explicador SHAP oficial com o modelo legítimo de 49 features.
  3. Adicionou-se validação em `tests/test_main_cli.py` para garantir alinhamento de dimensões sem efeito colateral.

---

## 4. Banco de Dados & Transações (PostgreSQL + SQLAlchemy 2.0)

O script `scripts/run_batch_predictions.py` executa inferências em lote na tabela de funcionários e armazena o histórico em PostgreSQL.
- Utiliza **SQLAlchemy 2.0+** com transações contextuais explícitas:
  ```python
  with engine.begin() as connection:
      connection.execute(text("TRUNCATE TABLE employee_attrition_predictions;"))
  ```
- Evita *deadlocks* ou transações pendentes em bancos de dados em nuvem ou locais.

---

## 5. Estrutura e Rotas da API REST (FastAPI)

A API (`api/main.py`) provê endpoints operacionais assíncronos:
- `GET /`: Healthcheck e metadados de carregamento do modelo.
- `POST /predict`: Recebe um objeto JSON com os atributos do colaborador, processa via `preprocess_employee_data`, calcula a probabilidade e retorna:
  - `prediction`: `0` (Permanece) ou `1` (Risco de Saída).
  - `probability`: Score contínuo calibrado (0.0 a 1.0).
  - `risk_level`: `Baixo Risco`, `Risco Moderado`, `Alto Risco` ou `Risco Crítico`.
- **Compatibilidade Windows:** Saídas no console da API utilizam formatação ASCII pura (`[OK]`, `[AVISO]`, `[ERRO]`), prevenindo exceções `UnicodeEncodeError` em consoles Windows (`cp1252`).

---

## 6. Guia de Execução e Comandos CLI (`py -3.12`)

O projeto foi configurado para execução nativa via Python 3.12, dispensando dependências exclusivas de venv ou poetry:

### **A. Bateria de Testes Automatizados**
```powershell
# Executar todos os testes unitários e de integração (0 warnings)
py -3.12 -m pytest tests/ -v
```

### **B. Verificação de Sanidade da Arquitetura**
```powershell
# Verifica anti-leakage, compatibilidade XGBoost e alinhamento do arquivo de features
py -3.12 scripts/test_architectural_integrity.py
```

### **C. Retreino e Otimização Bayesiana do Pipeline**
```powershell
# Executar pré-processamento, 100 trials Optuna, treino do modelo e criação do explainer SHAP
py -3.12 -m src.attrition.main run-pipeline --tune
```

### **D. Subir Aplicações**
```powershell
# Dashboard Interativo Streamlit (porta 8501)
py -3.12 -m streamlit run app/main_app.py

# API Operacional FastAPI (porta 8000 / Swagger em /docs)
py -3.12 -m uvicorn api.main:app --reload
```
