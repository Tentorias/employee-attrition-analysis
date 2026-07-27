# 🧠 Análise e Predição de Attrition de Colaboradores — Solução de BI & Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%2B%20Optuna-orange.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP%20TreeExplainer-red.svg)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%20REST-00a393.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/DB-PostgreSQL%20%2F%20SQLAlchemy-336791.svg)](https://www.postgresql.org/)

---

Esta solução integra **Machine Learning, Inteligência Analítica e Engenharia de Software** para transformar a gestão de retenção de talentos nas empresas. O sistema antecipa com alta precisão o risco de desligamento voluntário (*turnover* / *attrition*) de colaboradores e prescreve recomendações práticas e causais para intervenção de RH.

O documento abaixo está dividido em duas trilhas independentes e complementares:
1. **[👔 Trilha Executiva & RH](#-trilha-executiva--rh-guia-estratégico-e-operacional)** — Focada no impacto de negócio, interpretação das métricas, diagnósticos individuais e planos de ação de retenção.
2. **[🛠️ Trilha Técnica & MLOps](#-trilha-técnica--mlops-arquitetura-e-engenharia-de-dados)** — Focada na arquitetura à prova de vazamento de dados (*data leakage*), módulo centralizado de engenharia de *features*, otimização, testes e integração contínua.

---

## 👔 Trilha Executiva & RH (Guia Estratégico e Operacional)

### 1. O Desafio de Negócio
O desligamento voluntário de colaboradores geram custos excessivos para as organizações, incluindo perda de capital intelectual, custos de recrutamento, despesas de *onboarding* e queda na produtividade da equipe. Mapear o turnover apenas através de pesquisas de desligamento (*offboarding*) é uma medida **reativa** — quando o colaborador responde, a saída já ocorreu.

### 2. Nossa Proposta de Valor: Atuação Preditiva e Acionável
Nossa plataforma evolui o RH para o modelo **preditivo e prescritivo**:
- **Alerta Precoce:** O modelo avalia diariamente 49 fatores organizacionais, de cargo, compensação e satisfação, calculando uma probabilidade individual de saída.
- **Precisão Confiável (Precision Target):** O algoritmo foi calibrado para priorizar a **Precisão** dos alertas. Quando o sistema aponta um colaborador em zona de risco crítico, há alta garantia de que o perigo é real, evitando gastos com retenções desnecessárias.
- **Explicabilidade Individual (SHAP):** Nenhum alerta é uma "caixa preta". Para cada colaborador, o sistema apresenta quais fatores exatos estão aumentando ou reduzindo seu risco de saída.
- **Recomendações Causais de RH:** Além de apontar o risco, o sistema sugere ações de retenção personalizadas baseadas em efeitos causais (ex.: redução de horas extras, revisão de faixas salariais, programas de desenvolvimento ou rotação de cargo).

---

### 3. Como Utilizar o Dashboard Tático (Streamlit)
O Dashboard de Retenção é a ferramenta diária dos Business Partners (BPs) de RH e gestores:

```
+-------------------------------------------------------------------------+
|                  DASHBOARD TÁTICO DE RETENÇÃO (RH)                      |
+------------------------------------+------------------------------------+
| 📊 ABA 1: VISÃO GERAL              | 🎯 ABA 2: DIAGNÓSTICO INDIVIDUAL   |
|                                    |                                    |
| • Filtro por Departamento / Cargo  | • Colaborador Selecionado          |
| • Tabela Interativa de Risco       | • Risco Atual (Probabilidade %)    |
| • Alertas por Nível (Baixo/Alto)   | • Fatores Críticos Acionáveis      |
| • Busca por ID do Funcionário      | • Recomendações Causais de RH     |
+------------------------------------+------------------------------------+
```

#### **Aba 1 — Visão Geral & Priorização**
- **Ranking de Risco:** Permite visualizar a lista completa de colaboradores ordenada decrescentemente pela probabilidade de desligamento.
- **Filtros por Área:** Possibilita filtrar por departamento (*Sales*, *R&D*, *HR*) e identificar rapidamente quais equipes concentram maior pressão de saída.

#### **Aba 2 — Diagnóstico Individual & Plano de Ação**
- Selecione qualquer colaborador na tabela para inspecionar seu perfil.
- **Fatores Acionáveis (SHAP):** Exibe automaticamente as variáveis sobre as quais o RH tem controle (ex.: quantidade de horas extras, satisfação no trabalho, tempo sem promoção). Variáveis imutáveis (como estado civil ou gênero) são filtradas para não desviar o foco da liderança.
- **Plano Prescritivo:** O sistema gera um resumo contendo recomendações estratégicas com o impacto percentual que cada ajuste poderá ter na redução da probabilidade de saída.

---

### 4. Resumo Executivo das Métricas do Modelo
Para conciliar eficiência orçamentária de RH com sensibilidade na detecção:
- **Acurácia Geral do Sistema:** ~85% de acertos globais em todo o quadro funcional.
- **Foco na Precisão:** O modelo adota um limiar de decisão (*threshold*) calibrado que maximiza o equilíbrio preditivo protegendo a taxa de falso-alarme. Dessa forma, as campanhas de retenção de salários e benefícios são direcionadas de modo certeiro aos talentos reais em risco.

---
---

## 🛠️ Trilha Técnica & MLOps (Arquitetura e Engenharia de Dados)

### 1. Arquitetura Geral e Fluxo de Dados

A solução adota separação estrita de responsabilidades, garantindo consistência entre o treinamento de modelos e a inferência em tempo real:

```
  [CSV / PostgreSQL] --(Raw Data)--> [src/attrition/features.py] (Single Source of Truth)
                                               |
        +--------------------------------------+--------------------------------------+
        |                                      |                                      |
        v                                      v                                      v
 [Treino & Optuna]                     [API REST FastAPI]                   [Dashboard Streamlit]
 (Pipeline SMOTEENN)                   (/predict endpoint)                  (SHAP TreeExplainer)
        |                                      |                                      |
        v                                      v                                      v
  [model.pkl / exp.pkl]                  [JSON Prediction]                    [Gráficos & Insights]
```

---

### 2. Garantias Arquiteturais contra Vazamento de Dados (*Data Leakage*)
Durante a engenharia da solução, eliminamos três categorias críticas de vazamento de dados que comprometem sistemas preditivos em produção:

1. **Vazamento por Identificadores e Colunas Target (`EmployeeNumber` / `Attrition`):**
   - **Solução:** O módulo centralizado `preprocess_employee_data(df)` no arquivo `src/attrition/features.py` é o **único responsável** por preparar dados tanto para o treino quanto para a API/App. Ele realiza o expurgo obrigatório do ID e da coluna alvo, assegurando que o vetor final de atributos possua exatamente **49 features originais e derivadas** (`features.pkl`).
2. **Vazamento na Validação Cruzada com Resampling (`SMOTEENN`):**
   - **Solução:** Técnicas de reamostragem sintética aplicadas antes de realizar o split de treino/teste poluem a validação cruzada. Em nosso pipeline, o `SMOTEENN` e o `XGBClassifier` foram encapsulados em um `imblearn.pipeline.Pipeline` em `src/attrition/models/tunning.py`. O balanceamento ocorre **estritamente dentro dos folds de treino** a cada iteração do Optuna.
3. **Isolamento Completo entre Suítes de Teste e Produção:**
   - **Solução:** Todos os testes de integração em `tests/test_main_cli.py` utilizam diretórios temporários (`tmp_path`) e isolamento de argumentos CLI (`--explainer-path`, `--threshold-output-path`). Executar `pytest tests/` não polui nem sobrescreve os modelos oficiais em `models/` ou `artifacts/`.

---

### 3. Engenharia de Features & Pré-Processamento (`src/attrition/features.py`)
- **Single Source of Truth:** A função `preprocess_employee_data(df, is_training, model_features)` centraliza a tipagem, tratamento de binários (`Yes`/`No`, `Male`/`Female`), cria interações de salário com longevidade (`Income_Longevity_Interaction`), índice de estagnação (`Stagnation_Index`) e gera *One-Hot Encoding* alinhado de forma robusta às 49 features catalogadas em `features.pkl`.

---

### 4. Estrutura do Repositório

```text
.
├── api/
│   └── main.py                   # API REST FastAPI (Inferência Online / Saúde do Modelo)
├── app/
│   └── main_app.py               # Interface Tática Streamlit (Dashboard + Diagnóstico SHAP)
├── artifacts/
│   ├── features/                 # Catálogo oficial de colunas (features.pkl) e CSVs de teste
│   └── models/                   # Parâmetros otimizados (best_params.json) e thresholds (optimal_threshold.pkl)
├── models/
│   ├── production_model.pkl      # XGBClassifier treinado com 100% dos dados pré-processados
│   └── production_shap_explainer.pkl # Explicador TreeExplainer alinhado às 49 features
├── scripts/
│   ├── create_shap_explainer.py  # Script de geração automatizada do SHAP Explainer
│   ├── migrate_to_postgres.py    # Carga de dados brutos para PostgreSQL
│   ├── run_batch_predictions.py  # Inferência em lote com transações seguras no banco
│   └── test_architectural_integrity.py # Diagnóstico e verificação de sanidade da arquitetura
├── src/attrition/
│   ├── features.py               # Módulo central de pré-processamento (Single Source of Truth)
│   ├── data_processing.py        # Adaptador de carregamento de datasets
│   └── models/
│       ├── train.py              # Pipeline de treino, treino de produção e salvamento
│       ├── tunning.py            # Otimização Bayesiana com Optuna + Imbalanced Pipeline
│       └── evaluate.py           # Avaliação de métricas e calibração de Threshold
├── tests/
│   ├── test_main_cli.py          # Testes de integração de CLI sem efeito colateral
│   └── test_unit_functions.py    # Testes unitários do pré-processamento e treino
├── makefile                      # Receitas de automação do projeto
└── pyproject.toml                # Dependências declaradas
```

---

### 5. Como Executar o Projeto Localmente (Sem Dependência de Venv/Poetry)
O projeto é 100% executável via Python 3.12 (`py -3.12 -m`), sem exigir configuração do Poetry em ambientes Windows:

#### **A. Execução de Testes Unitários e Arquiteturais**
```powershell
# 1. Bateria completa de testes (com isolamento garantido de modelos originais)
py -3.12 -m pytest tests/ -v

# 2. Verificação de Sanidade Arquitetural (Anti-Leakage e Inspecção XGBoost/SHAP)
py -3.12 scripts/test_architectural_integrity.py
```

#### **B. Treino e Otimização do Pipeline Completo**
```powershell
# Executar validação, otimização Bayesiana, retreino para produção e geração do explicador SHAP
py -3.12 -m src.attrition.main run-pipeline --tune
```

#### **C. Inicialização das Aplicações (Streamlit & FastAPI)**
```powershell
# 1. Abrir o Dashboard Interativo de Retenção (Streamlit)
py -3.12 -m streamlit run app/main_app.py

# 2. Subir a API REST Operacional com Swagger (FastAPI)
py -3.12 -m uvicorn api.main:app --reload
```
- **Streamlit:** Acessível via `http://localhost:8501/`
- **FastAPI Docs (Swagger):** Acessível via `http://127.0.0.1:8000/docs`

---

## 📜 Licença
Este projeto é distribuído sob os termos da licença [LICENSE](LICENSE) do repositório.