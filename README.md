
# 🧠 Employee Attrition Analysis — Solução de BI & Machine Learning

Projeto de análise e predição de rotatividade de funcionários.  
A solução evolui de um pipeline de Machine Learning para uma arquitetura completa de Business Intelligence + ML, com camadas estratégicas (Power BI), táticas (Streamlit) e operacionais (API REST) para apoiar decisões no setor de RH.

---

## 🚀 API de Predição (Deploy Público)

A API de predição está disponível para testes:

**Documentação:**  
https://employee-attrition-analysis.onrender.com/docs

> ⚠️ *Atenção:* A API usa o plano gratuito da Render e pode levar até 60 segundos para responder à primeira requisição após inatividade.

**Exemplo de requisição:**
```bash
curl -X 'POST'   'https://employee-attrition-analysis.onrender.com/predict'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{ ... }'
```

---

## 🏛️ Arquitetura da Solução

**Fonte Única da Verdade:**  
`hr_analytics.db` (SQLite) → Dados brutos, tratados e predições.

### Camadas:
- **Estratégica (Power BI):** Dashboards e KPIs de turnover para alta gestão.
- **Tática (Streamlit):** Diagnóstico individual e ranking de risco para gestores e RH.
- **Operacional (API REST):** Serviço automatizado de predição para outros sistemas.

---

## 📊 Camada Estratégica — Power BI

- **Objetivo:** Analisar KPIs de turnover e saúde organizacional.
- **Ferramentas:** SQL + Power BI.
- **Público-Alvo:** Diretoria, C-Level, Heads de RH.
- **Pergunta típica:**  
  *"Qual departamento tem maior risco e qual o impacto financeiro?"*

---

## 🎯 Camada Tática — Diagnóstico Individual (Streamlit)

- **Objetivo:** Identificar risco individual e causas.
- **Ferramentas:** Streamlit + Python.
- **Público-Alvo:** Gestores e Analistas de RH.
- **Pergunta típica:**  
  *"Quais fatores explicam o risco de saída do João?"*

---

## 🔬 Camada Operacional — API REST (FastAPI)

- **Objetivo:** Disponibilizar predições para automação.
- **Ferramentas:** FastAPI.
- **Público-Alvo:** Times de dados e sistemas internos.
- **Uso:** Consumo por APIs externas, apps ou scripts internos.

---

## ⚙️ Stack Tecnológica

### **Dados & BI**
- SQLite, SQL, Power BI

### **Modelagem & Core**
- Python 3.10+
- Pandas, NumPy
- Scikit-learn, LightGBM, XGBoost
- SMOTEENN, Optuna

### **Visualização & Apps**
- Streamlit, SHAP, Matplotlib, Seaborn, Jupyter Notebook, FastAPI

### **Dev & MLOps**
- Poetry, Git, Git LFS
- Docker, Render
- GitHub Actions, Pre-commit, Black, isort, Flake8, Pytest

---

## 📁 Estrutura do Projeto

```bash
employee-attrition-analysis/
├── api/                   # API REST
├── app/                   # App Streamlit
├── artifacts/             # Modelos e explicadores
├── data/                  # Dados brutos e tratados
├── database/              # Banco de dados SQLite
├── models/                # Modelos finais
├── notebooks/             # EDA e explorações
├── reports/               # Dashboards Power BI
├── scripts/               # Scripts de suporte
├── sql/                   # Queries SQL
├── src/                   # Pipeline principal
├── tests/                 # Testes automatizados
├── Dockerfile             
├── .gitignore             
├── .gitattributes         
├── pyproject.toml         
├── poetry.lock            
└── pre-commit-config.yaml 
```

---

## 🚀 Guia de Uso (Local)

### **Pré-requisitos:**
- Python 3.10+
- Poetry
- Git + Git LFS

### **Instalação:**
```bash
git clone https://github.com/Tentorias/employee-attrition-analysis.git
cd employee-attrition-analysis
poetry install
```

### **Execução:**

**1. Criar banco de dados:**
```bash
poetry run python scripts/load_raw_to_db.py
```

**2. Treinar ou re-treinar modelo:**
```bash
poetry run python src/attrition/main.py [comando]
```

**3. Gerar explicador SHAP:**
```bash
poetry run python scripts/create_shap_explainer.py
```

**4. Fazer predições em lote:**
```bash
poetry run python scripts/generate_predictions.py
```

**5. Visualizar análises:**
- Power BI: abrir `reports/dashboard.pbix` e clicar em "Atualizar".
- Streamlit:
```bash
poetry run streamlit run app/main_app.py
```

**6. Avaliação detalhada do modelo (opcional):**
```bash
poetry run python scripts/evaluate_model_deeply.py
```

---

## 🔗 Pipeline de Machine Learning

- **Pré-processamento:** Limpeza, transformações logarítmicas.
- **Engenharia de Features:** Variáveis derivadas, One-Hot Encoding.
- **Balanceamento:** SMOTEENN.
- **Otimização:** Optuna.
- **Modelagem:** XGBoost.
- **Explicabilidade:** SHAP.
- **Avaliação:** F1, Recall, Precision, AUC.

---

## 📊 Resultados do Modelo (Produção)

| Modelo              | Precision (Yes) | Recall (Yes) | F1-Score (Yes) | AUC  |
|---------------------|-----------------|--------------|----------------|------|
| Logistic Regression | 0.70            | 0.34         | 0.46           | -    |
| **XGBoost (Prod)**  | 0.54            | 0.66         | 0.60           | 0.87 |
| LightGBM            | 0.65            | 0.28         | 0.39           | 0.83 |

- **Recall de 66%:** Detecta 2 em cada 3 saídas.
- **AUC 0.87:** Excelente separação entre classes.
- Foco em maximizar recall, dado o alto custo de perder talentos.

---

## 📦 Dataset

- **Fonte:** [IBM HR Analytics (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1.470 registros
- 35 features
- Target: `Attrition` (Yes/No)

---

## Considerações Éticas

Este modelo não toma decisões nem classifica funcionários. Ele apenas identifica padrões relacionados à intenção de saída e apresenta insights. Cabe exclusivamente ao RH interpretar e agir.

O risco não está no modelo, mas na inação diante dos sinais.