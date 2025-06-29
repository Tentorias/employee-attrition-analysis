# 🧠 Employee Attrition Analysis — BI & ML Solution

Análise e predição da rotatividade de funcionários. O projeto evolui de um pipeline puro de Machine Learning para uma solução completa de Business Intelligence + ML, com recursos estratégicos (Power BI) e táticos (Streamlit) para apoiar decisões no setor de RH.

---

## 🏛️ Arquitetura da Solução

A solução possui duas camadas complementares, alimentadas por uma fonte de dados central:

### 🔗 Fonte Única da Verdade
**SQLite Database**: `hr_analytics.db`  
Centraliza dados brutos, processados e predições.

---

## 📈 Camada Estratégica — Visão para a Liderança

- **Objetivo**: Diagnosticar saúde organizacional e monitorar KPIs  
- **Ferramentas**: SQL + Power BI  
- **Público-Alvo**: Diretoria, C-Level, Head de RH  
- **Exemplo de Pergunta**:  
  “Qual departamento tem maior turnover e qual o impacto financeiro disso?”

---

## 🚀 Camada Tática & Preditiva — Apoio à Decisão

- **Objetivo**: Diagnosticar risco individual, causas e simular retenção  
- **Ferramentas**: Python + Streamlit  
- **Público-Alvo**: Gestores, Analistas de RH  
- **Exemplo de Pergunta**:  
  “Quais fatores influenciam a saída do João? Se eu der um aumento, qual o novo risco?”

---

## 🎯 Objetivos do Projeto

- Identificar colaboradores com alto risco de saída via ML  
- Diagnosticar causas com SHAP (XAI)  
- Oferecer simulações "what-if" via Streamlit  
- Criar dashboards estratégicos com Power BI  

---

## 🛠️ Stack Tecnológica

### 📊 Dados & BI
- SQLite  
- SQL  
- Power BI  

### ⚙️ Core & Modelagem
- Python 3.10+  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- SMOTEENN  
- Optuna  

### 🖼️ Visualização & Aplicação
- Matplotlib, Seaborn  
- SHAP  
- Streamlit  
- Jupyter Notebook  

### ⚙️ Dev & MLOps
- Poetry  
- Git & Git LFS  
- Pytest  
- Pre-commit, Black, isort, Flake8  
- GitHub Actions  

---

## 📁 Estrutura do Projeto

```
employee-attrition-analysis/
├── app/                                # Streamlit app
├── artifacts/                          # Modelos e artefatos
├── data/                               # Dados brutos e tratados
├── database/                           # hr_analytics.db
├── models/                             # Modelo final
├── notebooks/                          # EDA
├── reports/                            # Power BI e gráficos
├── scripts/                            # Scripts auxiliares
├── sql/                                # Consultas SQL
├── src/                                # Código do pipeline
├── tests/                              # Testes automatizados
├── .gitignore
├── .gitattributes
├── pre-commit-config.yaml
├── pyproject.toml
└── poetry.lock
```

---

## 🚀 Guia de Uso

### 🔧 Pré-requisitos
- Python 3.10+  
- Poetry  
- Git + Git LFS

### ⚙️ Instalação

```bash
git clone https://github.com/Tentorias/employee-attrition-analysis.git
cd employee-attrition-analysis
poetry install
```

### ▶️ Execução

**1. Criar a Base de Dados:**

```bash
poetry run python scripts/load_raw_to_db.py
```

**2. Executar o Pipeline de ML:** (treinar/retreinar):

```bash
poetry run python src/attrition/main.py [comando]
```

**3. Gerar explicador SHAP:**
```bash
poetry run python scripts/create_shap_explainer.py
```

**4. Gerar Predições em Massa:**

```bash
poetry run python scripts/generate_predictions.py
```

**5. Visualizar as Análises:**

- **Power BI**: Abrir `reports/dashboard.pbix` e clicar em "Atualizar".
- **Streamlit**:

```bash
poetry run streamlit run app/main_app.py
```

**6. Validar performance do modelo (opcional):**

```bash
poetry run python scripts/evaluate_model_deeply.py
```

---

## 📊 Pipeline de ML

- **Processamento**: Limpeza de dados, transformações logarítmicas.
- **Engenharia de Features**: Criação de variáveis derivadas e One-Hot.
- **Balanceamento**: SMOTEENN
- **Otimização**: Optuna
- **Modelagem**: XGBoost
- **Avaliação**: F1-score, Precision, Recall, SHAP

---

## 📊 Resultados do Modelo Final

```
| Modelo              | Precisão (Yes) | Recall (Yes) | F1-Score (Yes) | AUC  |
| ------------------- | -------------- | ------------ | -------------- | ---- |
| Regressão Logística | 0.70           | 0.34         | 0.46           | -    |
| XGBoost (Produção)  | 0.54           | 0.66         | 0.60           | 0.87 |
```

- Recall 66%: identifica 2/3 funcionários que sairão (foco no custo de erro).

- AUC 0.87: excelente separação entre classes.

- Supera baseline em +32 pontos no recall.

- **Algoritmo**: XGBoost Classifier
- **Técnica de balanceamento**: SMOTEENN
- **F1-Score (Classe "Yes")**: \~0.53

Este F1-Score reflete uma estratégia que prioriza a capacidade de detectar verdadeiros positivos, mesmo com a classe "Yes" sendo minoritária (\~16%).

💼 Impacto no Negócio
- **Power BI**: Identifica áreas críticas com maior risco

- **Streamlit**: Permite análise e ranking por equipe

- **SHAP**: Diagnóstico individual instantâneo

- **Simulações**: Testa impacto de ações (ex: aumento salarial) no risco


---

## 📦 Dataset

- Fonte: [IBM HR Analytics Employee Attrition Dataset (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1.470 registros
- 35 features
- Target: `Attrition` (Yes/No)

---