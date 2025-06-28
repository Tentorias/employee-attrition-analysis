# 🧠 Solução de BI e ML para Análise e Predição de Attrition

Análise e predição da rotatividade de funcionários (employee attrition). Este projeto evoluiu de um pipeline de Machine Learning para uma solução completa de Business Intelligence e ML, implementando desde a análise exploratória e um modelo de produção otimizado até um ecossistema com ferramentas estratégicas (Power BI) e táticas (Streamlit) para apoiar o RH na retenção de talentos.

---

## 🏛️ Arquitetura da Solução

A solução final é dividida em duas camadas complementares que se alimentam de uma fonte de dados central, cada uma com um público e propósito distintos.

### Fonte Única da Verdade

- **SQLite Database (**``**)**: Centraliza todos os dados brutos, processados e, mais importante, os resultados das predições do modelo.

### As Duas Camadas de Análise

#### 📈 Camada Estratégica (Visão para a Liderança)

- **Propósito**: Diagnosticar a saúde da organização e monitorar KPIs de alto nível. Responde "O quê?" e "Onde?".
- **Ferramentas**: SQL + Power BI
- **Público**: Diretoria, C-Level, Head de RH
- **Exemplo de Pergunta**: "Qual departamento tem a maior taxa de turnover e qual o impacto financeiro disso para a empresa?"

#### 🚀 Camada Tática e Preditiva (Apoio à Decisão)

- **Propósito**: Analisar casos individuais, simular cenários e agir proativamente. Responde "E se?".
- **Ferramentas**: ML (Python) + Streamlit
- **Público**: Gestores, Analistas de RH
- **Exemplo de Pergunta**: "Qual a probabilidade do funcionário João sair e como podemos diminuir esse risco?"

---

## 🎯 Objetivos

- Identificar funcionários com alto risco de desligamento através de um modelo preditivo.
- Compreender os principais fatores que influenciam a rotatividade com técnicas de XAI.
- Fornecer uma ferramenta interativa (Streamlit) para simulações "what-if".
- Prover um dashboard executivo (Power BI) para o monitoramento dos KPIs.

---

## 🛠️ Stack Tecnológica

### Dados & BI

- SQLite
- Power BI
- SQL

### Core & Modelagem

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTEENN)
- Optuna

### Visualização & Aplicação

- Matplotlib, Seaborn
- SHAP
- Streamlit
- Jupyter Notebook

### Desenvolvimento & MLOps

- Poetry
- Git & Git LFS
- Pytest
- Pre-commit, Black, isort, Flake8
- GitHub Actions

---

## 📁 Estrutura do Projeto

```
employee-attrition-analysis/
├── app/                  # Código da aplicação Streamlit
├── artifacts/            # Saídas do pipeline (modelos, features, etc.)
├── data/                 # Dados brutos e processados
├── database/             # Banco de dados centralizado (hr_analytics.db)
├── models/               # Modelo final
├── notebooks/            # Análise exploratória
├── reports/              # Dashboard Power BI (.pbix)
├── scripts/              # Scripts de automação
├── sql/                  # Queries SQL para BI
├── src/                  # Código-fonte do pipeline de ML
├── tests/                # Testes automatizados
├── .gitignore
├── .gitattributes
├── pre-commit-config.yaml
├── pyproject.toml
└── poetry.lock
```

---

## 🚀 Guia de Uso

### ⚡️ Pré-requisitos

- Python 3.10+
- Poetry instalado
- Git e Git LFS

### 🔧 Instalação

```bash
git clone https://github.com/Tentorias/employee-attrition-analysis.git
cd employee-attrition-analysis
poetry install
```

### ⚙️ Fluxo de Execução

**1. Criar a Base de Dados**

```bash
poetry run python scripts/load_raw_to_db.py
```

**2. Executar o Pipeline de ML** (Se necessário retreinar)

```bash
poetry run python src/attrition/main.py [comando]
```

**3. Gerar Predições em Massa**

```bash
poetry run python scripts/generate_predictions.py
```

**4. Visualizar as Análises**

- **Power BI**: Abrir `reports/dashboard.pbix` e clicar em "Atualizar".
- **Streamlit**:

```bash
poetry run streamlit run app/main_app.py
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

- **Algoritmo**: XGBoost Classifier
- **Técnica de balanceamento**: SMOTEENN
- **F1-Score (Classe "Yes")**: \~0.53

Este F1-Score reflete uma estratégia que prioriza a capacidade de detectar verdadeiros positivos, mesmo com a classe "Yes" sendo minoritária (\~16%).

---

## 📦 Dataset

- Fonte: [IBM HR Analytics Employee Attrition Dataset (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1.470 registros
- 35 features
- Target: `Attrition` (Yes/No)

---

