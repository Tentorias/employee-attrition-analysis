# 🧠 Employee Attrition Analysis — BI & ML Solution

Análise e predição da rotatividade de funcionários. Este projeto evolui de um pipeline puro de Machine Learning para uma **solução completa de Business Intelligence e ML**, com recursos de análise estratégica (Power BI) e tática (Streamlit) para apoiar decisões no setor de Recursos Humanos.

---

## 🏛️ Arquitetura da Solução

A solução é composta por duas camadas complementares, alimentadas por uma **fonte de dados central**.

### 🔗 Fonte Única da Verdade
- **SQLite Database**: `hr_analytics.db`  
  Centraliza os dados brutos, processados e os resultados das predições.

### 📈 Camada Estratégica — Visão para a Liderança
- **Propósito**: Diagnosticar a saúde da organização e monitorar KPIs.
- **Ferramentas**: SQL + Power BI
- **Público-Alvo**: Diretoria, C-Level, Head de RH
- **Exemplo de Pergunta**:  
  _"Qual departamento tem maior turnover e qual o impacto financeiro disso?"_

### 🚀 Camada Tática & Preditiva — Apoio à Decisão
- **Propósito**: Analisar casos individuais e simular cenários.
- **Ferramentas**: Python (ML) + Streamlit
- **Público-Alvo**: Gestores, Analistas de RH
- **Exemplo de Pergunta**:  
  _"Qual a probabilidade do funcionário João sair? Como reduzir esse risco?"_

---

## 🎯 Objetivos

- Identificar colaboradores com **alto risco de desligamento** via modelo preditivo.
- Compreender **fatores que influenciam a rotatividade** com XAI (SHAP).
- Fornecer um **app interativo (Streamlit)** para simulações "e se".
- Disponibilizar **dashboards executivos (Power BI)** para liderança.

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
- Imbalanced-learn (SMOTEENN)  
- Optuna  

### 🖼️ Visualização & Aplicação
- Matplotlib, Seaborn  
- SHAP  
- Streamlit  
- Jupyter Notebook  

### ⚙️ Desenvolvimento & MLOps
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

### Pré-requisitos
- Python 3.10+
- Poetry instalado
- Git e Git LFS

### Instalação

```bash
git clone https://github.com/Tentorias/employee-attrition-analysis.git
cd employee-attrition-analysis
poetry install
```

### ⚙️ Fluxo de Execução

**1. Criar a Base de Dados:**

```bash
poetry run python scripts/load_raw_to_db.py
```

**2. Executar o Pipeline de ML:** (treinar/retreinar):

```bash
poetry run python src/attrition/main.py [comando]
```

**3. Gerar Predições em Massa:**

```bash
poetry run python scripts/generate_predictions.py
```

**4. Visualizar as Análises:**

- **Power BI**: Abrir `reports/dashboard.pbix` e clicar em "Atualizar".
- **Streamlit**:

```bash
poetry run streamlit run app/main_app.py
```

**5. Validar performance do modelo (opcional):**

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

---

## 📦 Dataset

- Fonte: [IBM HR Analytics Employee Attrition Dataset (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1.470 registros
- 35 features
- Target: `Attrition` (Yes/No)

---

