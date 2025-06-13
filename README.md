# 🧠 Employee Attrition Analysis & Prediction

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![CI](https://github.com/seu-usuario/employee_attrition_project/actions/workflows/ci.yml/badge.svg)](https://github.com/seu-usuario/employee_attrition_project/actions)

Análise e predição da rotatividade de funcionários (employee attrition) com machine learning. Pipeline completo de ponta a ponta, desde análise exploratória até modelo treinado e interpretável, ajudando o RH a agir de forma proativa na retenção de talentos.

## 🎯 Objetivos

* Identificar funcionários com alto risco de desligamento
* Compreender os principais fatores que influenciam a rotatividade
* Fornecer insights e alertas para ações preventivas nos programas de retenção de talentos

## 🧰 Stack Tecnológica

**Core:**

* Python 3.10+
* pandas, numpy
* scikit‑learn
* XGBoost
* imbalanced‑learn (SMOTE)

**Visualização & Explicabilidade:**

* matplotlib, seaborn
* SHAP (interpretabilidade)
* Jupyter Notebook

**Desenvolvimento & CI/CD:**

* pytest (testes automatizados)
* Poetry (gerenciamento de dependências)
* GitHub Actions (integração contínua)

## 📁 Estrutura do Projeto

```
employee_attrition_project/
│
├── .github/                  # CI/CD
│   └── workflows/
│       └── ci.yml
│
├── .pytest_cache/           # Cache de testes
│
├── artifacts/               # Artefatos do pipeline
│   ├── features/
│   ├── models/
│   └── results/
│
├── attrition.egg-info/      # Metadata para build
│
├── data/                    # Conjuntos de dados
│   ├── raw/
│   └── processed/
│
├── htmlcov/                 # Cobertura de testes
│
├── models/                  # Modelos e metadados finais
│
├── notebooks/               # Análise exploratória e modelagem
│   ├── eda_attrition_funcionarios.ipynb
│   └── modeling_ml.ipynb
│
├── outputs/                 # (reservado a saídas e plots)
│
├── reports/                 # Relatórios
│
├── src/                     # Código-fonte principal
│   └── attrition/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── utils/
│       └── main.py
│
├── tests/                   # Testes automatizados
│   ├── data/
│   ├── features/
│   ├── models/
│   └── test_main_cli.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── makefile
├── README.md
├── requirements.txt
└── setup.cfg
```

## 🚀 Quick Start

### Pré-requisitos

* Python 3.10+
* Git
* pip

### Instalação

```bash
# Clonar repositório
git clone https://github.com/Tentorias/employee-attrition-analysis.git
cd employee_attrition_project

# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# Instalar o pacote em modo desenvolvimento
pip install -e .


#### Via CLI

```bash
# 1. Processar dados brutos
    python src/attrition/main.py process --raw-path data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv --out-path data/processed/employee_attrition_processed.csv

# 2. Engenharia de features
    python src/attrition/main.py engineer --input-path data/processed/employee_attrition_processed.csv --output-path data/features_matrix.csv

# 3. Treinar modelo
    python src/attrition/main.py train --data-path data/features_matrix.csv --model-path models/model.pkl --thr-path models/threshold_optimizado.pkl


# 5. Explicabilidade (SHAP)
    python src/attrition/main.py explain --model-path models/model.pkl --data-path data/features_matrix.csv
```

#### Via Notebooks

```bash
jupyter notebook # ou jupyter lab
# Abra e execute em ordem:
# 1. notebooks/eda_attrition.ipynb
# 2. notebooks/modeling_ml.ipynb
```

### Testes

# Executar todos os testes
pytest

# Com cobertura de código
pytest --cov=src --cov-report=term-missing

# Executar testes específicos
pytest tests/data/test_process.py

## 📊 Pipeline de ML

1. **EDA**: análise univariada e bivariada, identificação de padrões e tratamento de outliers
2. **Pré-processamento**: limpeza, encoding, transformações log
3. **Engenharia de Features**: novas variáveis, balanceamento SMOTE
4. **Modelagem**: XGBoost, Optuna para hiperparâmetros, threshold tuning (F1)
5. **Avaliação & Explicabilidade**: precision/recall/F1, matriz de confusão, SHAP

## 🏢 Exemplo de Aplicação

> Uma equipe de RH carrega os dados dos funcionários e executa o pipeline:
>
> 1. Processa e limpa os dados
> 2. Gera matriz de features e treina o modelo XGBoost
> 3. Otimiza threshold para maximizar F1 na classe "Yes"
> 4. Analisa importâncias via SHAP para entender fatores-chave
> 5. Salva o modelo e threshold para uso em alertas internos

## 📈 Resultados (baseline)

* **Algoritmo:** XGBoost Classifier
* **F1-Score (classe "Yes"):** 0.6118
* **Threshold otimizado:** 0.52
* **Técnica de balanceamento:** SMOTE

### Top Features

1. OverTime
2. Age
3. JobRole
4. EnvironmentSatisfaction
5. MonthlyIncome
6. WorkLifeBalance

## 📦 Dataset

**Fonte:** [IBM HR Analytics Employee Attrition Dataset (Kaggle)](https://kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

* 1.470 registros
* 35 features (demográficas, satisfação, carreira)
* Target: Attrition (Yes/No) \~16% positivo

## 🤝 Contribuindo

1. Fork
2. Branch (`feature/...`)
3. Commit & PR
4. CI executa testes e coverage

## 📄 Licença

MIT © Mateus Cabral
