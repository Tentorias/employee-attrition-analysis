# 🧠 Análise e Predição de Attrition de Funcionários
Análise e predição da rotatividade de funcionários (employee attrition) com machine learning. Este projeto implementa um pipeline completo de ponta a ponta, desde a análise exploratória até um modelo de produção otimizado e interpretável, culminando em uma aplicação interativa com Streamlit para apoiar o RH na retenção de talentos.

# 🎯 Objetivos
- Identificar funcionários com alto risco de desligamento através de um modelo preditivo.

- Compreender os principais fatores que influenciam a rotatividade com técnicas de explicabilidade (XAI).

- Fornecer uma ferramenta interativa (app Streamlit) para simulações "what-if" e análises de casos individuais.

# 🧰 Stack Tecnológica

## Core & Modelagem:

- Python 3.10+

- Pandas, NumPy

- Scikit-learn

- XGBoost

- Imbalanced-learn (SMOTEENN)

- Optuna (Otimização de Hiperparâmetros)

## Visualização & Aplicação:

- Matplotlib, Seaborn

- SHAP (Explicabilidade do Modelo)

- Streamlit (Dashboard Interativo)

- Jupyter Notebook (Análise Exploratória)

##  Desenvolvimento & MLOps:

- Poetry (Gerenciamento de Dependências e Ambientes)

- Git & Git LFS (Versionamento de código e modelos)

- Pytest (Testes automatizados)

- Pre-commit, Black, isort, Flake8 (Qualidade e formatação de código)

- GitHub Actions (Integração Contínua - CI)

# 📁 Estrutura do Projeto:
```
employee-attrition-analysis/
│
├── app/
│   ├── main_app.py         # Script principal do app Streamlit
│   └── ui_config.py        # Dicionários de configuração da interface
│
├── artifacts/              # Saídas do pipeline (modelos, features, etc.)
│   ├── features/
│   └── models/
│
├── data/
│   ├── raw/                # Dados brutos originais
│   └── processed/          # Dados limpos após a primeira etapa
│
├── models/                 # Modelo final, pronto para produção
│   └── production_model.pkl
│
├── notebooks/              # Análise exploratória e prototipagem
│
├── src/
│   └── attrition/
│       ├── data/           # Scripts de processamento de dados
│       ├── features/       # Scripts de engenharia de features
│       └── models/         # Scripts de treino, avaliação, predição, etc.
│       └── main.py         # Orquestrador da linha de comando (CLI)
│
├── tests/                  # Testes automatizados
│
├── .gitignore
├── .gitattributes          # Configuração do Git LFS
├── pre-commit-config.yaml  # Configuração dos hooks de pre-commit
├── pyproject.toml          # Arquivo de configuração do projeto 
└── poetry.lock             # Garante instalações determinísticas
```

# 🚀 Guia de Uso

## Pré-requisitos
- Python 3.10+
- Poetry instalado (consulte a documentação oficial para instalar)
- Git e Git LFS instalados (git lfs install)

## Instalação
### 1. Clone o repositório
git clone https://github.com/Tentorias/employee-attrition-analysis.git
cd employee-attrition-analysis

### 2. Instale as dependências com Poetry
```
poetry install
```

### Como Usar o Pipeline via CLI
O projeto é orquestrado pelo src/attrition/main.py, que aceita vários comandos.

1. Executar o Pipeline Completo (Recomendado)
Este comando executa as etapas de processamento, engenharia, treino e avaliação em sequência.
```
poetry run python src/attrition/main.py run-pipeline
```
2. Executar Passos Individualmente

### Etapa de limpeza dos dados
```
poetry run python src/attrition/main.py process --raw-path data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv --out-path data/processed/employee_attrition_processed.csv
```

### Etapa de engenharia de features
```
poetry run python src/attrition/main.py engineer --input-path data/processed/employee_attrition_processed.csv --output-path artifacts/features/features_matrix.csv --features-out-path artifacts/features/features.pkl
```

### Etapa de Retreino Final (gera o modelo de produção em /models)
```
poetry run python src/attrition/main.py train --retrain-full-data --data-path artifacts/features/features_matrix.csv --features-path artifacts/features/features.pkl --model-path models/production_model.pkl
```

### Como Rodar a Aplicação Web (Streamlit)
Após gerar o modelo de produção com o comando de retreino final, execute:
```
poetry run streamlit run app/main_app.py
```
Um painel interativo será aberto no seu navegador.

# 📊 Pipeline de ML
1. Processamento: Limpeza de dados, transformações logarítmicas.

2. Engenharia de Features: Criação de variáveis derivadas (YearsPerCompany) e codificação One-Hot.

3. Balanceamento de Dados: Utilização da técnica híbrida SMOTEENN para criar dados sintéticos da classe minoritária e limpar ruídos, combatendo o desbalanceamento e o overfitting.

4. Otimização: Busca de hiperparâmetros com Optuna para encontrar a configuração mais robusta do XGBoost.

5. Modelagem: Treinamento do modelo final XGBoost com os parâmetros otimizados.

6. Avaliação & Explicabilidade: Análise de precision, recall e F1-score, além da preparação para uso de SHAP para interpretabilidade.

# 📈 Resultados do Modelo Final
Este projeto culminou em um modelo XGBoost otimizado para robustez e generalização.

- Algoritmo: XGBoost Classifier

- Técnica de balanceamento: SMOTEENN

- F1-Score (Classe "Yes" no teste): ~0.53

Este F1-Score é o resultado de uma estratégia focada em reduzir o overfitting, trocando um pico de performance potencialmente instável (~0.61) por um modelo mais confiável e generalista, ideal para uma aplicação de negócio. O modelo apresenta um excelente recall (~0.79), sendo muito eficaz em identificar a maioria dos funcionários com risco de saída.

# 📦 Dataset
Fonte: IBM HR Analytics Employee Attrition Dataset (Kaggle)

- 1.470 registros

- 35 features (demográficas, de satisfação e de carreira)

- Target: Attrition (Yes/No) ~16% positivo
