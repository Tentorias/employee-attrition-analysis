
# 🧠 Análise de Attrition de Funcionários — Solução de BI & Machine Learning

Projeto de análise e predição de rotatividade de funcionários.  
A solução evolui de um pipeline de Machine Learning para uma arquitetura completa de Business Intelligence + ML, com camadas estratégicas (Power BI), táticas (Streamlit) e operacionais (API REST) para apoiar decisões no setor de RH.

---

## 🏛️ Arquitetura da Solução

**Banco de Dados Central:**  
PostgreSQL → Armazena dados dos funcionários, logs de predições da API e serve como fonte para as camadas tática e estratégica.

### Camadas:
- **Estratégica (Power BI):** Dashboards e KPIs de turnover para alta gestão, conectados diretamente ao PostgreSQL.
- **Tática (Streamlit):** Diagnóstico individual com predições em tempo real e ranking de risco para gestores e RH.
- **Operacional (API REST):** Serviço automatizado de predição para outros sistemas, com deploy público e logging integrado ao banco de dados.

---

## 🚀 Principais Funcionalidades & Demonstração

Este projeto foi construído com foco em automação, reprodutibilidade e aplicação prática.  
Abaixo estão os comandos chave que demonstram as principais funcionalidades da solução, ideais para uma apresentação.

### **1. Pipeline de ML Orquestrado**
Todo o fluxo de Machine Learning, desde a limpeza dos dados até o treino e avaliação do modelo final, é executado com um único comando, garantindo consistência e facilidade de re-treinamento.

```bash
# Executa todo o pipeline de treino, gerando o modelo de produção
poetry run python src/attrition/main.py run-pipeline
```

### **2. Ferramenta de Diagnóstico Interativa (Streamlit)**
Uma aplicação web para o time de RH que consome os dados do PostgreSQL e utiliza o modelo treinado para gerar diagnósticos individuais com predições e explicações em tempo real.

```bash
# Inicia a aplicação tática
poetry run streamlit run app/main_app.py
```

### **3. API de Predição em Produção (FastAPI)**
O modelo de machine learning foi colocado em produção através de uma API REST, permitindo que qualquer outro sistema consuma as predições de forma automatizada.

**Documentação da API:**  
https://employee-attrition-analysis.onrender.com/docs

```bash
# Exemplo de como rodar a API localmente
poetry run uvicorn api.main:app --reload
```

---

## ⚙️ Stack Tecnológica

### **Dados & BI**
- PostgreSQL, SQLAlchemy, SQL, Power BI

### **Modelagem & Core**
- Python 3.10+, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, SMOTEENN, Optuna

### **Visualização & Apps**
- Streamlit, SHAP, Matplotlib, Seaborn, Jupyter Notebook, FastAPI

### **Dev & MLOps**
- Poetry, python-dotenv, Git, Git LFS, Docker, Render, GitHub Actions, Pre-commit, Black, isort, Flake8, Pytest

---

## 🚀 Guia de Uso (Local)

### **Pré-requisitos:**
- Python 3.10+
- Poetry e Git
- Cliente PostgreSQL (psql) instalado e no PATH do sistema.

### **1. Instalação e Configuração:**

```bash
# Clonar o repositório
git clone https://github.com/Tentorias/employee-attrition-analysis.git
cd employee-attrition-analysis

# Instalar dependências
poetry install

# Criar e configurar o ficheiro de ambiente
cp .env.example .env
```

Após o último comando, edite o arquivo `.env` e preencha a `DATABASE_URL` com a URL do seu banco PostgreSQL.

### **2. Execução Essencial:**

#### a. Povoar a Base de Dados:

```bash
# Carrega os dados do CSV para o PostgreSQL (execute apenas uma vez)
poetry run python scripts/seed_database.py
```

#### b. Treinar o Modelo e Visualizar a Análise:

```bash
# Treina o modelo
poetry run python src/attrition/main.py run-pipeline

# Inicia o dashboard interativo
poetry run streamlit run app/main_app.py
```

Para uma lista completa de todos os comandos individuais do pipeline, consulte o Guia do Desenvolvedor.

---

## 🔗 Pipeline de Machine Learning

- **Pré-processamento:** Limpeza, transformações logarítmicas e engenharia de features.
- **Balanceamento e Otimização:** SMOTEENN para balanceamento de classes e Optuna para otimização de hiperparâmetros.
- **Modelagem e Explicabilidade:** XGBoost como modelo principal e SHAP para explicar as predições.
- **Avaliação:** Foco em F1-Score e Recall, além da AUC, para maximizar a detecção de talentos em risco.

---

## 📊 Resultados do Modelo (Produção)

| Modelo             | Precision (Yes) | Recall (Yes) | F1-Score (Yes) | AUC  |
|--------------------|-----------------|--------------|----------------|------|
| **XGBoost (Prod)** | 0.54            | 0.66         | 0.60           | 0.87 |

- **Recall de 66%:** O modelo consegue identificar corretamente 2 em cada 3 funcionários que de fato sairiam, um indicador chave dado o alto custo de perder talentos.

---

## 🧑‍⚖️ Considerações Éticas

O modelo não toma decisões, mas gera insights para apoiar a ação humana.  
Para garantir a utilidade e ética, o dashboard tático foi refinado para exibir apenas fatores de risco que são diretamente influenciáveis pela gestão de RH, omitindo características puramente pessoais e não acionáveis.