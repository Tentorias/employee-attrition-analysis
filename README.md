
# 🧠 Análise de Attrition de Funcionários — Solução de BI & Machine Learning

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

**Banco de Dados Central:**  
PostgreSQL → Armazena dados dos funcionários, logs de predições da API e serve como fonte para as camadas tática e estratégica.

### Camadas:
- **Estratégica (Power BI):** Dashboards e KPIs de turnover para alta gestão.
- **Tática (Streamlit):** Diagnóstico individual e ranking de risco para gestores e RH.
- **Operacional (API REST):** Serviço automatizado de predição para outros sistemas.

---

## ⚙️ Stack Tecnológica

### **Dados & BI**
- PostgreSQL, SQLAlchemy, SQL, Power BI

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
├── api/                   # API REST (FastAPI)
├── app/                   # App de diagnóstico (Streamlit)
├── artifacts/             # Modelos, features e explicadores
├── data/                  # Dados brutos
├── models/                # Modelos finais de produção
├── notebooks/             # EDA e explorações
├── reports/               # Dashboards Power BI
├── scripts/               # Scripts de suporte (ex: migração de dados)
├── sql/                   # Queries SQL para a camada de BI
├── src/                   # Pipeline principal de ML
├── tests/                 # Testes automatizados
├── .env.example           # Molde para variáveis de ambiente
├── .gitignore             
├── Dockerfile             
├── pyproject.toml         
└── README.md
```

---

## 🚀 Guia de Uso (Local)

### **Pré-requisitos:**
- Python 3.10+
- Poetry
- Git + Git LFS
- Cliente PostgreSQL (psql) instalado e no PATH do sistema.

### **Instalação:**
```bash
git clone https://github.com/Tentorias/employee-attrition-analysis.git
cd employee-attrition-analysis
poetry install
```

### **Configuração do Ambiente:**
```bash
cp .env.example .env
```
Edite o arquivo `.env` e preencha a variável `DATABASE_URL` com a URL do seu banco PostgreSQL.

### **Execução:**

**1. Migrar dados para o PostgreSQL:**
```bash
poetry run python scripts/migrate_to_postgres.py
```

**2. Treinar o modelo:**
```bash
poetry run python src/attrition/main.py run-pipeline
```

**3. Visualizar análises:**
- Power BI: abra `reports/dashboard.pbix` e conecte-se ao banco.
- Streamlit:
```bash
poetry run streamlit run app/main_app.py
```

**4. Rodar API localmente (opcional):**
```bash
poetry run uvicorn api.main:app --reload
```

Acesse em http://127.0.0.1:8000/docs

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

- **Fonte:** IBM HR Analytics (Kaggle)
- 1.470 registros
- 35 features
- Target: `Attrition` (Yes/No)

---

## 🧑‍⚖️ Considerações Éticas

Este modelo não toma decisões nem classifica funcionários. Ele apenas identifica padrões relacionados à intenção de saída e apresenta insights. Cabe exclusivamente ao RH interpretar e agir.

Para garantir que os insights sejam úteis e éticos, o dashboard tático foi refinado para exibir apenas fatores de risco que são diretamente influenciáveis pela gestão de RH, omitindo características puramente pessoais.

O risco não está no modelo, mas na inação diante dos sinais.
