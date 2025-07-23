🧠 **Análise de Attrition de Funcionários — Solução de BI & Machine Learning**
Projeto de análise e predição de rotatividade de funcionários. A solução evolui de um pipeline de Machine Learning para uma arquitetura completa de Business Intelligence + ML, com camadas estratégicas (Power BI), táticas (Streamlit) e operacionais (API REST) para apoiar decisões no setor de RH.

---

🏛️ **Arquitetura da Solução**

**Banco de Dados Central:**
- PostgreSQL → Armazena dados dos funcionários, logs de predições da API e serve como fonte para as camadas tática e estratégica.

**Camadas:**
- **Estratégica (Power BI):** Dashboards e KPIs de turnover para alta gestão.
- **Tática (Streamlit):** Dashboard de diagnóstico individual e ranking de risco para gestores e RH.
- **Operacional (API REST):** Serviço automatizado de predição para outros sistemas, com deploy na nuvem.

---

''''

------------------------------------
📂 ESTRUTURA DO PROJETO
------------------------------------
.
├── 📁 .github/
├── 📁 .pytest_cache/
├── 📁 .venv/
├── 📁 api/
├── 📁 app/
├── 📁 artifacts/
├── 📁 attrition.egg-info/
├── 📁 data/
├── 📁 database/
├── 📁 htmlcov/
├── 📁 models/
├── 📁 notebooks/
├── 📁 reports/
├── 📁 scripts/
├── 📁 sql/
├── 📁 src/
├── 📁 tests/
├── 📄 .coverage
├── 📄 .dockerignore
├── 📄 .env
├── 📄 .env.example
├── 📄 .gitignore
├── 📄 .pre-commit-config.yaml
├── 📄 Dockerfile
├── 📄 LICENSE
├── 📄 Makefile
├── 📄 poetry.lock
├── 📄 pyproject.toml
├── 📄 README.md
└── 📄 setup.cfg

'''''

⚙️ **Stack Tecnológica**

**Dados & BI**
- PostgreSQL, SQLAlchemy, Power BI

**Modelagem & Core**
- Python 3.10+, Pandas, NumPy
- Scikit-learn, XGBoost, Optuna, Imbalanced-learn

**Visualização & Apps**
- Streamlit, SHAP, Matplotlib, Seaborn, FastAPI

**Dev & MLOps**
- Poetry, python-dotenv, Git, Docker, Render
- Pytest, Pre-commit, Black, isort, Flake8

---

🚀 **Como Rodar o Projeto (Localmente)**

1. **Pré-requisitos:**
   - Python 3.10+
   - Poetry e Git

2. **Instalação e Configuração:**

```bash
# Clone o repositório
git clone https://github.com/Tentorias/employee-attrition-analysis.git
cd employee-attrition-analysis

# Instale as dependências
poetry install

# Crie e configure o arquivo de ambiente
cp .env.example .env
```

- Após o último comando, edite o arquivo `.env` e preencha a `DATABASE_URL` com a URL do seu banco PostgreSQL.

3. **Execução do Pipeline e Aplicações:**

**a. Popule a Base de Dados**

```bash
# Carrega os dados do CSV para o PostgreSQL (execute apenas uma vez)
poetry run python scripts/migrate_to_postgres.py
```

**b. Execute o Pipeline de ML Completo**

```bash
# O argumento --tune ativa a otimização de hiperparâmetros para maximizar o recall
poetry run python src/attrition/main.py run-pipeline --tune
```

**c. Popule o Dashboard com as Predições**

```bash
# Usa a API local para gerar e salvar as predições de toda a base no banco de dados
poetry run python scripts/run_batch_predictions.py
```

**d. Inicie as Aplicações**

```bash
# Inicia o dashboard tático
poetry run streamlit run app/main_app.py

# Inicia a API operacional localmente
poetry run uvicorn api.main:app --reload
```

---

🔗 **Pipeline de Machine Learning (À Prova de Data Leakage)**

- **Divisão de Dados Primeiro:** O dataset bruto é imediatamente dividido em conjuntos de treino e teste.
- **Pré-processamento Separado:** Todas as etapas que "aprendem" com os dados (encoding de categorias, etc.) são treinadas apenas no conjunto de treino e depois aplicadas ao conjunto de teste.
- **Otimização com Optuna:** Os hiperparâmetros do XGBoost são otimizados com foco em maximizar o Recall, utilizando o parâmetro `scale_pos_weight` para lidar com o desbalanceamento de classes.
- **Calibração de Threshold:** Após o treino, um threshold de decisão ótimo é calculado para encontrar o melhor equilíbrio entre Recall e Precision, de acordo com a estratégia de negócio.
- **Modelagem e Explicabilidade:** O modelo `XGBClassifier` treinado é salvo, e o SHAP é usado para garantir a explicabilidade das predições.

---

📊 **Resultados do Modelo (Otimizado para Recall)**

O modelo final foi calibrado para atender à necessidade de negócio de minimizar a perda de talentos, priorizando um alto Recall.

| Métrica            | Modelo Otimizado (Prod) |
|--------------------|--------------------------|
| Precision (Yes)    | 0.41 (41%)               |
| Recall (Yes)       | 0.74 (74%)               |
| F1-Score (Yes)     | 0.53 (53%)               |

**Exportar para as Planilhas:**

- **Recall de 74%:** O modelo consegue identificar corretamente 3 em cada 4 funcionários que de fato sairiam. Essa é a métrica mais importante para a estratégia de retenção.
- **Precision de 41%:** De cada 10 funcionários sinalizados como risco, aproximadamente 4 são casos de risco real, permitindo que a ação do RH seja focada e eficiente.
