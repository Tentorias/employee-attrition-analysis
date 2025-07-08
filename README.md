🧠 Análise de Attrition de Funcionários — Solução de BI & Machine Learning
Projeto de análise e predição de rotatividade de funcionários. A solução evolui de um pipeline de Machine Learning para uma arquitetura completa de Business Intelligence + ML, com camadas estratégicas (Power BI), táticas (Streamlit) e operacionais (API REST) para apoiar decisões no setor de RH.

🚀 API de Predição (Deploy Público)
A API de predição está disponível para testes:

Documentação: https://employee-attrition-analysis.onrender.com/docs

⚠️ Atenção: A API usa o plano gratuito da Render e pode levar até 60 segundos para responder à primeira requisição após inatividade.

Exemplo de requisição:

curl -X 'POST'   'https://employee-attrition-analysis.onrender.com/predict'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{ ... }'

🏛️ Arquitetura da Solução
Banco de Dados Central: PostgreSQL → Armazena dados dos funcionários, logs de predições da API e serve como fonte para as camadas tática e estratégica.

Camadas:
Estratégica (Power BI): Dashboards e KPIs de turnover para alta gestão, conectados diretamente ao PostgreSQL.

Tática (Streamlit): Diagnóstico individual com predições em tempo real e ranking de risco para gestores e RH.

Operacional (API REST): Serviço automatizado de predição para outros sistemas, com logging integrado ao banco de dados.

⚙️ Stack Tecnológica
Dados & BI
PostgreSQL, SQLAlchemy, SQL, Power BI

Modelagem & Core
Python 3.10+
Pandas, NumPy
Scikit-learn, LightGBM, XGBoost
SMOTEENN, Optuna

Visualização & Apps
Streamlit, SHAP, Matplotlib, Seaborn, Jupyter Notebook, FastAPI

Dev & MLOps
Poetry, Git, Git LFS
python-dotenv
Docker, Render
GitHub Actions, Pre-commit, Black, isort, Flake8, Pytest

📁 Estrutura do Projeto
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

🚀 Guia de Uso (Local)
Pré-requisitos:
Python 3.10+
Poetry
Git + Git LFS
Cliente PostgreSQL (psql) instalado e no PATH do sistema.

1. Instalação:
git clone https://github.com/Tentorias/employee-attrition-analysis.git
cd employee-attrition-analysis
poetry install

2. Configuração do Ambiente:
Crie o ficheiro de ambiente a partir do molde:

cp .env.example .env

Configure a Base de Dados:
Abra o ficheiro .env que acabou de ser criado e preencha a variável DATABASE_URL com a URL de conexão do seu banco de dados PostgreSQL. Para testes locais, use a URL de conexão externa.

3. Execução:
a. Migrar dados para o PostgreSQL:
Este comando lê os dados brutos do CSV e os carrega na sua base de dados PostgreSQL. Execute apenas uma vez.

poetry run python scripts/migrate_to_postgres.py

b. Treinar o modelo:
Execute o pipeline completo para processar os dados e treinar o modelo de produção.

poetry run python src/attrition/main.py run-pipeline

c. Visualizar análises:

Power BI: Abra reports/dashboard.pbix e conecte-o à sua base de dados PostgreSQL.

Streamlit (com predição em tempo real):

poetry run streamlit run app/main_app.py

d. Usar a API localmente (Opcional):

Inicie a API:

poetry run uvicorn api.main:app --reload

Acesse a documentação em http://127.0.0.1:8000/docs.

🔗 Pipeline de Machine Learning
Pré-processamento: Limpeza, transformações logarítmicas.
Engenharia de Features: Variáveis derivadas, One-Hot Encoding.
Balanceamento: SMOTEENN.
Otimização: Optuna.
Modelagem: XGBoost.
Explicabilidade: SHAP.
Avaliação: F1, Recall, Precision, AUC.

📊 Resultados do Modelo (Produção)

| Modelo              | Precision (Yes) | Recall (Yes) | F1-Score (Yes) | AUC  |
|---------------------|-----------------|--------------|----------------|------|
| Logistic Regression | 0.70            | 0.34         | 0.46           | -    |
| **XGBoost (Prod)**  | 0.54            | 0.66         | 0.60           | 0.87 |
| LightGBM            | 0.65            | 0.28         | 0.39           | 0.83 |

Recall de 66%: Detecta 2 em cada 3 saídas.
AUC 0.87: Excelente separação entre classes.
Foco em maximizar recall, dado o alto custo de perder talentos.

📦 Dataset
Fonte: IBM HR Analytics (Kaggle)
1.470 registros
35 features
Target: Attrition (Yes/No)

Considerações Éticas
Este modelo não toma decisões nem classifica funcionários. Ele apenas identifica padrões relacionados à intenção de saída e apresenta insights. Cabe exclusivamente ao RH interpretar e agir.

Para garantir que os insights sejam úteis e éticos, o dashboard tático foi refinado para exibir apenas fatores de risco que são diretamente influenciáveis pela gestão de RH, omitindo características puramente pessoais.

O risco não está no modelo, mas na inação diante dos sinais.