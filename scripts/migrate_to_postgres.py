import os
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path
import sys

# Configuração para encontrar a raiz do projeto
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- CAMINHOS E CONFIGURAÇÕES ---
SQLITE_DB_PATH = project_root / "database" / "hr_analytics.db"
POSTGRES_URL = os.getenv("DATABASE_URL")

def migrate_data():
    """
    Lê a tabela 'employees' do SQLite e a escreve no PostgreSQL.
    """
    if not POSTGRES_URL:
        print("❌ Erro: A variável de ambiente DATABASE_URL não foi encontrada.")
        print("Verifique se o seu arquivo .env está configurado corretamente.")
        return

    print("Iniciando a migração de dados...")

    try:
        # 1. Conectar ao SQLite e ler os dados
        print(f"📖 Lendo dados do SQLite em: {SQLITE_DB_PATH}")
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM employees", conn)
        print(f"✅ {len(df)} registros lidos do SQLite.")

        # 2. Conectar ao PostgreSQL usando SQLAlchemy
        print("🔗 Conectando ao PostgreSQL...")
        engine = create_engine(POSTGRES_URL)

        # 3. Escrever o DataFrame para uma nova tabela 'employees' no PostgreSQL
        print("⏳ Gravando dados na tabela 'employees' do PostgreSQL... (Isso pode levar um momento)")
        # 'if_exists='replace'' apaga a tabela se ela já existir, útil para re-executar o script
        df.to_sql('employees', engine, if_exists='replace', index=False)

        print("✅ Migração concluída com sucesso!")

    except FileNotFoundError:
        print(f"❌ Erro: Arquivo do banco de dados SQLite não encontrado em {SQLITE_DB_PATH}.")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado durante a migração: {e}")

if __name__ == "__main__":
    migrate_data()