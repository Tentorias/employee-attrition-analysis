<<<<<<< HEAD
# seed_database.py

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

# --- Configurações ---
load_dotenv() # Carrega variáveis do .env
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
TABLE_NAME = "employees"

# Pega a URL do banco de dados do arquivo .env
DATABASE_URL = os.getenv("DATABASE_URL")

def seed_database():
    """
    Lê dados do CSV e os carrega em uma tabela no PostgreSQL,
    substituindo-a se já existir.
    """
    print("🚀 Iniciando a carga de dados para o PostgreSQL...")

    if not DATABASE_URL:
        print("❌ ERRO: A variável de ambiente DATABASE_URL não foi definida no arquivo .env.")
        return
    
    if not RAW_DATA_PATH.exists():
        print(f"❌ ERRO: Arquivo de dados não encontrado em: {RAW_DATA_PATH}")
        return

    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"✔ Arquivo CSV lido com sucesso ({len(df)} linhas).")

        # Converte nomes de colunas para minúsculas para evitar problemas de case no SQL
        df.columns = df.columns.str.lower()
        
        with engine.connect() as conn:
            # Usando 'if_exists="replace"' para apagar e recriar a tabela
            df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
            print(f"✔ Dados salvos com sucesso na tabela '{TABLE_NAME}' no PostgreSQL.")

        print("\n🏛️ - Marco Concluído: Banco de dados PostgreSQL populado com sucesso!")

    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado durante a conexão ou carga para o PostgreSQL: {e}")

if __name__ == "__main__":
    seed_database()
=======
import os
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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
       
        print(f"📖 Lendo dados do SQLite em: {SQLITE_DB_PATH}")
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM employees", conn)
        print(f"✅ {len(df)} registros lidos do SQLite.")
        
        print("🔗 Conectando ao PostgreSQL...")
        engine = create_engine(POSTGRES_URL)
        
        print("⏳ Gravando dados na tabela 'employees' do PostgreSQL... (Isso pode levar um momento)")
        
        df.to_sql('employees', engine, if_exists='replace', index=False)

        print("✅ Migração concluída com sucesso!")

    except FileNotFoundError:
        print(f"❌ Erro: Arquivo do banco de dados SQLite não encontrado em {SQLITE_DB_PATH}.")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado durante a migração: {e}")

if __name__ == "__main__":
    migrate_data()
>>>>>>> aa5bb25655f252f82be0d23e27fbccceac13bf76
