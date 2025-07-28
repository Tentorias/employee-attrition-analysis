import logging
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Configuração do logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def migrate_data():
    """
    Lê os dados brutos de funcionários de um arquivo CSV e os insere
    em uma tabela 'employees' no banco de dados PostgreSQL.
    """
    # Carrega as variáveis de ambiente do arquivo .env
    load_dotenv()

    # Pega a URL do banco de dados do ambiente
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logging.error("A variável de ambiente DATABASE_URL não foi definida.")
        return

    # Caminho para o arquivo de dados brutos
    raw_data_path = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    table_name = "employees"

    try:
        logging.info("Conectando ao banco de dados...")
        # Cria a 'engine' do SQLAlchemy
        engine = create_engine(database_url)

        logging.info(f"Lendo dados brutos de '{raw_data_path}'...")
        df = pd.read_csv(raw_data_path)

        # F541: Corrigir a f-string para ter a variável `len(df)` dentro das chaves {}
        logging.info(
            f"Iniciando a migração de {len(df)} registros para a tabela '{table_name}'..."
        )

        # Usa o método to_sql do pandas para inserir os dados
        # if_exists='replace' apaga a tabela existente e a recria.
        # Use 'append' se quiser adicionar dados sem apagar.
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)

        logging.info(
            f"✅ Migração concluída com sucesso! A tabela '{table_name}' foi criada/atualizada."
        )

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo de dados '{raw_data_path}' não foi encontrado.")
    except Exception as e:
        logging.error(f"Ocorreu um erro durante a migração: {e}")


if __name__ == "__main__":
    migrate_data()
