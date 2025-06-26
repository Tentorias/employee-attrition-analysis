import sqlite3
import pandas as pd
from pathlib import Path

# --- Configurações ---
# Define os caminhos de forma segura, garantindo que o script funcione de qualquer lugar.
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
DB_PATH = BASE_DIR / "database" / "hr_analytics.db"
TABLE_NAME = "employees"


def load_csv_to_sqlite():
    """
    Lê os dados do CSV de origem e os carrega em uma tabela SQLite,
    substituindo a tabela se ela já existir para garantir dados frescos.
    """
    print("🚀 Iniciando a carga de dados para o banco de dados...")

    if not RAW_DATA_PATH.exists():
        print(f"❌ ERRO: Arquivo de dados não encontrado em: {RAW_DATA_PATH}")
        return

    try:
        # 1. Ler o arquivo CSV
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"✔ Arquivo CSV lido com sucesso ({len(df)} linhas).")

        # 2. Conectar ao banco de dados SQLite
        with sqlite3.connect(DB_PATH) as conn:
            # 3. Salvar o DataFrame na tabela, substituindo se existir
            df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
            print(f"✔ Dados salvos com sucesso na tabela '{TABLE_NAME}' no banco de dados '{DB_PATH.name}'.")

        print("\n🏛️ - Marco Concluído: A Fundação de Dados foi criada com sucesso!")

    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")


if __name__ == "__main__":
    load_csv_to_sqlite()