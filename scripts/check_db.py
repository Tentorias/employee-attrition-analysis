# check_db.py
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("Erro: DATABASE_URL não definida no arquivo .env.")
else:
    try:
        engine = create_engine(DATABASE_URL)
        print("Conectado ao banco de dados.")

        # Consulta para ver as probabilidades e as predições
        df_predictions = pd.read_sql_table("predictions", con=engine)

        print("\n--- Primeiras 10 Predições ---")
        print(df_predictions.head(10))

        print("\n--- Estatísticas das Probabilidades ---")
        print(df_predictions["predicted_probability"].describe())

        print("\n--- Contagem de Predições por Label ---")
        print(df_predictions["prediction_label"].value_counts())

    except Exception as e:
        print(f"Erro ao consultar o banco de dados: {e}")
