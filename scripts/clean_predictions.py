# scripts/clean_predictions.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def clean_duplicate_predictions():
    """
    Remove predições duplicadas da tabela 'predictions', mantendo apenas a
    mais recente para cada 'EmployeeNumber'.
    """
    print("--- INICIANDO LIMPEZA DE PREDIÇÕES DUPLICADAS ---")
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ ERRO: DATABASE_URL não encontrada.")
        return

    try:
        engine = create_engine(db_url)
        
        # Query SQL para deletar duplicatas, mantendo a mais recente
        # A sintaxe pode variar um pouco dependendo do SQL (PostgreSQL aqui)
        query = text("""
        DELETE FROM predictions
        WHERE ctid IN (
            SELECT ctid
            FROM (
                SELECT
                    ctid,
                    ROW_NUMBER() OVER(
                        PARTITION BY "EmployeeNumber"
                        ORDER BY prediction_timestamp DESC
                    ) as rn
                FROM predictions
            ) t
            WHERE t.rn > 1
        );
        """)

        with engine.connect() as conn:
            result = conn.execute(query)
            conn.commit() # Efetiva a transação
            print(f"✅ Limpeza concluída. {result.rowcount} registros duplicados foram removidos.")

    except Exception as e:
        print(f"❌ Erro durante a limpeza: {e}")

if __name__ == "__main__":
    clean_duplicate_predictions()