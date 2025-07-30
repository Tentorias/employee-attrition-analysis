# scripts/clean_predictions.py
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


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

        query = text(
            """
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
        """
        )

        with engine.connect() as conn:
            result = conn.execute(query)
            conn.commit()
            print(
                f"✅ Limpeza concluída. {result.rowcount} registros duplicados foram removidos."
            )

    except Exception as e:
        print(f"❌ Erro durante a limpeza: {e}")


if __name__ == "__main__":
    clean_duplicate_predictions()
