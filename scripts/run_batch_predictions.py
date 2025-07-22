# scripts/run_batch_predictions.py

import os
import time
import pandas as pd
import requests
from sqlalchemy import create_engine
from dotenv import load_dotenv
from tqdm import tqdm  # Biblioteca para barras de progresso

def run_batch_predictions():
    """
    Lê todos os funcionários do banco de dados e solicita uma predição para cada um
    através da API local, populando a tabela 'predictions'.
    """
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    api_url = "http://127.0.0.1:8000/predict" # URL da sua API local

    if not db_url:
        print("❌ DATABASE_URL não encontrada. Verifique seu arquivo .env.")
        return

    print("🔌 Conectando ao banco de dados para buscar a lista de funcionários...")
    engine = create_engine(db_url)
    try:
        df = pd.read_sql("SELECT * FROM employees", engine)
        print(f"✅ Encontrados {len(df)} funcionários para processar.")
    except Exception as e:
        print(f"❌ Erro ao ler a tabela 'employees': {e}")
        return

    # Limpa a tabela de predições antigas para começar do zero
    try:
        with engine.connect() as conn:
            conn.execute(requests.text("TRUNCATE TABLE predictions RESTART IDENTITY;"))
            print("🧹 Tabela 'predictions' limpa com sucesso.")
    except Exception:
        print("ℹ️ Tabela 'predictions' não encontrada ou não pôde ser limpa. Será criada pela API.")

    print(f"\n🚀 Iniciando predições em lote via API em {api_url}...")

    # Itera sobre cada funcionário com uma barra de progresso (tqdm)
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processando funcionários"):
        # Converte a linha do DataFrame para o formato JSON que a API espera
        payload = row.to_dict()

        try:
            # Envia a requisição POST para a API
            response = requests.post(api_url, json=payload)
            if response.status_code != 200:
                # Imprime um erro se a API não responder com sucesso
                print(f"\n⚠️ Erro na API para o funcionário {payload.get('EmployeeNumber')}: {response.text}")
        except requests.exceptions.ConnectionError as e:
            print(f"\n❌ Erro de conexão com a API. A API está rodando em {api_url}? Detalhes: {e}")
            break # Interrompe o script se a API não estiver acessível
        
        time.sleep(0.05) # Pausa pequena para não sobrecarregar a API

    print("\n✅ Predições em lote concluídas! Verifique seu dashboard.")


if __name__ == "__main__":
    # Instale a dependência 'tqdm' se não tiver: poetry add tqdm
    run_batch_predictions()