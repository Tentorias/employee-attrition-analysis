.PHONY: setup db-seed run-pipeline batch-predict start-api start-streamlit clean test all deploy-ci-cd

POETRY_RUN = poetry run

# --- VARIÁVEIS COM CAMINHOS DE ARTEFATOS ---
RAW_DATA_PATH = data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
PROCESSED_DATA_PATH = data/processed/employee_attrition_processed.csv
MODEL_EVAL_PATH = artifacts/models/model.pkl
FEATURES_PATH = artifacts/features/features.pkl
BEST_PARAMS_PATH = artifacts/models/best_params.json
X_TEST_PATH = artifacts/features/X_test.csv
Y_TEST_PATH = artifacts/features/y_test.csv
PROD_MODEL_PATH = models/production_model.pkl
OPTIMAL_THRESHOLD_PATH = artifacts/models/optimal_threshold.pkl
SHAP_EXPLAINER_PATH = models/production_shap_explainer.pkl



setup:
	@echo "Instalando dependências Poetry..."
	$(POETRY_RUN) poetry install
	@echo "Criando .env a partir de .env.example (se não existir)..."
	cp .env.example .env || true
	@echo "✅ Setup inicial concluído. Preencha seu .env com DATABASE_URL."


db-seed:
	@echo "Semeando o banco de dados PostgreSQL com dados brutos..."
	$(POETRY_RUN) python scripts/seed_database.py
	@echo "✅ Banco de dados populado."

create-explainer:
	@echo "Criando e salvando o SHAP Explainer..."
	$(POETRY_RUN) python scripts/create_explainer.py \
		--model-path $(PROD_MODEL_PATH) \
		--output-path $(SHAP_EXPLAINER_PATH)
	@echo "✅ SHAP Explainer criado."


batch-predict:
	@echo "Gerando predições em lote e salvando no banco de dados..."
	$(POETRY_RUN) python scripts/run_batch_predictions.py
	@echo "✅ Predições em lote concluídas e salvas no DB."


start-api:
	@echo "Iniciando a API FastAPI em http://127.0.0.1:8000..."
	$(POETRY_RUN) uvicorn api.main:app --reload
	@echo "✅ API iniciada."


start-streamlit:
	@echo "Iniciando o dashboard Streamlit..."
	$(POETRY_RUN) streamlit run app/main_app.py
	@echo "✅ Dashboard Streamlit iniciado."

# --- MANUTENÇÃO E TESTE ---

clean:
	@echo "Limpando caches e artefatos..."
	rm -rf $(MODEL_EVAL_PATH) $(PROD_MODEL_PATH) $(FEATURES_PATH) $(BEST_PARAMS_PATH) $(X_TEST_PATH) $(Y_TEST_PATH) $(OPTIMAL_THRESHOLD_PATH) $(SHAP_EXPLAINER_PATH)
	rm -rf __pycache__/ .pytest_cache/ htmlcov/ attrition.egg-info/ .mypy_cache/
	$(POETRY_RUN) streamlit cache clear
	@echo "✅ Limpeza concluída."


test:
	@echo "Executando testes automatizados..."
	$(POETRY_RUN) pytest tests/
	@echo "✅ Testes concluídos."


all: setup db-seed run-pipeline batch-predict test start-streamlit


deploy-ci-cd:
	@echo "Iniciando processo de deploy contínuo (CD)..."
	@echo "Este passo será implementado para automatizar o deploy em plataformas como Render."
	# Exemplo: Render CLI deploy command
	# $(POETRY_RUN) render deploy --service-id your-streamlit-service-id
	# $(POETRY_RUN) render deploy --service-id your-api-service-id
	@echo "✅ CD placeholder concluído."

run-pipeline: db-seed
	@echo "Iniciando o pipeline completo de ML..."
	$(POETRY_RUN) python src/attrition/main.py run-pipeline
	@echo "✅ Pipeline de ML concluído."

	# NOVO: Gerar o SHAP Explainer após o pipeline de ML
	$(MAKE) create-explainer # Chama a receita create-explainer
	@echo "✅ SHAP Explainer gerado e salvo."
