.PHONY: process engineer train evaluate explain test all

process:
	python -m attrition.data.process \
		--raw-path data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
		--out-path data/processed/employee_attrition_processed.csv

engineer:
	python -m attrition.features.engineer \
		--in-path data/processed/employee_attrition_processed.csv \
		--out-path artifacts/features_matrix.csv

train:
	train-model --in-path artifacts/features_matrix.csv

evaluate:
	python -m attrition.models.evaluate \
		--model artifacts/models/xgb_attrition_final.pkl \
		--threshold artifacts/models/threshold_optimizado.pkl \
		--features artifacts/features_matrix.csv

explain:
	python -m attrition.models.explain \
		--model artifacts/models/xgb_attrition_final.pkl \
		--features artifacts/features_matrix.csv \
		--output artifacts/shap_beeswarm.png

test:
	pytest tests/

all: process engineer train evaluate explain test
