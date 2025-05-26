# HR Employee Attrition Analysis & Prediction

Projeto de análise e predição da rotatividade de funcionários usando dados reais de RH. Inclui EDA (análise exploratória), engenharia de atributos e modelagem preditiva com machine learning.

## 🔍 Objetivo
Ajudar times de RH a identificar padrões de desligamento e prever possíveis saídas, com base em dados históricos.

## 🧰 Tecnologias usadas
- Python 3.10  
- pandas, numpy, seaborn, matplotlib  
- scikit-learn, xgboost  
- Jupyter Notebook  
- VSCode  

## 📁 Estrutura do Projeto
```
EMPLOYEE_ATTRITION_PROJECT/
├── artifacts/
│   └── models/
│       ├── smote_resampler.pkl
│       ├── threshold_optimizado.pkl
│       └── xgb_attrition_final.pkl
├── dashboards/
│   └── .gitkeep
├── data/
│   ├── processed/
│   │   └── employee_attrition_processed.csv
│   └── raw/
│       └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│   ├── eda_attrition_funcionarios.ipynb
│   └── modeling_ml.ipynb
├── outputs/
│   └── .gitkeep
├── reports/
│   └── figures/
├── src/
│   └── .gitkeep
├── tests/
│   └── test_example.py
├── LICENSE
└── README.md
```

## 📊 Etapas do Projeto
1. **Coleta de dados** – Dataset público do Kaggle  
2. **Análise exploratória (EDA)** – Identificação de padrões relevantes  
3. **Pré-processamento e Feature Engineering**  
4. **Modelagem preditiva** – XGBoost otimizado com tuning de threshold e balanceamento SMOTE  
5. **Exportação de modelo treinado** – Para uso futuro em aplicações reais  

## 📦 Dataset
[Kaggle: IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## 📌 Resultados
- Modelo final: XGBoost  
- F1-Score (classe minoritária): `0.6118`  
- Principais variáveis associadas à saída: `OverTime`, `Age`, `JobRole`, `EnvironmentSatisfaction`, entre outras  
- Modelo exportado com `joblib` para uso em produção

## 📫 Contato
**Mateus Cabral**  
📧 mateuscsq@email.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mateus-cabral-b25aa3250/)