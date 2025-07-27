#src/attrition/causal_analysis.py

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, String, Float, Date, Integer
from sqlalchemy.ext.declarative import declarative_base
from datetime import date
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()
class CausalInsight(Base):
    __tablename__ = 'causal_insights'
    id = Column(Integer, primary_key=True)
    factor_causal = Column(String, unique=True, nullable=False) 
    efeito_causal = Column(Float, nullable=False)
    unidade_efeito = Column(String, nullable=False)
    data_analise = Column(Date, nullable=False, default=date.today)
    observacoes = Column(String)

    def __repr__(self):
        return f"<CausalInsight(factor_causal='{self.factor_causal}', efeito_causal={self.efeito_causal})>"

def save_causal_insight(factor: str, effect: float, unit: str, observation: str):
    """Salva ou atualiza um insight causal no banco de dados."""
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("DATABASE_URL não configurada. Não foi possível salvar insights causais.")
        return

    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine) 

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        insight = session.query(CausalInsight).filter_by(factor_causal=factor).first()
        if not insight:
            insight = CausalInsight(factor_causal=factor)
        
        insight.efeito_causal = effect
        insight.unidade_efeito = unit
        insight.data_analise = date.today()
        insight.observacoes = observation
        
        session.add(insight)
        session.commit()
        print(f"Insight para '{factor}' salvo/atualizado no BD com efeito: {effect}")
    except Exception as e:
        session.rollback()
        print(f"Erro ao salvar insight para '{factor}': {e}")
    finally:
        session.close()
