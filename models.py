from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base.metadata.create_all(bind=engine) 

class OCRSentence(Base):
    __tablename__ = "ocr_sentences"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String)
    sentence = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    faiss_id = Column(Integer, unique=True, index=True)  # ✅ FAISS ID는 반드시 UNIQUE!
