from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class OCRSentence(Base):
    __tablename__ = "ocr_sentences"
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String)
    sentence = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    faiss_id = Column(Integer, unique=True, index=True)
