from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    hashed_password = Column(String)

class OCRSentence(Base):
    __tablename__ = "ocr_sentences"
    id = Column(Integer, primary_key=True)
    image_path = Column(String)
    sentence = Column(String)
    created_at = Column(DateTime)
    faiss_id = Column(Integer)