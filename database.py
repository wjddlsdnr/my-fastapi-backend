import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
# database.py에 추가
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from models import Base, User, OCRSentence

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)


SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ocr_data.db")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
