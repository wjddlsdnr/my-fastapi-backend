# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, User, OCRSentence

USER_DB_PATH = "./user.db"
USER_ENGINE = create_engine(f"sqlite:///{USER_DB_PATH}")
UserSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=USER_ENGINE)
# 테이블 생성을 여기서 하지 않아도 됨(main.py에서 한 번만)

def get_user_db(username: str):
    db_path = f"./ocr_data_{username}.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()
