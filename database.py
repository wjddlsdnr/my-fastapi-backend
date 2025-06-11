import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# 공용 회원 DB
USER_DB_PATH = "./user.db"
USER_ENGINE = create_engine(f"sqlite:///{USER_DB_PATH}")
UserSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=USER_ENGINE)

# 유저별 DB 생성 함수
def get_user_db(username: str):
    db_path = f"./ocr_data_{username}.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)  # 테이블 보장
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()
