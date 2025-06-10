from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base  # models.py에서 가져오기

SQLALCHEMY_DATABASE_URL = "sqlite:///./ocr_data.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ 테이블 생성
Base.metadata.create_all(bind=engine)



def get_all_texts():
    from models import OCRSentence
    db: Session = SessionLocal()
    data = db.query(OCRSentence).all()
    texts = [(item.sentence, item.faiss_id) for item in data]
    db.close()
    return texts


def delete_ocr_by_filename(filename: str):
    conn = sqlite3.connect("db/your_database.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ocr_data WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()




