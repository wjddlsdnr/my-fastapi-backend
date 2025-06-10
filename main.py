import os
import stat
import re
import models
import numpy as np
from fastapi import FastAPI, File, UploadFile, Depends, Query, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from ocr import extract_text_from_image
from database import SessionLocal, engine
from models import Base, OCRSentence
from semantic_search import (
    get_all_texts,
    build_faiss_index,
    get_text_embedding,
    remove_faiss_ids,
)
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import User, Base
from sqlalchemy import create_engine
from passlib.context import CryptContext
from jose import jwt
from pydantic import BaseModel
from fastapi import Header

SECRET_KEY = "wjddlsdnr8832"
ALGORITHM = "HS256"

model = SentenceTransformer("BAAI/bge-m3")
Base.metadata.create_all(bind=engine)
app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # 개발용
        "https://your-frontend.onrender.com"  # 실제 배포용 프론트 주소로 변경
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "내_매우_강력한_비밀키"
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
engine = create_engine("sqlite:///./ocr_data.db")
Base.metadata.create_all(bind=engine)

class UserCreate(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

@app.post("/signup/", response_model=UserOut)
def signup(user: UserCreate):
    session = Session(bind=engine)
    if session.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="이미 가입된 아이디입니다")
    db_user = User(username=user.username, hashed_password=get_password_hash(user.password))
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user

@app.post("/login/")
def login(user: UserCreate):
    session = Session(bind=engine)
    db_user = session.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="로그인 실패")
    token = jwt.encode({"sub": user.username}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

def highlight_keyword(text: str, keyword: str) -> str:
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
    token: str = Header(...),  # 프론트에서 token 헤더로 JWT 보내야 함
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="인증 실패")
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=401, detail="사용자 없음")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="인증 실패")

def split_into_sentences(text):
    if isinstance(text, list):
        parts = text
    else:
        parts = re.split(r'[.\n]', text)
    return [s.strip() for s in parts if len(s.strip()) > 1]

faiss_index = None
indexed_texts = []
embeddings = []

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    global faiss_index, indexed_texts, embeddings
    try:
        contents = await file.read()
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        raw_result = extract_text_from_image(contents)
        print("📦 OCR 결과 (원본):", raw_result)

        if isinstance(raw_result, list) and isinstance(raw_result[0], tuple):
            text_list = [item[1] for item in raw_result]
        else:
            text_list = raw_result

        text = " ".join(text_list)
        print("🧩 변환된 문장형 텍스트:", text)

        sentences = split_into_sentences(text)
        print("✂️ 분리된 문장 수:", len(sentences))
        print("✂️ 분리된 문장 목록:", sentences)

        if not sentences:
            return {"message": "문장이 추출되지 않음", "uploaded": file.filename}

        texts = get_all_texts()
        faiss_index, indexed_texts, embeddings = build_faiss_index(texts)

        current_vector_id = faiss_index.ntotal if faiss_index else 0
        vectors, ids = [], []
        for s in sentences:
            db.add(OCRSentence(
                image_path=save_path,
                sentence=s,
                faiss_id=current_vector_id
            ))
            embedding = get_text_embedding(s)
            vectors.append(embedding)
            ids.append(current_vector_id)
            current_vector_id += 1

        db.commit()
        print(f"✅ 총 {len(sentences)}개 문장 저장 완료")

        texts = get_all_texts()
        faiss_index, indexed_texts, embeddings = build_faiss_index(texts)
        print("✅ FAISS 인덱스 전체 재구축 완료")
        print("🧠 현재 인덱스 상태: 벡터 수", faiss_index.ntotal)

        return {
            "uploaded": file.filename,
            "num_sentences": len(sentences),
            "preview": sentences[:3],
            "image_path": f"uploads/{file.filename}",
        }

    except Exception as e:
        print("❌ 업로드 중 오류:", e)
        return {"error": str(e)}
@app.get("/myinfo/")
def get_myinfo(user: User = Depends(get_current_user)):
    return {"username": user.username, "id": user.id}


@app.delete("/delete_image/{filename:path}")
def delete_image(filename: str = Path(...), db: Session = Depends(get_db)):
    global faiss_index, indexed_texts, embeddings
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="이미지 파일이 존재하지 않습니다.")

    try:
        os.chmod(file_path, stat.S_IWRITE)
        os.remove(file_path)
        sentences_to_delete = db.query(OCRSentence).filter(OCRSentence.image_path == file_path).all()
        faiss_ids = [s.faiss_id for s in sentences_to_delete]
        remove_faiss_ids(faiss_index, faiss_ids)
        for s in sentences_to_delete:
            db.delete(s)
        db.commit()
        texts = get_all_texts()
        faiss_index, indexed_texts, embeddings = build_faiss_index(texts)
        print("✅ 이미지 및 관련 OCR문장, 인덱스 삭제 완료")
        return JSONResponse(content={"message": "삭제 완료"}, status_code=200)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("❌ 이미지 삭제 중 오류:", e)
        raise HTTPException(status_code=500, detail=f"삭제 실패: {str(e)}")

@app.get("/semantic_search/")
def semantic_search_api(query: str):
    global faiss_index, indexed_texts, embeddings
    if faiss_index is None or not indexed_texts:
        return {"error": "FAISS 인덱스가 초기화되지 않았거나 문장이 없음"}

    try:
        query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)
        top_k = 10
        threshold = 0.45
        D, I = faiss_index.search(query_vec.astype("float32"), top_k)
        from collections import defaultdict
        results_dict = defaultdict(list)
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            matched_sentence, image_path = indexed_texts[idx]
            if score >= threshold:
                highlighted = highlight_keyword(matched_sentence, query)
                results_dict[image_path].append({
                    "highlighted": highlighted,
                    "original": matched_sentence,
                    "similarity": float(score)
                })
        results = []
        for image_path, matched_sentences in results_dict.items():
            if matched_sentences:
                results.append({
                    "image_path": image_path,
                    "matches": matched_sentences
                })
        return results
    except Exception as e:
        print("❌ 문맥 검색 중 오류:", e)
        return {"error": str(e)}
@app.get("/images/")
def get_image_list():
    image_folder = "uploaded_images"  # 실제 업로드 폴더 경로 맞게 수정
    files = []
    for fname in os.listdir(image_folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            files.append(f"{image_folder}/{fname}")
    return {"images": files}
    
@app.post("/highlighted_image/")
async def highlight_image(request: Request):
    data = await request.json()
    image_path = data.get("image_path")
    keyword = data.get("query")
    if not image_path or not keyword:
        return {"error": "image_path와 query는 필수입니다."}
    full_path = os.path.join(image_path) if os.path.exists(image_path) else os.path.join("backend", image_path)
    if not os.path.exists(full_path):
        return {"error": f"이미지 파일을 찾을 수 없습니다: {full_path}"}
    import easyocr
    import cv2
    from io import BytesIO
    image = cv2.imread(full_path)
    if image is None:
        return {"error": "이미지를 읽는 데 실패했습니다."}
    reader = easyocr.Reader(['ko', 'en'])
    results = reader.readtext(image)
    found = False
    for (bbox, text, conf) in results:
        if keyword.lower() in text.lower():
            found = True
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 3)
            cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    _, buffer = cv2.imencode('.png', image)
    io_buf = BytesIO(buffer)
    return StreamingResponse(io_buf, media_type="image/png")

@app.on_event("startup")
def load_faiss_index():
    global faiss_index, indexed_texts, embeddings
    texts = get_all_texts()
    faiss_index, indexed_texts, embeddings = build_faiss_index(texts)
    print("✅ 서버 시작 시 FAISS 인덱스 자동 초기화 완료, 문장 개수:", len(indexed_texts))

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
