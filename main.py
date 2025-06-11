import os
import stat
import re
import numpy as np
from fastapi import FastAPI, File, UploadFile, Depends, Query, HTTPException, Path, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# 모델은 오로지 models.py에서!
from models import Base, User, OCRSentence

# 세션 등은 database.py에서
from database import UserSessionLocal, get_user_db

from sentence_transformers import SentenceTransformer
from passlib.context import CryptContext
from jose import jwt
from pydantic import BaseModel
from ocr import extract_text_from_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hopeful-education-production.up.railway.app",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "wjddlsdnr8832"
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploaded_images", StaticFiles(directory="uploaded_images"), name="uploaded_images")

# --- 인증 모델 등 ---
class UserCreate(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str

def get_user_session():
    db = UserSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_current_user(token: str = Header(...), db: Session = Depends(get_user_session)):
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

@app.post("/signup/", response_model=UserOut)
def signup(user: UserCreate, db: Session = Depends(get_user_session)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="이미 가입된 아이디입니다")
    db_user = User(username=user.username, hashed_password=get_password_hash(user.password))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/login/")
def login(user: UserCreate, db: Session = Depends(get_user_session)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="로그인 실패")
    token = jwt.encode({"sub": user.username}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

# ... 이하 기존 라우터 그대로 복붙 ...

# 예시: 업로드, 내 사진, 삭제, 검색 등등


# ====== 문장 분리 ======
def split_into_sentences(text):
    if isinstance(text, list):
        parts = text
    else:
        parts = re.split(r'[.\n]', text)
    return [s.strip() for s in parts if len(s.strip()) > 1]

def highlight_keyword(text: str, keyword: str) -> str:
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

# ====== FAISS (검색 인덱스, 유저별 관리 권장) ======
model = SentenceTransformer("BAAI/bge-m3")

# ====== 내 정보 확인 ======
@app.get("/myinfo/")
def get_myinfo(user: User = Depends(get_current_user)):
    return {"username": user.username, "id": user.id}

# ====== 파일 업로드 (유저별 DB 사용) ======
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    db = get_user_db(user.username)
    try:
        contents = await file.read()
        user_folder = os.path.join(UPLOAD_DIR, user.username)
        os.makedirs(user_folder, exist_ok=True)
        save_path = os.path.join(user_folder, file.filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        image_rel_path = f"{UPLOAD_DIR}/{user.username}/{file.filename}"
        raw_result = extract_text_from_image(contents)
        if isinstance(raw_result, list) and isinstance(raw_result[0], tuple):
            text_list = [item[1] for item in raw_result]
        else:
            text_list = raw_result
        text = " ".join(text_list)
        sentences = split_into_sentences(text)
        if not sentences:
            return {"message": "문장이 추출되지 않음", "uploaded": file.filename}
        current_vector_id = 0
        vectors, ids = [], []
        for s in sentences:
            db.add(OCRSentence(
                image_path=image_rel_path,
                sentence=s,
                faiss_id=current_vector_id
            ))
            embedding = get_text_embedding(s)
            vectors.append(embedding)
            ids.append(current_vector_id)
            current_vector_id += 1
        db.commit()
        return {
            "uploaded": file.filename,
            "num_sentences": len(sentences),
            "preview": sentences[:3],
            "image_path": image_rel_path,
        }
    except Exception as e:
        print("❌ 업로드 중 오류:", e)
        return {"error": str(e)}

# ====== 내 사진(갤러리, 유저별 DB 사용) ======
@app.get("/my_images/")
def my_images(user: User = Depends(get_current_user)):
    db = get_user_db(user.username)
    user_folder = os.path.join(UPLOAD_DIR, user.username)
    if not os.path.exists(user_folder):
        return {"images": []}
    files = [
        f"{UPLOAD_DIR}/{user.username}/{fname}"
        for fname in os.listdir(user_folder)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
    ]
    return {"images": files}

# ====== 이미지 삭제 (유저별 DB 사용) ======
@app.delete("/delete_image/{filename:path}")
def delete_image(filename: str = Path(...), user: User = Depends(get_current_user)):
    db = get_user_db(user.username)
    image_rel_path = f"{UPLOAD_DIR}/{user.username}/{filename}"
    file_path = os.path.join(UPLOAD_DIR, user.username, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="이미지 파일이 존재하지 않습니다.")

    try:
        os.chmod(file_path, stat.S_IWRITE)
        os.remove(file_path)
        sentences_to_delete = db.query(OCRSentence).filter(OCRSentence.image_path == image_rel_path).all()
        for s in sentences_to_delete:
            db.delete(s)
        db.commit()
        print("✅ 이미지 및 관련 OCR문장 삭제 완료")
        return JSONResponse(content={"message": "삭제 완료"}, status_code=200)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("❌ 이미지 삭제 중 오류:", e)
        raise HTTPException(status_code=500, detail=f"삭제 실패: {str(e)}")

# ====== 문맥 검색 (유저별 DB 사용) ======
@app.get("/semantic_search/")
def semantic_search_api(query: str, user: User = Depends(get_current_user)):
    db = get_user_db(user.username)
    all_sentences = db.query(OCRSentence).all()
    indexed_texts = [(s.sentence, s.image_path) for s in all_sentences]
    if not indexed_texts:
        return {"error": "문장이 없음"}
    try:
        sentences = [s.sentence for s in all_sentences]
        query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)
        corpus_embeds = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        # Cosine similarity top-k 검색 (직접 구현)
        scores = np.dot(corpus_embeds, query_vec.T).flatten()
        top_k_idx = np.argsort(scores)[::-1][:10]
        threshold = 0.45
        from collections import defaultdict
        results_dict = defaultdict(list)
        for idx in top_k_idx:
            if scores[idx] < threshold:
                continue
            matched_sentence = sentences[idx]
            image_path = all_sentences[idx].image_path
            highlighted = highlight_keyword(matched_sentence, query)
            results_dict[image_path].append({
                "highlighted": highlighted,
                "original": matched_sentence,
                "similarity": float(scores[idx])
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

# ====== 이미지 하이라이트 (OCR+CV) ======
@app.post("/highlighted_image/")
async def highlight_image(request: Request):
    data = await request.json()
    image_path = data.get("image_path")
    keyword = data.get("query")
    if not image_path or not keyword:
        return {"error": "image_path와 query는 필수입니다."}
    full_path = os.path.join(image_path)
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
    for (bbox, text, conf) in results:
        if keyword.lower() in text.lower():
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 3)
            cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    _, buffer = cv2.imencode('.png', image)
    io_buf = BytesIO(buffer)
    return StreamingResponse(io_buf, media_type="image/png")
