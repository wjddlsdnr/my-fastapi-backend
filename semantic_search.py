import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from database import SessionLocal
from models import OCRSentence

model = SentenceTransformer("BAAI/bge-m3")


def get_all_texts():
    db = SessionLocal()
    data = db.query(OCRSentence).all()
    texts = [(item.faiss_id, item.sentence, item.image_path) for item in data]
    db.close()
    return texts

def get_text_embedding(text: str):
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

def build_faiss_index(text_tuples):
    if not text_tuples:
        return None, [], []
    ids, sentences, image_paths = zip(*text_tuples)
    embeddings = [get_text_embedding(s) for s in sentences]
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))
    # (문장, 이미지경로)로 저장!
    return index, list(zip(sentences, image_paths)), embeddings

def highlight_keyword(text: str, keyword: str) -> str:
    import re
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

def remove_faiss_ids(index: faiss.IndexIDMap, ids_to_remove: List[int]):
    if not ids_to_remove or index is None:
        return
    ids_array = np.array(ids_to_remove).astype(np.int64)
    index.remove_ids(ids_array)
