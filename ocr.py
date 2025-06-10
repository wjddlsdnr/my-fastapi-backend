import os
from google.cloud import vision

def extract_text_from_image(image_bytes):
    cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_json or not os.path.exists(cred_json):
        raise RuntimeError("Google Vision 서비스 계정 키(.json) 환경 변수 설정 필요!")
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return []
    full_text = texts[0].description.strip()
    sentences = [line.strip() for line in full_text.split('\n') if line.strip()]
    return sentences
