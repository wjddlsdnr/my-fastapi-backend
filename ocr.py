# ocr.py
from google.cloud import vision
import io

def extract_text_from_image(image_bytes):
    """구글 Vision API로 OCR 텍스트 추출"""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        return []

    # 첫 번째 항목이 전체 텍스트, 두 번째부터는 각 영역별 텍스트
    full_text = texts[0].description.strip()
    # 한 줄씩 문장 분리
    sentences = [line.strip() for line in full_text.split('\n') if line.strip()]

    return sentences
