import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def split_text_into_sections(text, delimiter="\n\n"):
    sections = text.split(delimiter)
    return sections

if __name__ == "__main__":
    # 파일 경로 설정
    pdf_path = r"C:\Users\yubeen\OneDrive\Desktop\car_ins.pdf"
    output_text_path = r"C:\Users\yubeen\OneDrive\Desktop\약관.txt"

 # 1. PDF에서 텍스트 추출
    insurance_text = extract_text_from_pdf(pdf_path)

    # 2. 텍스트를 파일로 저장
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(insurance_text)

    # 3. 텍스트를 섹션별로 분할
    sections = split_text_into_sections(insurance_text)

    # 4. 섹션 임베딩 생성
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    section_embeddings = embedding_model.encode(sections)

    # 5. 사용자 질문 받기
    question = input("보험 관련 질문을 입력하세요: ")

    # 6. 질문 임베딩 생성
    question_embedding = embedding_model.encode([question])

    # 7. 유사도 계산
    similarities = cosine_similarity(question_embedding, section_embeddings)

    # 8. 가장 유사한 섹션 찾기
    most_similar_section_idx = np.argmax(similarities)

    # 9. 가장 유사한 섹션 출력
    print(f"가장 유사한 섹션: {sections[most_similar_section_idx]}")
   
