# !pip install langchain langchain-community langchain-google-vertexai faiss-cpu

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# API 키 설정
os.environ["GOOGLE_API_KEY"] = "AIzaSyBNUhCrEfJjNIIfy4gE78N_GwuMr9JTUcM"

# 1. 실습용 데이터 (가상의 사내 규정)
raw_text = """
[주식회사 랭체인 재택근무 규정]
1. 주 2회 재택근무가 가능하며, 팀장 승인이 필요합니다.
2. 재택근무 신청은 전날 오후 6시까지 사내 시스템에 등록해야 합니다.
3. 코어 타임(오전 11시 ~ 오후 3시)에는 반드시 메신저에 접속해 있어야 합니다.
4. 제주도 워케이션은 연 1회, 최대 2주간 지원됩니다.
"""

# 문서 객체 생성
docs = [Document(page_content=raw_text)]

# 2. 임베딩 모델 설정 (Google Generative AI)
# text-embedding-004는 Google의 최신 임베딩 모델입니다.
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

print("⏳ 문서 벡터화 진행 중...")

# 3. FAISS 인덱스 생성 (메모리 상에 구축)
db = FAISS.from_documents(docs, embeddings)

# 4. 로컬 디스크에 저장 (핵심!)
# 실행 경로에 'my_company_index'라는 폴더가 생성됩니다.
db.save_local("my_company_index")

print("✅ 인덱스 저장 완료! ('my_company_index' 폴더 확인)")