#!pip install langchain langchain-community langchain-google-genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np

embeddings = GoogleGenerativeAIEmbeddings(
  model="models/gemini-embedding-001" # Google의 임베딩 모델
)

# 여러 문서 임베딩
v = np.array(embeddings.embed_documents([
  "재택근무 신청은 어떻게 하나요?",
  "재택 근무를 신청하는 방법을 알려줘.",
  "가까운 회사 셔틀버스 승차 장소는 어디인가요?"
]))

# 거리 계산
print(np.sum(v[0] * v[1]), np.sum(v[1] * v[2]), np.sum(v[2] * v[0]))