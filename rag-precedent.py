# !pip install google-cloud-aiplatform langchain langchain-google-vertexai langchain-community

import os
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import MatchingEngine
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# ==========================================
# 1. 환경 설정
# ==========================================
PROJECT_ID = "your-project-id"       # 프로젝트 ID 입력 (Console 대시보드에서 확인)
LOCATION = "us-central1"             # 리전 (서울은 asia-northeast3)
INDEX_ID = "your-index-id"           # Vertex AI Vector Search Index ID
ENDPOINT_ID = "your-endpoint-id"     # Vertex AI Vector Search Endpoint ID

# ==========================================
# 2. 인증 설정 (서비스 계정 JSON 키 사용)
# ==========================================
# Google Cloud Console에서 서비스 계정 JSON 키 다운로드 후 경로 지정
# 1. Console → "IAM 및 관리자" → "서비스 계정"
# 2. 서비스 계정 만들기 → 역할 부여: "Vertex AI User", "Storage Admin"
# 3. 키 생성 → JSON 다운로드
SERVICE_ACCOUNT_KEY_PATH = "path/to/your-service-account-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH

# Vertex AI 초기화
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ==========================================
# 3. LangChain 컴포넌트 초기화
# ==========================================

# 임베딩 모델 초기화 (Vertex AI Embeddings)
embeddings = VertexAIEmbeddings(
    model_name="textembedding-gecko@003",  # 또는 text-embedding-004
    project=PROJECT_ID,
    location=LOCATION
)

# LLM 초기화 (Vertex AI Gemini)
llm = VertexAI(
    model_name="gemini-1.5-pro",  # 또는 gemini-1.5-flash
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0.2,
    max_output_tokens=1024,
)

# ==========================================
# 4. 문서 준비 및 Vector Store 연결
# ==========================================

# 예제 문서 데이터 (실제로는 PDF, 웹사이트 등에서 로드)
sample_texts = [
    "LG이노텍은 전자부품 전문기업입니다. 카메라 모듈, 기판, 모터 등을 생산합니다.",
    "LG이노텍의 주요 제품은 스마트폰용 카메라 모듈입니다.",
    "회사는 2008년 LG전자의 부품 사업부가 분사하여 설립되었습니다.",
    "본사는 서울특별시 중구에 위치하고 있습니다.",
    "LG이노텍은 Apple의 주요 카메라 모듈 공급업체입니다.",
]

# Document 객체로 변환
documents = [Document(page_content=text, metadata={"source": f"doc_{i}"})
             for i, text in enumerate(sample_texts)]

# 텍스트 분할 (긴 문서의 경우)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
split_docs = text_splitter.split_documents(documents)

print(f"📄 총 {len(split_docs)}개의 문서 청크 준비 완료")

# ==========================================
# 5. Vertex AI Vector Search 연결
# ==========================================
# Option 1: 기존 Index 사용 (이미 생성된 경우)
# INDEX_ID와 ENDPOINT_ID를 Console에서 확인하여 입력

vector_store = MatchingEngine.from_components(
    project_id=PROJECT_ID,
    region=LOCATION,
    index_id=INDEX_ID,
    endpoint_id=ENDPOINT_ID,
    embedding=embeddings,
)

print("✅ Vector Store 연결 완료")

# ==========================================
# Option 2: 새로운 문서로 Vector Store 생성 및 업로드
# ==========================================
# 주의: 이 방법은 새로운 Index와 Endpoint를 생성합니다 (시간 소요: 약 1시간)
#
# vector_store = MatchingEngine.from_documents(
#     documents=split_docs,
#     embedding=embeddings,
#     project_id=PROJECT_ID,
#     region=LOCATION,
#     gcs_bucket_name="your-bucket-name",  # GCS 버킷 필요
#     index_id="my_langchain_index",
#     endpoint_id="my_langchain_endpoint",
# )

# ==========================================
# 6. RAG Chain 구성
# ==========================================

# Retriever 생성 (유사도 검색)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 상위 3개 문서 검색
)

# RetrievalQA Chain 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 검색된 문서를 모두 컨텍스트에 포함
    retriever=retriever,
    return_source_documents=True,  # 검색된 문서도 함께 반환
    verbose=True
)

print("✅ RAG Chain 구성 완료")

# ==========================================
# 7. 질의응답 실행
# ==========================================

# 사용자 질문
question = "LG이노텍의 주요 제품은 무엇인가요?"

print(f"\n❓ 질문: {question}")
print("="*60)

# RAG 실행
result = qa_chain.invoke({"query": question})

# 결과 출력
print(f"\n💡 답변:\n{result['result']}")
print("\n📚 참조 문서:")
for i, doc in enumerate(result['source_documents'], 1):
    print(f"  {i}. {doc.page_content} (출처: {doc.metadata.get('source', 'N/A')})")

# ==========================================
# 8. 추가 질문 예제
# ==========================================

def ask_question(question: str):
    """RAG 시스템에 질문하는 헬퍼 함수"""
    print(f"\n{'='*60}")
    print(f"❓ 질문: {question}")
    print('='*60)

    result = qa_chain.invoke({"query": question})

    print(f"\n💡 답변:\n{result['result']}")
    print("\n📚 참조 문서:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"  {i}. {doc.page_content}")

    return result

# 여러 질문 테스트
if __name__ == "__main__":
    questions = [
        "LG이노텍은 언제 설립되었나요?",
        "LG이노텍의 본사는 어디에 있나요?",
        "LG이노텍의 주요 고객은 누구인가요?",
    ]

    for q in questions:
        ask_question(q)
