# !pip install google-cloud-aiplatform langchain langchain-google-genai langchain-google-vertexai langchain-community

from google.cloud import aiplatform
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VectorSearchVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 환경 설정
# ==========================================
PROJECT_ID = "project-id"            # 프로젝트 ID 입력 (Console 대시보드에서 확인)
LOCATION = "us-central1"             # 리전 (서울은 asia-northeast3)
INDEX_ID = "your-index-id"           # Vertex AI Vector Search Index ID
ENDPOINT_ID = "your-endpoint-id"     # Vertex AI Vector Search Endpoint ID

# ==========================================
# 2. 인증 설정
# ==========================================
import os

# Colab 환경 감지
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("🔧 Colab 환경 감지 - gcloud 인증 시작")
    print("\n다음 단계를 따라주세요:")
    print("1. 아래 명령어 실행 후 나오는 URL을 브라우저에서 열기")
    print("2. Google Cloud 크레딧 계정으로 로그인")
    print("3. 인증 코드를 복사하여 입력\n")

    # gcloud 인증
    os.system("gcloud auth login --no-launch-browser")

    # 프로젝트 설정
    os.system(f"gcloud config set project {PROJECT_ID}")

    # Application Default Credentials 설정
    os.system("gcloud auth application-default login --no-launch-browser")

    print("\n✅ Colab 인증 완료")
else:
    # 로컬 환경: 터미널에서 한 번만 실행
    # $ gcloud auth application-default login
    print("💻 로컬 환경 - ADC 사용")
    print("터미널에서 다음 명령어를 실행하세요:")
    print("$ gcloud auth application-default login")

# 필요한 권한:
# - Vertex AI User (roles/aiplatform.user)
# - Storage Object Admin (roles/storage.objectAdmin) - GCS 사용 시

# Vertex AI 초기화
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ==========================================
# 3. LangChain 컴포넌트 초기화
# ==========================================

# 임베딩 모델 초기화 (Google Generative AI Embeddings)
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",  # 최신 임베딩 모델
    project=PROJECT_ID,
    location=LOCATION
)

# LLM 초기화 (Google Generative AI)
llm = GoogleGenerativeAI(
    model="gemini-1.5-pro",  # 또는 gemini-1.5-flash
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

vector_store = VectorSearchVectorStore.from_components(
    project_id=PROJECT_ID,
    region=LOCATION,
    index_id=INDEX_ID,
    endpoint_id=ENDPOINT_ID,
    embedding=embeddings,
)

print("✅ Vector Store 연결 완료")

# 문서를 Vector Store에 추가 (최초 실행 시 또는 문서 업데이트 시)
# 주석을 해제하여 사용:
# texts = [doc.page_content for doc in split_docs]
# metadatas = [doc.metadata for doc in split_docs]
# vector_store.add_texts(texts=texts, metadatas=metadatas)
# print(f"✅ {len(texts)}개 문서를 Vector Store에 추가했습니다")

# ==========================================
# Option 2: 기존 Index에 새로운 문서 추가
# ==========================================
# 주의: Index와 Endpoint가 이미 생성되어 있어야 합니다
#
# # Vector Store 초기화
# vector_store = VectorSearchVectorStore.from_components(
#     project_id=PROJECT_ID,
#     region=LOCATION,
#     index_id=INDEX_ID,
#     endpoint_id=ENDPOINT_ID,
#     embedding=embeddings,
# )
#
# # 문서 추가 (텍스트와 메타데이터 분리)
# texts = [doc.page_content for doc in split_docs]
# metadatas = [doc.metadata for doc in split_docs]
#
# # Vector Store에 문서 추가
# vector_store.add_texts(texts=texts, metadatas=metadatas)
# print(f"✅ {len(texts)}개 문서를 Vector Store에 추가했습니다")

# ==========================================
# 6. RAG Chain 구성 (LCEL 방식)
# ==========================================

# Retriever 생성 (유사도 검색)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 상위 3개 문서 검색
)

# RAG 프롬프트 템플릿
template = """다음 컨텍스트를 사용하여 질문에 답변하세요.
답을 모르면 모른다고 답변하세요. 답변은 간결하게 3-4 문장으로 작성하세요.

컨텍스트: {context}

질문: {question}

답변:"""

prompt = ChatPromptTemplate.from_template(template)

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL Chain 구성
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
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
answer = rag_chain.invoke(question)

# 결과 출력
print(f"\n💡 답변:\n{answer}")

# 참조 문서 확인
print("\n📚 참조 문서:")
docs = retriever.invoke(question)
for i, doc in enumerate(docs, 1):
    print(f"  {i}. {doc.page_content} (출처: {doc.metadata.get('source', 'N/A')})")

# ==========================================
# 8. 추가 질문 예제
# ==========================================

def ask_question(question: str):
    """RAG 시스템에 질문하는 헬퍼 함수"""
    print(f"\n{'='*60}")
    print(f"❓ 질문: {question}")
    print('='*60)

    # RAG 실행
    answer = rag_chain.invoke(question)

    print(f"\n💡 답변:\n{answer}")

    # 참조 문서 출력
    print("\n📚 참조 문서:")
    docs = retriever.invoke(question)
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.page_content}")

    return {"answer": answer, "source_documents": docs}

# 여러 질문 테스트
if __name__ == "__main__":
    questions = [
        "LG이노텍은 언제 설립되었나요?",
        "LG이노텍의 본사는 어디에 있나요?",
        "LG이노텍의 주요 고객은 누구인가요?",
    ]

    for q in questions:
        ask_question(q)
