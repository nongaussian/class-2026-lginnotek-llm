import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 설정 (Google API 키 필요)
# ==========================================
# 환경 변수 설정 필요: GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"

# LLM 및 임베딩 모델 초기화
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0,  # 사실 기반 답변을 위해 0으로 설정
    convert_system_message_to_human=True  # 시스템 메시지를 사용자 메시지로 변환
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# ==========================================
# 2. 데이터 준비 (가상의 사내 규정 데이터)
# ==========================================
raw_text = """
[2026년 주식회사 랭체인 사내 업무 가이드라인]

1. 근무 시간
- 기본 근무 시간은 오전 10시부터 오후 7시까지입니다.
- 유연 근무제를 시행하고 있어, 오전 8시~11시 사이에 자유롭게 출근이 가능합니다.
- 점심시간은 12시 30분부터 1시 30분까지 1시간입니다.

2. 재택근무 규정 (RAG 핵심 테스트 구간)
- 주 2회 재택근무가 가능합니다. (화요일, 목요일 권장)
- 재택근무 신청은 전날 오후 4시까지 사내 메신저 '슬랙'의 #wfh 채널에 남겨야 합니다.
- 긴급한 회의가 잡힐 경우, 팀장의 승인 하에 재택근무가 취소될 수 있습니다.
- 제주도 등 원격지 근무(워케이션)는 분기당 1회, 최대 1주일 지원됩니다.

3. 비용 청구
- 야근 식대는 오후 8시 이후 퇴근 시 15,000원까지 지원됩니다.
- 법인카드 영수증은 매월 말일까지 재무팀에 실물로 제출해야 합니다.
"""

# LangChain Document 형식으로 변환
docs = [Document(page_content=raw_text)]

# ==========================================
# 3. 데이터 분할 (Splitting)
# ==========================================
# 긴 문서를 처리하기 좋게 작은 청크(Chunk)로 나눕니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(docs)

print(f"총 분할된 청크 수: {len(splits)}개")

# ==========================================
# 4. 벡터 저장소 구축 (Indexing)
# ==========================================
# 문서를 벡터화하여 FAISS(로컬 검색기)에 저장합니다.
# 실제 서비스에서는 이 부분을 Vertex AI Vector Search로 대체합니다.
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# 검색기(Retriever) 생성
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2} # 가장 유사한 문서 2개만 참조
)

# ==========================================
# 5. RAG 체인 생성 (Modern LangChain 1.x approach)
# ==========================================
# 프롬프트 템플릿: 검색된 정보(Context)를 기반으로만 답하도록 지시
template = """당신은 회사의 인사 규정을 안내하는 친절한 AI 봇입니다.
아래의 [참고 문서] 내용을 바탕으로 질문에 답하세요.
문서에 없는 내용은 "죄송합니다, 해당 내용은 규정에 나와있지 않습니다."라고 답하세요.

[참고 문서]
{context}

질문: {question}
답변:"""

prompt = ChatPromptTemplate.from_template(template)

# 문서를 문자열로 포맷하는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL (LangChain Expression Language)로 체인 구성
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 6. 테스트 및 실행
# ==========================================
def ask_bot(question):
    print(f"\n🙋 질문: {question}")
    result = qa_chain.invoke(question)
    print(f"🤖 답변: {result}")

    # 참조 문서를 보려면 retriever를 직접 호출할 수 있습니다
    # docs = retriever.invoke(question)
    # print(f"📄 참조 문서: {docs[0].page_content[:50]}...")

# 테스트 케이스
ask_bot("재택근무 신청은 언제까지 해야 해?")
ask_bot("야근 식대는 얼마까지 지원돼?")
ask_bot("연차는 며칠까지 쓸 수 있어?") # 문서에 없는 내용 테스트