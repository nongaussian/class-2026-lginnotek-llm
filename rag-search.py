from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# API 키 설정
os.environ["GOOGLE_API_KEY"] = "AIzaSyDWeeAT3iJ1nUAk3UrX1LVeIMlVv2gpBV4"

# 1. 저장된 인덱스 불러오기
# rag-build-index.py와 동일한 임베딩 모델 사용
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 'allow_dangerous_deserialization=True'는 신뢰할 수 있는 로컬 파일일 경우에만 사용합니다.
new_db = FAISS.load_local(
    "my_company_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 2. 검색기(Retriever)로 변환
retriever = new_db.as_retriever(search_kwargs={"k": 2})

# 3. LLM 설정 (Gemini 2.0 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# 4. 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 참고하여 질문에 답변해주세요.
답변은 컨텍스트에 있는 정보만 사용하세요.

컨텍스트:
{context}

질문: {question}

답변:
""")

# 5. 문서 포맷팅 함수
def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

# 6. LCEL 체인 구성
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 질문하기 함수
def ask_question(query):
    print(f"\n🙋 질문: {query}")
    result = rag_chain.invoke(query)
    print(f"🤖 답변: {result}")

# 테스트
if __name__ == "__main__":
    ask_question("재택근무 신청은 몇 시까지야?")
    ask_question("워케이션 규정 알려줘")