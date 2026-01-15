from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# API 키 설정
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# 1. 저장된 판례 인덱스 불러오기
# rag-build-index.py와 동일한 임베딩 모델 사용
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 'allow_dangerous_deserialization=True'는 신뢰할 수 있는 로컬 파일일 경우에만 사용합니다.
new_db = FAISS.load_local(
    "precedent_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 2. 검색기(Retriever)로 변환 - 유사한 판례 3개 검색
retriever = new_db.as_retriever(search_kwargs={"k": 3})

# 3. LLM 설정 (Gemini 2.0 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)

# 4. 프롬프트 템플릿 정의 (법률 상담용)
prompt = ChatPromptTemplate.from_template("""
당신은 법률 전문가입니다. 아래 제공된 판례들을 참고하여 질문에 답변해주세요.

[참고 판례]
{context}

[질문]
{question}

[답변 지침]
1. 참고 판례의 사실관계와 질문의 상황을 비교 분석하세요.
2. 관련 판례의 법원 판단을 근거로 답변하세요.
3. 참고한 판례의 사건번호를 명시하세요.
4. 실제 법적 조언이 아닌 참고용 정보임을 명시하세요.

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
    #ask_question("A는 B의 배우자인데 B가 회사를 다니다 간경화 진단을 받아 일을 더이상 못하고 1년전에 퇴사를 하였고 6개월전에 간세포암 진단을 받아 3개월전에 사망하였다. A는 직장 내 괴롭힘, 따돌림, 폭언, 폭행, 사직 강요 등은 없었던 것으로 판명되었다. A는 산업재해보상을 받을수 있을까?")
    ask_question("A는 아주 유명한 쿠키 제과점 B의 상표가 아직 상표등록이 안된 것을 알고 자기가 먼저 상표 등록을 하려고 했고, 이를 B가 우연히 알아내고 상표 등록 취하 소송을 했어. 그리고 B는 A에게 저작권 침해 손해배상까지 걸었어. B는 소송에서 이길 수 있을까?")
    #ask_question("2020허4570")