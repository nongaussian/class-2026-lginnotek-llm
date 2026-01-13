import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. API 키 설정 (OpenAI 키 필요)
os.environ = "sk-..."  # 여기에 OpenAI API Key 입력

# 2. 상태(State) 정의: 노드 간에 주고받을 데이터 구조
class EmailState(TypedDict):
    email_content: str      # 원본 이메일 내용
    category: str           # 분류된 카테고리
    draft: str              # 생성된 답장 초안

# 3. 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. 노드 함수 정의 (에이전트의 행동)

# [Node 1] 이메일 분류
def classify_email(state: EmailState):
    print("--- 1. 이메일 분류 중 ---")
    prompt = ChatPromptTemplate.from_template(
        "다음 이메일을 분석하여 'SPAM', 'REFUND', 'INQUIRY' 중 하나로 분류해줘. 오직 단어 하나만 출력해.\n\n이메일: {email}"
    )
    chain = prompt | llm
    category = chain.invoke({"email": state["email_content"]}).content.strip()
    print(f"분류 결과: {category}")
    return {"category": category}

# [Node 2] 답장 초안 작성
def draft_reply(state: EmailState):
    print("--- 2. 답장 초안 작성 중 ---")
    category = state["category"]
    email = state["email_content"]
    
    if category == "SPAM":
        return {"draft": "IGNORE"}
    
    prompt = ChatPromptTemplate.from_template(
        "당신은 친절한 고객 지원 AI입니다. 다음 {category} 유형의 이메일에 대한 정중한 답장을 작성해주세요.\n\n원문: {email}"
    )
    chain = prompt | llm
    draft = chain.invoke({"category": category, "email": email}).content
    return {"draft": draft}

# 5. 그래프(Workflow) 구성
workflow = StateGraph(EmailState)

# 노드 추가
workflow.add_node("classifier", classify_email)
workflow.add_node("drafter", draft_reply)

# 엣지(흐름) 연결
workflow.set_entry_point("classifier") # 시작점
workflow.add_edge("classifier", "drafter") # 분류 -> 작성
workflow.add_edge("drafter", END) # 작성 -> 종료

# 컴파일
app = workflow.compile()

# 6. 실행 및 테스트
test_email = """
안녕하세요, 지난주에 구매한 커피 머신이 작동하지 않습니다. 
전원이 켜지지 않아요. 환불이나 교환을 받고 싶습니다.
주문 번호는 12345입니다.
"""

print(f"입력 이메일: {test_email}")
result = app.invoke({"email_content": test_email})

print("\n================ 최종 결과 ================")
print(f"카테고리: {result['category']}")
print(f"생성된 초안:\n{result['draft']}")