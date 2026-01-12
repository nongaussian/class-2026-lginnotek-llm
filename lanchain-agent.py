import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool

# API 키 설정
os.environ["GOOGLE_API_KEY"] = "여기에_키_입력"

# 1. 도구(Tool) 정의
@tool
def get_weather(city: str) -> str:
    """도시의 날씨 정보를 가져옵니다."""
    # 실제로는 여기에서 기상청 Open API를 접근하는 등의
    # 방식으로 {city}의 날씨를 가져와서 답변을 반환
    weather = "맑음"
    return f"{city}의 날씨는 {weather}입니다!"

# 2. LLM 및 도구 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0
)

tools = [get_weather]

# 3. 에이전트 생성 (LangChain v1 최신 방식)
# create_agent는 LLM과 도구를 받아 자동으로 에이전트를 생성
agent = create_agent(
    llm,
    tools,
    system_prompt="당신은 도움이 되는 비서입니다. 날씨 정보가 필요하면 도구를 사용하세요."
)

# 4. 에이전트 실행
# LangGraph 에이전트는 messages 형식으로 입력
result = agent.invoke(
    {"messages": [("user", "지금 도쿄 시간 몇시야?")]}
)

# 결과 출력 (messages의 마지막 메시지가 최종 답변)
print(result["messages"][-1].content)