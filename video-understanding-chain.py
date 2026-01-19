# !pip install langchain langchain-core langchain-community langchain-google-genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import base64
import os
import requests

# API 키 설정
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# 모델 초기화
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

# 동영상 파일 읽기 및 인코딩
video_file_path = "https://github.com/nongaussian/class-2026-lginnotek-llm/raw/refs/heads/main/video_sample/human-accident_jam_rgb_1892_cctv3.mp4"
#video_file_path = "https://github.com/nongaussian/class-2026-lginnotek-llm/raw/refs/heads/main/video_sample/human-accident_jam_rgb_0957_cctv1.mp4"
#video_file_path = "https://github.com/nongaussian/class-2026-lginnotek-llm/raw/refs/heads/main/video_sample/generated_sample.mp4"
response = requests.get(video_file_path)
response.raise_for_status()
encoded_video = base64.b64encode(response.content).decode('utf-8')

# ============================================
# 1단계: 비디오 파일 -> 메시지 변환 함수
# ============================================
def create_video_message(inputs: dict) -> list:
    """비디오 파일을 읽어서 HumanMessage로 변환"""
    prompt = inputs["prompt"]
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "media", "data": encoded_video, "mime_type": "video/mp4"}
        ]
    )
    return [message]

# 6하원칙 기반 동영상 분석 프롬프트
initial_prompt = """
당신은 안전관리자입니다. 이 동영상을 분석하여 6하원칙에 따라 상황을 설명해주세요.

### 1. 누가 (Who)
- 작업자 수와 각자의 역할
- 보호장비 착용 여부 (안전모, 안전화, 보안경 등)

### 2. 언제 (When)
- 작업 시간대 추정
- 공정 단계 (준비/진행/마무리)

### 3. 어디서 (Where)
- 작업 구역 및 라인 특성
- 주변 장비 및 설비 현황

### 4. 무엇을 (What)
- 수행 중인 작업 내용
- 취급 중인 부품/자재/제품
- 사용 중인 작업자의 신체 부위

### 5. 어떻게 (How)
- 작업 절차 및 방법
- 사용 장비 및 도구

### 6. 왜 (Why)
- 작업의 목적
- 해당 공정의 역할
"""

second_prompt = """
당신은 공장의 안전관리자입니다. 작업자의 동작을 폐쇄회로 카메라로 모니터링하고 있는 도중 다음과 같은 상황이 있음을 발견했습니다. 
다음 상황에서 발생할 수 있는 사고의 유형을 다음과 같은 형식을 지켜 나열하세요.

예)
사고유형: 추락
발생상황: 작업자가 작업대에 올라갔다가 발을 헛딛고 넘어짐

사고유형: 끼임
발생상황: 작업자가 롤러에 손을 넣고 빼려고 힘을 쓰고 있음

[현재 상황]
{analysis}
"""
second_prompt_template = ChatPromptTemplate.from_template(second_prompt)


third_prompt = """
당신은 안전관리자입니다. 작업자의 동작을 폐쇄회로 카메라로 모니터링하고 있는 도중 다음과 같은 상황이 있음을 발견했습니다. 
이 동영상에서 다음 나열한 사고 중 하나가 발생했는지 판단하세요. 
**단, 다음 위험 요소가 있거나 사고가 발생할 가능성을 기술하는 것이 아니라 실제 이러한 사고가 영상에서 발생했는지를 판단하시오.**

답변에서 **사고유형** 및 해당 사고가 발생했다고 반단하게 된 **발생상황**을 명시적으로 밝히세요.
[가능한 사고 유형]
{possible_dangers}
"""
third_prompt_template = ChatPromptTemplate.from_template(third_prompt)

# ============================================
# 3단계: LCEL 체인 구성
# ============================================

chain1 = (
    # Step 1: 비디오 -> 메시지 -> LLM -> 상황인식 (6하원칙 분석)
    RunnableLambda(create_video_message) # invoke의 입력을 lambda 함수의 입력으로 전달해 반환
    | llm # 반환된 [HumanMessage]가 llm에 전달
    | StrOutputParser() # llm이 출력한 메시지를 그대로 문자열로 패스
)

chain2 = (
    # Step 2: 상황인식 결과 -> 가능한 위험 요소 -> 최종 프롬프트 생성
    second_prompt_template # 앞에서 패스한 딕셔너리를 프롬프트 생성기에 전달
    | llm
    | StrOutputParser()
    | {"possible_dangers": RunnablePassthrough()}
    | third_prompt_template
    | (lambda x: x.to_string())  # ChatPromptValue -> string 변환
)

chain3 = (
    # Step 3: 가능한 위험 요소 -> 실제 발생 여부 판단
    RunnableLambda(create_video_message)
    | llm
    | StrOutputParser()
)

# ============================================
# 실행
# ============================================
result1 = chain1.invoke({
    "prompt": initial_prompt
})
print("="*30)
print(result1)
print()

result2 = chain2.invoke({
    "analysis": result1
})
print("="*30)
print(result2)
print()

result3 = chain3.invoke({
    "prompt": result2
})
print("="*30)
print(result3)
print()


# ============================================
# 체인의 체인
# ============================================
combined = (
    RunnableLambda(lambda _: {"prompt": initial_prompt})
    | chain1
    | {"analysis": RunnablePassthrough()}
    | chain2
    | {"prompt": RunnablePassthrough()}
    | chain3
)

print(combined.invoke({}))