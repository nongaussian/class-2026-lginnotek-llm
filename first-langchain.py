# 1. 라이브러리 임포트 및 API 키 설정
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 실제 환경에서는 .env 파일이나 환경변수로 관리하세요
os.environ["OPENAI_API_KEY"] = "your_api_key"

# 2. 모델 초기화
llm = OpenAI(temperature=0.7)

# 3. 프롬프트 템플릿 생성
# {topic} 변수를 동적으로 입력받습니다.
template = "{topic}에 대해 배울 수 있는 좋은 책 3권을 추천해줘."
prompt_template = PromptTemplate.from_template(template)

# 4. LCEL을 사용한 체인 구성 (파이프 연산자 사용)
# 프롬프트 -> LLM -> 문자열 출력 파서
chain = prompt_template | llm | StrOutputParser()

# 5. 체인 실행
response = chain.invoke({"topic": "인공지능"})
print(response)