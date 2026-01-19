# !pip install langchain langchain-google-genai langchain-experimental pandas matplotlib

import pandas as pd
import numpy as np

# 가상의 제조 공정 데이터 생성
np.random.seed(42)
rows = 500
data = {
    'Timestamp': pd.date_range(start='2024-01-01', periods=rows, freq='H'),
    'Machine_ID': np.random.choice(['MC_A', 'MC_B'], rows),
    'Temperature': np.random.normal(150, 10, rows),
    'Pressure': np.random.normal(50, 5, rows),
    'Defect': np.random.choice([0, 1], rows, p=[0.95, 0.05])
}
df = pd.DataFrame(data)

# 교육용 이상치(Outlier) 강제 주입
df.loc[100, 'Temperature'] = 300  # 이상치


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# 1. 모델 설정 (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0, google_api_key="...")

# 2. 에이전트 생성 (여기가 핵심입니다)
# verbose=True를 설정하면 LLM의 사고 과정을 로그로 볼 수 있습니다.
# allow_dangerous_code=True는 로컬에서 코드 실행을 허용한다는 의미로, 최신 버전에서 필요합니다.
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
)


# 질문: 데이터의 기초 통계량 확인
response = agent.invoke("데이터의 요약 정보를 알려줘. 특히 결측치가 있는지 확인해줘.")
print(response['output'])