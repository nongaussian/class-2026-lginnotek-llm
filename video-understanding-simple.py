from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
import os

# API 키 설정
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# 모델 초기화
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

# 동영상 파일 읽기 및 인코딩
video_file_path = "https://github.com/nongaussian/class-2026-lginnotek-llm/raw/refs/heads/main/video_sample/human-accident_jam_rgb_1892_cctv3.mp4"
video_mime_type = "video/mp4"

# ============================================
# 1단계: 비디오 파일 -> 메시지 변환 함수
# ============================================
import requests

response = requests.get(video_file_path)
response.raise_for_status()
encoded_video = base64.b64encode(response.content).decode('utf-8')

# 6하원칙 기반 동영상 분석 프롬프트
VIDEO_ANALYSIS_PROMPT = """
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

# 메시지 생성
message = HumanMessage(
    content=[
        {"type": "text", "text": VIDEO_ANALYSIS_PROMPT},
        {"type": "media", "data": encoded_video, "mime_type": video_mime_type}
    ]
)

# 응답 생성
response = llm.invoke([message])
print(response.content)
