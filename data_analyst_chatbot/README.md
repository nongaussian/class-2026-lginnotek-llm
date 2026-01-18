# 📊 데이터 분석 챗봇

Streamlit + LangChain + PythonREPLTool을 활용한 CSV 데이터 분석 챗봇입니다.

## 🚀 설치 및 실행 방법

### 1단계: 프로젝트 폴더 생성 및 파일 저장

```bash
mkdir data_analyst_chatbot
cd data_analyst_chatbot
```

`app.py`와 `requirements.txt` 파일을 이 폴더에 저장합니다.

### 2단계: 가상환경 생성 (권장)

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3단계: 패키지 설치

```bash
pip install -r requirements.txt
```

### 4단계: 앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속됩니다.

## 📝 사용 방법

1. **API 키 입력**: 사이드바에 OpenAI API 키를 입력합니다
2. **CSV 업로드**: 분석할 CSV 파일을 업로드합니다
3. **질문하기**: 자연어로 분석을 요청합니다

## 💡 예시 질문

- "데이터의 기본 통계를 보여줘"
- "결측치가 있는지 확인해줘"
- "컬럼별 데이터 타입을 알려줘"
- "age 컬럼의 분포를 히스토그램으로 보여줘"
- "salary와 experience의 상관관계를 분석해줘"
- "department별 평균 salary를 계산해줘"

## ⚠️ 주의사항

- **로컬 전용**: 이 앱은 PythonREPLTool을 사용하여 Python 코드를 직접 실행합니다. 보안상 로컬 환경에서만 사용하세요.
- **API 비용**: OpenAI API 호출에 비용이 발생합니다.
- **민감한 데이터**: 민감한 데이터는 업로드하지 마세요.

## 🔧 문제 해결

### "ModuleNotFoundError" 발생 시
```bash
pip install --upgrade langchain langchain-openai langchain-experimental
```

### Streamlit 버전 충돌 시
```bash
pip install streamlit --upgrade
```

## 📁 프로젝트 구조

```
data_analyst_chatbot/
├── app.py              # 메인 애플리케이션
├── requirements.txt    # 의존성 파일
└── README.md          # 사용 설명서
```
