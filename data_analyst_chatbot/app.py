import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_experimental.tools import PythonREPLTool
import tempfile

# 페이지 설정
st.set_page_config(
    page_title="데이터 분석 챗봇",
    page_icon="📊",
    layout="wide"
)

st.title("📊 데이터 분석 챗봇")
st.markdown("CSV 파일을 업로드하고 자연어로 분석을 요청하세요!")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None

# 사이드바: API 키 및 파일 업로드
with st.sidebar:
    st.header("⚙️ 설정")
    
    # Google API 키 입력
    api_key = st.text_input("Google API Key", type="password",
                            help="Google API 키를 입력하세요")
    
    st.divider()
    
    # CSV 파일 업로드
    st.header("📁 데이터 업로드")
    uploaded_file = st.file_uploader("CSV 파일 선택", type=['csv'])
    
    if uploaded_file is not None:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.csv_path = tmp_file.name
        
        # DataFrame 로드
        st.session_state.df = pd.read_csv(st.session_state.csv_path)
        st.success(f"✅ 파일 로드 완료: {uploaded_file.name}")
        
        # 데이터 미리보기
        st.subheader("데이터 미리보기")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        # 데이터 정보
        st.subheader("데이터 정보")
        st.write(f"- 행: {len(st.session_state.df):,}개")
        st.write(f"- 열: {len(st.session_state.df.columns)}개")
        st.write(f"- 컬럼: {', '.join(st.session_state.df.columns.tolist())}")

# 시스템 프롬프트 생성 함수
def get_system_prompt(csv_path: str) -> str:
    return f"""당신은 전문 데이터 분석가입니다. 사용자의 요청에 따라 Python 코드를 작성하고 실행하여 데이터를 분석합니다.

분석할 CSV 파일 경로: {csv_path}

중요한 규칙:
1. 항상 pandas를 사용하여 데이터를 분석하세요
2. 데이터 로드: df = pd.read_csv("{csv_path}")
3. 시각화가 필요하면 matplotlib 또는 seaborn을 사용하고, plt.savefig('output.png')로 저장 후 print("차트가 output.png에 저장되었습니다")를 출력하세요
4. 결과는 항상 print()로 출력하세요
5. 한국어로 친절하게 설명하세요
6. 코드 실행 후 결과를 해석해서 설명하세요"""


# 에이전트 생성 함수
def create_data_analyst_agent(api_key: str, csv_path: str):
    """데이터 분석 에이전트 생성"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=api_key
    )

    # PythonREPLTool 설정
    python_repl = PythonREPLTool()
    tools = [python_repl]

    # 시스템 프롬프트
    system_prompt = get_system_prompt(csv_path)

    # 에이전트 생성 (LangChain v1 방식)
    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt
    )

    return agent

# 예시 질문 (사이드바에 추가)
with st.sidebar:
    st.divider()
    st.subheader("📋 예시 질문")
    st.markdown("""
    데이터가 로드되면 다음과 같은 질문을 해보세요:

    - "데이터의 기본 통계를 보여줘"
    - "결측치가 있는지 확인해줘"
    - "각 컬럼의 데이터 타입을 알려줘"
    - "특정 컬럼의 분포를 시각화해줘"
    - "두 변수 간의 상관관계를 분석해줘"
    - "데이터를 그룹별로 집계해줘"
    """)

    # 채팅 초기화 버튼
    if st.button("🗑️ 채팅 기록 삭제"):
        st.session_state.messages = []
        st.rerun()

# 메인 채팅 영역
st.subheader("💬 채팅")

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "code" in message:
            with st.expander("🔍 실행된 코드 보기"):
                st.code(message["code"], language="python")

# 대기 중인 이미지 표시
if st.session_state.pending_image and os.path.exists(st.session_state.pending_image):
    st.image(st.session_state.pending_image)
    os.remove(st.session_state.pending_image)
    st.session_state.pending_image = None

# 사용자 입력 (화면 하단에 고정)
if prompt := st.chat_input("분석하고 싶은 내용을 입력하세요..."):
    # 입력 검증
    if not api_key:
        st.error("⚠️ Google API 키를 입력해주세요.")
    elif st.session_state.df is None:
        st.error("⚠️ 먼저 CSV 파일을 업로드해주세요.")
    else:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 에이전트 실행
        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                try:
                    # 에이전트 생성
                    agent = create_data_analyst_agent(api_key, st.session_state.csv_path)

                    # 대화 히스토리 구성 (최근 6개 메시지, 현재 입력 제외)
                    messages = []
                    for m in st.session_state.messages[-7:-1]:
                        if m["role"] == "user":
                            messages.append(("user", m["content"]))
                        else:
                            messages.append(("assistant", m["content"]))
                    # 현재 사용자 입력 추가
                    messages.append(("user", prompt))

                    # 에이전트 실행 (messages 형식)
                    result = agent.invoke({"messages": messages})

                    # 실행된 코드 추출
                    executed_codes = []
                    for msg in result.get("messages", []):
                        if hasattr(msg, "tool_calls"):
                            for tool_call in msg.tool_calls:
                                tool_input = tool_call.get("args", {}).get("query", "")
                                if tool_input:
                                    executed_codes.append(tool_input)

                    # 최종 응답 추출 (마지막 메시지)
                    final_messages = result.get("messages", [])
                    answer = final_messages[-1].content if final_messages else "분석을 완료하지 못했습니다."

                    # 이미지 표시 (시각화가 생성된 경우)
                    if os.path.exists("output.png"):
                        st.session_state.pending_image = "output.png"

                    # 메시지 저장
                    msg_data = {"role": "assistant", "content": answer}
                    if executed_codes:
                        msg_data["code"] = "\n\n".join(executed_codes)
                    st.session_state.messages.append(msg_data)

                    # 페이지 새로고침하여 히스토리에서 표시
                    st.rerun()

                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# 푸터
st.divider()
st.caption("Made with Streamlit + LangChain | 로컬 환경에서만 실행하세요")
