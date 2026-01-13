import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool

# API 키 설정
os.environ["GOOGLE_API_KEY"] = "여기에_키_입력"

# ============================================================
# 스마트홈 기기 제어 도구 정의
# ============================================================

# 현재 집안 상태 (시뮬레이션용)
home_state = {
    "lights": {"living_room": {"on": False, "brightness": 0, "color": "white"}},
    "temperature": {"current": 22, "target": 22, "mode": "off"},
    "curtains": {"living_room": "open"},
    "music": {"playing": False, "volume": 0},
    "tv": {"on": False, "channel": 1}
}

@tool
def control_lights(room: str, action: str, brightness: int = 50, color: str = "white") -> str:
    """
    조명을 제어합니다.

    Args:
        room: 방 이름 (예: living_room, bedroom)
        action: 'on' 또는 'off'
        brightness: 밝기 (0-100), action이 'on'일 때 사용
        color: 색상 (white, warm, cool, red, blue 등), action이 'on'일 때 사용

    Returns:
        조명 제어 결과 메시지
    """
    if room not in home_state["lights"]:
        home_state["lights"][room] = {"on": False, "brightness": 0, "color": "white"}

    if action == "on":
        home_state["lights"][room] = {"on": True, "brightness": brightness, "color": color}
        return f"{room} 조명을 켰습니다. 밝기: {brightness}%, 색상: {color}"
    elif action == "off":
        home_state["lights"][room]["on"] = False
        return f"{room} 조명을 껐습니다."
    else:
        return "action은 'on' 또는 'off'여야 합니다."

@tool
def control_temperature(action: str, target_temp: int = 22) -> str:
    """
    온도를 제어합니다 (에어컨/히터).

    Args:
        action: 'cool' (냉방), 'heat' (난방), 'off' (끄기)
        target_temp: 목표 온도 (°C)

    Returns:
        온도 제어 결과 메시지
    """
    current = home_state["temperature"]["current"]

    if action == "off":
        home_state["temperature"]["mode"] = "off"
        return f"냉난방을 껐습니다. 현재 온도: {current}°C"
    elif action in ["cool", "heat"]:
        home_state["temperature"]["mode"] = action
        home_state["temperature"]["target"] = target_temp
        return f"{action} 모드로 설정했습니다. 목표 온도: {target_temp}°C, 현재 온도: {current}°C"
    else:
        return "action은 'cool', 'heat', 또는 'off'여야 합니다."

@tool
def get_temperature() -> str:
    """
    현재 실내 온도를 확인합니다.

    Returns:
        현재 온도 정보
    """
    temp = home_state["temperature"]["current"]
    mode = home_state["temperature"]["mode"]
    target = home_state["temperature"]["target"]

    if mode == "off":
        return f"현재 온도: {temp}°C (냉난방 꺼짐)"
    else:
        return f"현재 온도: {temp}°C, 목표: {target}°C, 모드: {mode}"

@tool
def control_curtains(room: str, action: str) -> str:
    """
    커튼을 제어합니다.

    Args:
        room: 방 이름 (예: living_room, bedroom)
        action: 'open' (열기) 또는 'close' (닫기)

    Returns:
        커튼 제어 결과 메시지
    """
    if room not in home_state["curtains"]:
        home_state["curtains"][room] = "open"

    if action in ["open", "close"]:
        home_state["curtains"][room] = action
        return f"{room} 커튼을 {action}했습니다."
    else:
        return "action은 'open' 또는 'close'여야 합니다."

@tool
def control_music(action: str, volume: int = 30) -> str:
    """
    CD플레이어/스피커를 제어합니다.

    Args:
        action: 'play' (재생) 또는 'stop' (정지)
        volume: 볼륨 (0-100)

    Returns:
        CD플레이어 제어 결과 메시지
    """
    if action == "play":
        home_state["music"]["playing"] = True
        home_state["music"]["volume"] = volume
        return f"음악을 재생합니다. 볼륨: {volume}%"
    elif action == "stop":
        home_state["music"]["playing"] = False
        return "음악을 정지했습니다."
    else:
        return "action은 'play' 또는 'stop'이어야 합니다."

@tool
def control_tv(action: str, channel: int = 1) -> str:
    """
    TV를 제어합니다.

    Args:
        action: 'on' (켜기) 또는 'off' (끄기)
        channel: 채널 번호

    Returns:
        TV 제어 결과 메시지
    """
    if action == "on":
        home_state["tv"]["on"] = True
        home_state["tv"]["channel"] = channel
        return f"TV를 켰습니다. 채널: {channel}"
    elif action == "off":
        home_state["tv"]["on"] = False
        return "TV를 껐습니다."
    else:
        return "action은 'on' 또는 'off'여야 합니다."

@tool
def get_home_status() -> str:
    """
    현재 집안의 모든 기기 상태를 확인합니다.

    Returns:
        모든 기기의 현재 상태
    """
    status = "=== 현재 집안 상태 ===\n"

    status += "\n[조명]\n"
    for room, light in home_state["lights"].items():
        status += f"  {room}: {'켜짐' if light['on'] else '꺼짐'}"
        if light['on']:
            status += f" (밝기: {light['brightness']}%, 색상: {light['color']})"
        status += "\n"

    status += f"\n[온도]\n  {get_temperature()}\n"

    status += "\n[커튼]\n"
    for room, curtain in home_state["curtains"].items():
        status += f"  {room}: {curtain}\n"

    music = home_state["music"]
    status += f"\n[음악]\n  {'재생 중' if music['playing'] else '정지'}"
    if music['playing']:
        status += f" (볼륨: {music['volume']}%)"
    status += "\n"

    tv = home_state["tv"]
    status += f"\n[TV]\n  {'켜짐' if tv['on'] else '꺼짐'}"
    if tv['on']:
        status += f" (채널: {tv['channel']})"
    status += "\n"

    return status

# ============================================================
# LLM 및 에이전트 설정
# ============================================================

# LLM 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0
)

# 도구 목록
tools = [
    control_lights,
    control_temperature,
    get_temperature,
    control_curtains,
    control_music,
    control_tv,
    get_home_status
]

# 시스템 프롬프트 - 재귀적 추론을 유도
system_prompt = """[시스템 지시사항]
당신은 스마트 홈을 제어하는 AI 비서입니다. 사용자의 요청을 가장 잘 수행하기 위해 오직 아래의 도구들만을 사용할 수 있습니다:

1. control_temperature (온도조절기): 설정 온도를 변경할 때 사용.
2. get_temperature (온도센서): 현재 온도를 확인하기 위해 사용.
3. control_curtains (커튼): 커튼을 제어하기 위해 사용.
4. control_music (음악): CD플레이어와 스피커를 제어하는데 사용.
5. control_lights (조명): 조명을 켜거나 끄고, 밝기를 조절할 때 사용.
6. control_tv (TV): TV를 켜거나 특정 앱을 실행할 때 사용.
7. get_home_status: 집안의 모든 기기 상태를 확인할 때 사용.

응답할 때는 반드시 다음 형식을 엄격하게 지키세요:

Question: 사용자의 입력 질문/명령
Thought: 질문데 답하거나 명령을 수행하기 위해 무엇을 해야 할지 스스로 생각하세요. (이 단계에서 상황을 판단합니다)
Action: 수행할 행동. 반드시 [Thermostat, Smart_Light, Smart_TV] 중 하나여야 합니다.
Action Input: 행동에 들어갈 입력값
Observation: 행동의 결과 (이 값은 도구 실행 후 시스템이 채워줍니다)
... (필요한 만큼 Thought/Action/Observation 반복) ...
Final Answer: 최종 답변

[대화 시작]
Question: {사용자의 입력 질문/명령}
"""

# 에이전트 생성 (LangChain v1 방식)
# create_agent는 LLM과 도구를 받아 자동으로 LangGraph 기반 에이전트를 생성
agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt
)

# ============================================================
# 실행 예제
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("스마트홈 AI 에이전트 - 재귀적 추론 예제")
    print("=" * 60)

    # 예제 1: 조건부 연쇄 작업 - 영화 보기 모드
    print("\n\n[예제 1] 영화 볼 준비해줘")
    print("-" * 60)
    result1 = agent.invoke({
        "messages": [("user", "영화 볼 준비해줘. 조명은 어둡게, 커튼 닫고, 온도 확인해서 필요하면 조절하고, TV 켜줘.")]
    })
    print("\n최종 결과:", result1["messages"][-1].content)

    # 예제 2: 상태 확인 후 조건부 실행
    print("\n\n[예제 2] 잘 준비해줘")
    print("-" * 60)
    result2 = agent.invoke({
        "messages": [("user", "잘 준비해줘. 모든 조명과 TV 끄고, 음악도 끄고, 온도는 22도로 맞춰줘.")]
    })
    print("\n최종 결과:", result2["messages"][-1].content)

    # 예제 3: 현재 상태 확인
    print("\n\n[예제 3] 집안 상태 확인")
    print("-" * 60)
    result3 = agent.invoke({
        "messages": [("user", "지금 집안 상태 알려줘")]
    })
    print("\n최종 결과:", result3["messages"][-1].content)

    # 예제 4: 아침 루틴
    print("\n\n[예제 4] 아침 시작해줘")
    print("-" * 60)
    result4 = agent.invoke({
        "messages": [("user", "아침 시작해줘. 조명 켜고, 커튼 열고, 상쾌한 음악 틀어줘. 온도도 확인해서 쾌적하게 해줘.")]
    })
    print("\n최종 결과:", result4["messages"][-1].content)