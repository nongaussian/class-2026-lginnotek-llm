# !pip install langchain langchain-community langchain-google-vertexai faiss-cpu requests

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import json
import glob
import requests

# API 키 설정
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# GitHub에서 precedent_sample 디렉토리의 JSON 파일들 다운로드
GITHUB_API_URL = "https://api.github.com/repos/nongaussian/class-2026-lginnotek-llm/contents/precedent_sample"
RAW_BASE_URL = "https://raw.githubusercontent.com/nongaussian/class-2026-lginnotek-llm/main/precedent_sample"
LOCAL_DIR = "precedent_sample"

# 로컬 디렉토리 생성 (없으면)
os.makedirs(LOCAL_DIR, exist_ok=True)
print(f"📁 '{LOCAL_DIR}' 디렉토리 준비 완료")

# GitHub API로 파일 목록 가져오기
print("🔍 GitHub에서 파일 목록을 가져오는 중...")
response = requests.get(GITHUB_API_URL)
if response.status_code != 200:
    raise Exception(f"GitHub API 요청 실패: {response.status_code}")

files = response.json()
json_files_info = [f for f in files if f["name"].endswith(".json")]
print(f"📂 GitHub에서 {len(json_files_info)}개의 JSON 파일을 발견했습니다.")

# 각 JSON 파일 다운로드
for file_info in json_files_info:
    file_name = file_info["name"]
    local_path = os.path.join(LOCAL_DIR, file_name)

    # 파일 다운로드
    raw_url = f"{RAW_BASE_URL}/{file_name}"
    file_response = requests.get(raw_url)

    if file_response.status_code == 200:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(file_response.text)
        print(f"  ✓ {file_name} 다운로드 완료")
    else:
        print(f"  ✗ {file_name} 다운로드 실패: {file_response.status_code}")

# 로컬 디렉토리에서 JSON 파일들 로드
json_files = glob.glob(f"{LOCAL_DIR}/*.json")
print(f"\n📂 {len(json_files)}개의 JSON 파일을 로드합니다.")

docs = []
for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 판례 정보 추출
    info = data.get("info", {})
    case_no = info.get("caseNo", "")
    case_nm = info.get("caseNm", "")
    court_nm = info.get("courtNm", "")
    judmn_date = info.get("judmnAdjuDe", "")

    # 본문 내용 구성 (사실관계 + 법원 판단 + 결론)
    facts = data.get("facts", {}).get("bsisFacts", [])
    dcss = data.get("dcss", {}).get("courtDcss", [])
    cnclsns = data.get("close", {}).get("cnclsns", [])

    content = f"[사건번호: {case_no}] {case_nm}\n"
    content += f"법원: {court_nm} | 선고일: {judmn_date}\n\n"
    content += "[사실관계]\n" + "\n".join(facts) + "\n\n"
    content += "[법원 판단]\n" + "\n".join(dcss) + "\n\n"
    content += "[결론]\n" + "\n".join(cnclsns)

    # 메타데이터와 함께 Document 생성
    doc = Document(
        page_content=content,
        metadata={
            "case_no": case_no,
            "case_nm": case_nm,
            "court_nm": court_nm,
            "judmn_date": judmn_date,
            "source": file_path
        }
    )
    docs.append(doc)

print(f"📄 {len(docs)}개의 문서를 생성했습니다.")

# 2. 임베딩 모델 설정 (Google Generative AI)
# text-embedding-004는 Google의 최신 임베딩 모델입니다.
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

print("⏳ 문서 벡터화 진행 중...")

# 3. FAISS 인덱스 생성 (메모리 상에 구축)
db = FAISS.from_documents(docs, embeddings)

# 4. 로컬 디스크에 저장 (핵심!)
# 실행 경로에 'precedent_index'라는 폴더가 생성됩니다.
db.save_local("precedent_index")

print("✅ 인덱스 저장 완료! ('precedent_index' 폴더 확인)")