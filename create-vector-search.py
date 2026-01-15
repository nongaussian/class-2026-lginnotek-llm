# Vertex AI Vector Search Index와 Endpoint를 생성하는 스크립트
# 주의: 실행하면 약 1-2시간 소요됩니다

from google.cloud import aiplatform
from google.cloud import storage
import time

# ==========================================
# 설정
# ==========================================
PROJECT_ID = "project-2a5393c8-4c80-45af-ae9"
LOCATION = "us-central1"
BUCKET_NAME = "my-rag-vectors-bucket"  # 전역적으로 고유한 이름으로 변경
INDEX_DISPLAY_NAME = "my-rag-index"
ENDPOINT_DISPLAY_NAME = "my-rag-endpoint"
DIMENSIONS = 768  # text-embedding-004의 차원

# ==========================================
# 1. GCS 버킷 생성
# ==========================================
print("📦 GCS 버킷 생성 중...")
try:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)

    if not bucket.exists():
        bucket = storage_client.create_bucket(
            BUCKET_NAME,
            location=LOCATION
        )
        print(f"✅ 버킷 생성 완료: gs://{BUCKET_NAME}")
    else:
        print(f"ℹ️  버킷이 이미 존재합니다: gs://{BUCKET_NAME}")
except Exception as e:
    print(f"❌ 버킷 생성 실패: {e}")
    print("💡 버킷 이름을 변경하거나 Console에서 수동으로 생성하세요")
    exit(1)

# Vertex AI 초기화
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ==========================================
# 2. Vector Search Index 생성
# ==========================================
print("\n🔍 Vector Search Index 생성 중... (약 30-60분 소요)")
print("⏳ 이 과정은 백그라운드에서 진행되며, 콘솔에서 진행 상황을 확인할 수 있습니다")

try:
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        contents_delta_uri=f"gs://{BUCKET_NAME}/initial",  # 초기 더미 경로
        dimensions=DIMENSIONS,
        approximate_neighbors_count=10,
        distance_measure_type="DOT_PRODUCT_DISTANCE",  # 또는 "COSINE_DISTANCE"
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=7,
        description="RAG를 위한 Vector Search Index",
    )

    print(f"✅ Index 생성 완료!")
    print(f"   Index ID: {index.resource_name.split('/')[-1]}")
    print(f"   Index Name: {index.display_name}")

    INDEX_ID = index.resource_name.split('/')[-1]

except Exception as e:
    print(f"❌ Index 생성 실패: {e}")
    exit(1)

# ==========================================
# 3. Endpoint 생성
# ==========================================
print("\n🌐 Endpoint 생성 중... (약 10-20분 소요)")

try:
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        description="RAG를 위한 Vector Search Endpoint",
        public_endpoint_enabled=True,
    )

    print(f"✅ Endpoint 생성 완료!")
    print(f"   Endpoint ID: {endpoint.resource_name.split('/')[-1]}")
    print(f"   Endpoint Name: {endpoint.display_name}")

    ENDPOINT_ID = endpoint.resource_name.split('/')[-1]

except Exception as e:
    print(f"❌ Endpoint 생성 실패: {e}")
    exit(1)

# ==========================================
# 4. Index를 Endpoint에 배포
# ==========================================
print("\n🚀 Index를 Endpoint에 배포 중... (약 20-30분 소요)")
print("⏳ 배포가 완료될 때까지 기다려주세요...")

try:
    endpoint.deploy_index(
        index=index,
        deployed_index_id=f"deployed_{INDEX_DISPLAY_NAME}",
        display_name=f"Deployed {INDEX_DISPLAY_NAME}",
        machine_type="e2-standard-2",  # 실습용, 프로덕션은 e2-standard-16 이상 권장
        min_replica_count=1,
        max_replica_count=1,
    )

    print("✅ 배포 완료!")

except Exception as e:
    print(f"❌ 배포 실패: {e}")
    print("💡 Console에서 수동으로 배포를 진행하세요")

# ==========================================
# 5. 결과 출력
# ==========================================
print("\n" + "="*60)
print("🎉 Vector Search 설정 완료!")
print("="*60)
print("\n다음 정보를 rag-precedent.py 파일에 입력하세요:\n")
print(f"PROJECT_ID = \"{PROJECT_ID}\"")
print(f"LOCATION = \"{LOCATION}\"")
print(f"INDEX_ID = \"{INDEX_ID}\"")
print(f"ENDPOINT_ID = \"{ENDPOINT_ID}\"")
print(f"\n# Option 2에서 사용할 버킷:")
print(f"gcs_bucket_name = \"{BUCKET_NAME}\"")
print("\n" + "="*60)
print("\n💡 Google Cloud Console에서 확인:")
print(f"   https://console.cloud.google.com/vertex-ai/matching-engine/indexes?project={PROJECT_ID}")
