import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_upstage import UpstageDocumentParseLoader, UpstageEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document

# === 환경 변수 로드 및 설정 ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# === Pinecone 설정 ===
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "boardgame-rag"
index = pc.Index(index_name)

# === PDF 목록 ===
pdf_list = [
    {"game": "메이플 벨리", "file": "pdfs/MapleValley.pdf", "namespace": "MapleValley"},
    {"game": "스플렌더", "file": "pdfs/Splendor.pdf", "namespace": "Splendor"},
]

# === 공통 설정 ===
text_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=32)
embedding = UpstageEmbeddings(model="solar-embedding-1-large")
batch_size = 20  # Pinecone 요청 크기 제한 대비

# === PDF 처리 루프 ===
for entry in pdf_list:
    game_name = entry["game"]
    file_path = entry["file"]
    namespace = entry["namespace"]

    print(f"\n📥 처리 중: {game_name} ({file_path})")

    try:
        # 0. 기존 namespace가 있을 경우 삭제
        existing_namespaces = index.describe_index_stats()["namespaces"].keys()
        if namespace in existing_namespaces:
            print(f"🧹 기존 데이터 삭제 중... ({namespace})")
            index.delete(delete_all=True, namespace=namespace)
            print("🗑️ 삭제 완료!")
        else:
            print(f"ℹ️ 삭제 생략: '{namespace}' namespace가 존재하지 않음")

        # 1. 문서 파싱
        loader = UpstageDocumentParseLoader(
            file_path=file_path,
            output_format='html',
            coordinates=False
        )
        parsed_docs = loader.load()

        # 2. 메타데이터 추가
        docs = [
            Document(page_content=doc.page_content, metadata={"game": game_name})
            for doc in parsed_docs
        ]

        # 3. 문서 분할
        chunks = text_splitter.split_documents(docs)

        # 4. Pinecone에 벡터 저장 (batch 처리)
        for i in range(0, len(chunks), batch_size):
            chunk_batch = chunks[i:i+batch_size]
            PineconeVectorStore.from_documents(
                documents=chunk_batch,
                embedding=embedding,
                index_name=index_name,
                namespace=namespace
            )

        print(f"✅ 저장 완료: {game_name} ({len(chunks)} chunks)")

    except Exception as e:
        print(f"❌ 오류 발생: {game_name} 처리 실패 - {str(e)}")

print("\n📦 모든 게임 룰이 Pinecone에 안전하게 다시 저장되었습니다!")
