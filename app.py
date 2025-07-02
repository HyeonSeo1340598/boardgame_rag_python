from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# 환경변수 가져오기
openai_key = os.getenv("OPENAI_API_KEY")
upstage_key = os.getenv("UPSTAGE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

# 환경 변수 설정 (또는 .env로 관리 가능)
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["UPSTAGE_API_KEY"] = upstage_key
os.environ["PINECONE_API_KEY"] = pinecone_key

# FastAPI 앱 생성
app = FastAPI()

# ChatMessage 클래스 정의
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

# 요청 모델
class RagRequest(BaseModel):
    question: str
    game: str  # 예: "katan", "splendor"
    history: List[ChatMessage] = []  # 이전 대화 기록

# API 엔드포인트
@app.post("/api/rag")
def answer_rag(request: RagRequest):
    # 1. LLM 및 임베딩 모델
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="boardgame-rag",
        embedding=embedding,
        namespace=request.game
    )

    # 2. 관련 문서 검색
    docs = vectorstore.similarity_search(request.question, k=5)

    # # 2. 관련 문서 검색
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # docs = retriever.invoke(request.question)
    # context = "\n".join([doc.page_content for doc in docs])

    # print(context)

    # 3. 프롬프트 메시지 구성
    messages = [("system", 
        "너는 보드게임 룰을 설명해주는 AI야. 질문에 대해 친절하게 존댓말로 설명해줘. "
        "사용자가 보내준 문서에 있는 내용으로만 답변하고, 내용이 없다면 잘 모르겠다고 말해."
        "만약 사용자가 보내준 문서가 없으면 아는 범위 내로 대답해 주고 모르는 내용은 모르겠다고 답변해줘")]

    # history 반영
    for msg in request.history:
        messages.append((msg.role, msg.content))
    
    # 문서 기반 RAG 답변 vs 일반 LLM 답변
    if docs:
        context = "\n".join([doc.page_content for doc in docs])
        print(context)
        messages.append(("human", f"다음 문서를 참고해서 질문에 답변해줘.\n\n문서: {context}\n\n질문: {request.question}"))
    else:
        print(request.game)
        print("LLM만의 답변")  
        messages.append(("human", f"'{request.game}' 게임에 대한 문서는 없지만, 네가 아는 범위 내에서 답변해줘. "
                                  f"정확한 정보가 없으면 잘 모르겠다고 답변해야 해. 질문: {request.question}"))
        
    # # 마지막 질문 + context
    # messages.append(("human", f"다음 문서를 참고해서 질문에 답변해줘.\n\n문서: {context}\n\n질문: {request.question}"))

    # 프롬프트 체인 구성
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({})  # context, question 모두 메시지 안에 포함됨

    return {"answer": answer}


@app.get("/")
def root():
    return {"message": "🎲 Boardgame RAG API is running!"}

# 서버 실행 코드
# uvicorn rag_server:app --reload --port 8000