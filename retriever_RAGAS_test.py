from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from ragas.evaluation import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from datasets import Dataset
import pandas as pd
import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# 환경변수 가져오기
openai_key = os.getenv("OPENAI_API_KEY")
upstage_key = os.getenv("UPSTAGE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

# 환경 변수 설정
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["UPSTAGE_API_KEY"] = upstage_key
os.environ["PINECONE_API_KEY"] = pinecone_key

# 모델 및 벡터스토어
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embedding = UpstageEmbeddings(model="solar-embedding-1-large")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="boardgame-rag",
    embedding=embedding,
    namespace="katan"
)

# 평가용 질문
questions = [
    "카탄에서 도로를 놓을 수 있는 조건은?",
    "도시를 건설하려면 무엇이 필요한가요?",
    "강도는 어떤 상황에서 이동하나요?",
    "개척지를 어디에 지을 수 있나요?",
    "자원 교환은 어떻게 이루어지나요?"
]

ground_truths = [
    "자신의 기존 도로나 정착지에 연결되어야 도로를 놓을 수 있습니다.",
    "도시는 기존 정착지 위에 건설하며, 자원 카드 3개와 곡물 카드 2장이 필요합니다.",
    "플레이어가 7을 굴리면 강도가 이동합니다.",
    "다른 개척지나 도시와 일정 거리를 두고 지어야 합니다.",
    "자원 교환은 항구 또는 다른 플레이어와의 협상을 통해 이루어집니다."
]

# 평가용 데이터 구성
eval_data = {
    "question": [],
    "ground_truth": [],
    "answer": [],
    "contexts": []
}

# 질문별 평가용 문맥/응답 생성
for q, gt in zip(questions, ground_truths):
    # ✅ MultiQueryRetriever 적용
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        llm=llm
    )
    docs = retriever.invoke(q)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 보드게임 룰을 설명하는 AI야. 문서를 기반으로 정확하게 존댓말로 설명해. 사용자가 보내준 문서에 있는 내용으로만 답변하고, 내용이 없다면 잘 모르겠다고 말해."),
        ("human", f"다음 문서를 참고해서 질문에 답변해줘:\n\n{context}\n\n질문: {q}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({})

    eval_data["question"].append(q)
    eval_data["ground_truth"].append(gt)
    eval_data["answer"].append(answer)
    eval_data["contexts"].append([doc.page_content for doc in docs])

# RAGAS 평가 실행
dataset = Dataset.from_dict(eval_data)
results = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    ]
)

# 평가 결과 CSV 저장
df = results.to_pandas()
df.to_csv("ragas_MultiQuery_result.csv", index=False)
print("✅ 평가 결과가 ragas_MultiQuery_result.csv 에 저장되었습니다!")
print(df)





# Retriever 유형에 따른 RAGAS 평가 지표 성능을 비교해 보려고 하는 코드입니다.
# dense, MMR, MultiQuery 방식을 따로 실행시켜야 합니다.
# 방식 별로 코드는 각각 변경해야 합니다.