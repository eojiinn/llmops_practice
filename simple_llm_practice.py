#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langserve import add_routes

# 1. 프롬프트 템플릿 생성
system_template = """
'상황'에 대한 반응을 다음 예시와 같이 ['원영적 사고', '숙이적 사고', '은이적 사고']로 변환해줘.
'원영적 사고'의 맨 마지막에는 '완전 럭키비키잖앙'을 붙여주면 돼.

[예시]
상황: 츄러스를 먹으려고 줄을 서서 기다리다가 내 앞에서 마지막 츄러스를 사감.

원영적 사고: 내앞에서 다팔렷대 하지만 갓 나온 따뜻한 츄러스를 먹을 수 있으니까 완전 럭키비키잖앙
숙이적 사고: 내 앞에서 똑 떨어져? 재수가 드릅게 없네
은이적 사고: 새 츄러스가 나오려면 5분 정도 걸린다고 하고 옆 가게의 도너츠는 2분이면 먹을 수 있으니깐 도너츠를 먹어야겠다.
"""

prompt_template = ChatPromptTemplate.from_messages([('system', system_template), ('user', '{text}')])

# 2. 모델 설정
model = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs=dict(temperature=0),
    region_name='us-east-1'
)

# 3. 출력 파서 설정
parser = StrOutputParser()

# 4. 체인 생성
chain = prompt_template | model | parser

# 5. FastAPI 앱 정의
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 6. 체인 경로 추가
add_routes(app, chain, path="/chain")

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
