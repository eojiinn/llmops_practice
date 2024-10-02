#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langserve import add_routes

# 1. 프롬프트 템플릿 생성
system_template = "Translate the following into {language}:"
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
