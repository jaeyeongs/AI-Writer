from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from streaming import StreamHandler
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title('AI Writer')
st.write("---")

content = st.text_input('글의 주제를 제시해주세요')

if st.button('글짓기 요청하기'):
    with st.spinner('글 작성 중...'):
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box)

        # 객체 생성
        llm = ChatOpenAI(temperature=1,  # 창의성 (0.0 ~ 2.0)
                         max_tokens=512,  # 최대 토큰수
                         model_name='gpt-3.5-turbo',  # 모델명
                         streaming=True,  # streaming 설정
                         callbacks=[stream_handler]
                         )

        # 템플릿 정의
        template = '{content}에 대해 글을 작성해줘'

        # 템플릿 완성
        prompt = PromptTemplate(template=template, input_variables=['content'])

        # 연결된 체인(Chain)객체 생성
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        # llm_chain 실행
        result = llm_chain.run(content=content)
        st.write(result)
