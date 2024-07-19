import time
import random
from pathlib import Path

import streamlit as st
import torch
import numpy as np
from api import gen
#from model import run

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="럭키비키 챗봇", page_icon="👦🏻")
st.header("럭키비키 챗봇", anchor="top", divider="rainbow")

# st.image(str(BASE_DIR.joinpath("assets", "boyfriend.jpeg")), width=200)


def seed_everything(seed):
    torch.manual_seed(seed)  # torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed)  # cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # 딥러닝에 특화된 CuDNN의 난수시드도 고정
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  # numpy를 사용할 경우 고정
    random.seed(seed)  # 파이썬 자체 모듈 random 모듈의 시드 고정


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] != "system":
            st.markdown(message["content"])

if prompt := st.chat_input("하고싶은 말을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    print(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
    
        seed_everything(42)
        stream = gen(prompt)
        with st.spinner("생성중...."):
            time.sleep(5)

        chunks = []
        for chunk in stream:
            chunks.append(chunk)
            message_placeholder.markdown("".join(chunks))
            time.sleep(0.02)
        #message_placeholder.markdown(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream})