import streamlit as st
import base64
import requests
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit_authenticator as stauth
import faiss
import numpy as np
import os
import yaml
from openai import OpenAI
from yaml.loader import SafeLoader
from string import Template

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
model_name = "gpt-4o"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def save_uploaded_files(directory, files):
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 파일이 하나인 경우와 여러 개인 경우를 모두 처리
    if isinstance(files, list):  # 파일이 리스트인 경우, 여러 파일 처리
        file_paths = []
        for file in files:
            file_path = os.path.join(directory, file.name)
            with open(file_path, 'wb') as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
        return file_paths
    else:  # 단일 파일 처리
        file_path = os.path.join(directory, files.name)
        with open(file_path, 'wb') as f:
            f.write(files.getbuffer())
        return file_path

def extract_text(image_path):
    query = '다음 그림은 SAP에서 에러메시지 표출 화면입니다. 화면에서 답변없이 에러 메시지만 표출해주세요'
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}" 
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    response_text = response_json['choices'][0]['message']['content']

    return response_text

# 엑셀 파일 불러오기
def load_excel(file_pth):
    df = pd.read_excel(file_pth)
    return df

# 데이터 전처리 및 임베딩
def preprocess_and_embed(df, question_column):
    # 텍스트 데이터를 문자열로 변환하고 결측값을 빈 문자열로 대체
    df[question_column] = df[question_column].astype(str).fillna('')
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    question_embeddings = model.encode(df[question_column].tolist(), convert_to_tensor=True)
    return model, question_embeddings

system_prompt = Template("""
너는 한국의 자동화 회사인 현대자동차 재무경제팀에 소속된 인공지능 챗봇이고,
연구개발팀 연구원들이 진행하는 재경관련업무인 전표처리부터 고정자산 업무에 대한 처리 방법을 알려주는 것이 너의 업무야.
그 중에서 많은 질문이 나왔던 FAQ(Frequently Asked Questions)을 참고하여서 사용자의 질문에 친절하게 답변해줘.

구분의 속성은 5개이고, 단답형 질의, 프로세스/처리방법 질의, 판단을 요하는 질의, 시스템 에러에 대한 질의, 권한 부여/변경 요청으로 구성되어 있어.
단답형 질의 : 질문에 단순한 내용의 답변이 필요한 유형. FAQ 정리
프로세스/처리방법 질의 : 해당 업무를 어떤 순서로 어떻게 처리할지 문의하는 유형. 매뉴얼, 업무표준 등 처리방법 제시
판단을 요하는 질의 : 해당 비용을 어떻게 처리해야 할지 판단이 필요한 유형. 처리 기준 제시. 답변 부족시 담당자 연락
"시스템 에러에 대한 질의 : 시스템상 발생한 에러 내용의 해결방법을 문의하는 유형
(SAP 이미지를 캡처하여 질의하는 경우 다수). 해결방법 제시 (이미지 해석할 수 있는 LLM 필요)"
권한 부여/변경 요청 : 재경실 담당자가 시스템 권한을 부여해 줘야 하는 유형. 담당자 조치가 필요하여 챗봇이 아닌 담당자 호출로 답변

주요 질문 및 답변
$terms
""")

prompt = Template("""
다음 사용자의 요청에 대한 답변을 해줘

요청: $query
""")
client = OpenAI(
    api_key=OPENAI_API_KEY,
)
def create_answer(query):
  chat_completion = client.chat.completions.create(
      messages=[
          {
              "role": "system",
              "content": system_prompt_text,
          },
          {
              "role": "user",
              "content": prompt_h,
          }
      ],
      model=model_name,
  )

  return chat_completion.choices[0].message.content

system_prompt_text = system_prompt.substitute(terms="잘 답변부탁드립니다.")
prompt_h = prompt.substitute(query=prompt)

# FAISS 벡터화
def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)  # L2 distance
    index.add(np.array(embeddings))  # Add vectors to the index
    return index

# 에러메시지-답변 시스템
def answer_question(question, model, index, df, question_column, answer_column, top_k=1):
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), top_k)
    answers = df.iloc[I[0]][answer_column].tolist()
    return answers

# 파일 경로 및 텍스트 컬럼명 설정
file_pth = 'testdata.xlsx'  # 엑셀 파일 경로를 입력하세요
question_column = '에러메세지'    # 질문 텍스트 컬럼명을 입력하세요
answer_column = '답변'      # 답변 텍스트 컬럼명을 입력하세요

# 엑셀 파일 불러오기
df = load_excel(file_pth)

# 데이터 전처리 및 임베딩
model, question_embeddings = preprocess_and_embed(df, question_column)

# FAISS 벡터화
index = create_faiss_index(question_embeddings)

def get_answer(extracted_text):
    answer= answer_question(extracted_text,model, index, df, question_column,answer_column, top_k=1)
    return answer

def handle_image_and_provide_solution(image, excel_path, question_column, answer_column):
    
    image_path = save_uploaded_files("uploaded_images", [image])[0]
    extracted_text = extract_text(image_path)
    
    df = load_excel(excel_path)
    model, question_embeddings = preprocess_and_embed(df, question_column)
    index = create_faiss_index(question_embeddings)
    
    answer = get_answer(extracted_text, model, index, df, question_column, answer_column)
    return answer

def main():
    st.set_page_config(page_title="SAP 에러메세지 질의응답", page_icon=":robot:")

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    authenticator.login()
    
    if st.session_state["authentication_status"]:
        
        st.header("KPMG 챗봇")

        with st.sidebar:
            st.write(f'Welcome *{st.session_state["name"]}*')
            authenticator.logout()
            uploaded_files = st.file_uploader("dataset", type=['png', 'jpg'], accept_multiple_files=True)
            if uploaded_files:
                saved_file_paths = save_uploaded_files('dataset', uploaded_files)
                for file, path in zip(uploaded_files, saved_file_paths):
                    st.success(f"'{file.name}' 파일이 성공적으로 업로드 되었습니다.")
                    image = Image.open(file)
                    st.image(image, caption=f'Uploaded Image: {file.name}', use_column_width=True)

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            # 이미지가 업로드되었는지 확인
            if uploaded_files:
                try:
                    extracted_text = extract_text(path)
                    search_results = get_answer(extracted_text)
                    results = f"에러메세지: {extracted_text}\n\n해결방법: \n{search_results}"
                    converted_text = results.replace("\\n", "\n")
                    answer=converted_text.replace("[", "").replace("]", "").replace("'''", "").replace("\"\"\"", "")

                    
                except Exception as e:
                    answer = f"이미지 처리 중 오류 발생: {e}"
            else:
                # 이미지가 업로드되지 않았을 경우 일반 LLM을 사용한 답변
                response = create_answer(user_prompt)
                answer = response

            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')

if __name__ == '__main__':
    main()
