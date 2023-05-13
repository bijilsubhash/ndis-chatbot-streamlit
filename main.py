import streamlit as st
from streamlit_chat import message

import time
import os
import sys
import csv
import ast

from langchain.document_loaders import CSVLoader
from langchain.llms import OpenAI
from langchain.chains import  RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

import pinecone
import openai


st.set_page_config(page_title="NDIS Chatbot", 
                   page_icon=":pixel-art-neutral:",
                   layout="centered", 
                   initial_sidebar_state="auto")
col1, col2 = st.columns([1,1])
col1.markdown(" # NDIS Chatbot ")
col1.markdown(" ##### Ask me anything about NDIS ")
col2.markdown(" <h2>This app is an LLM powered chatbot, built with</h2>
<h2><img alt="File:OpenAI Logo.svg - Wikimedia Commons" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/2560px-OpenAI_Logo.svg.png" style="font-size: 13px; width: 100px; height: 24px;" /><span style="font-size: 13px;">&nbsp; &nbsp; &nbsp;&nbsp;</span><img alt="Screenshot-2023-05-14-at-9-27-13-am" src="https://i.ibb.co/TPGScwC/Screenshot-2023-05-14-at-9-27-13-am.png" style="width: 130px; height: 32px;" />&nbsp;&nbsp;<img alt="Pinecone 2.0 is Available and Free | Pinecone" src="https://global.discourse-cdn.com/standard14/uploads/pinecone/original/1X/d8e002f5074a908faee547fc24a48e77dec727c4.png" style="font-size: 13px; width: 120px; height: 33px;" /></h2>
", unsafe_allow_html=True)

#configurations
csv.field_size_limit(sys.maxsize)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

def llm_chain(query):

    #loading the data
    loader = CSVLoader(file_path="content.csv", csv_args={"delimiter": "\n"})
    data = loader.load()

    #split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    #select which embeddings we want to use
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

    # initialize pinecone
    pinecone.init(api_key=PINECONE_API_KEY,  environment=PINECONE_ENV)
    index_name = "ndis-chatbot"
    db = Pinecone.from_existing_index(index_name, embeddings)

    #expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = """You are a conversational  NDIS expert with access to NDIS context. 
    You are informative and provides details from the context. If the query from the user lies outside the scope of NDIS context, please say you don't know the answer.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    #create a chain to answer questions
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(ChatOpenAI(model_name="gpt-3.5-turbo"), 
                                    chain_type="stuff", retriever=retriever, 
                                    chain_type_kwargs=chain_type_kwargs)

    result = qa({"query": query})
    return result['result']

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

placeholder = st.empty()

def get_text():
    with st.form(key='my_form', clear_on_submit=True):
        input_text = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    return input_text

user_input = get_text()

if user_input:
    with st.spinner('Wait for it...'):
        output = llm_chain(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
        time.sleep(0.03)


if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

